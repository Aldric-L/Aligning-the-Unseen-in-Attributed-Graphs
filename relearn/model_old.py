import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class GCNEncoder(nn.Module):
    """GCN Encoder for node features processing"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        return x

class EdgeEncoder(nn.Module):
    """Edge encoder for computing local relation factors"""
    def __init__(self, node_emb_dim, hidden_dim, z_dim, num_mixtures=5):
        super(EdgeEncoder, self).__init__()
        self.z_dim = z_dim
        self.num_mixtures = num_mixtures
        
        # Neural network for local relation factors
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Output mixture components (logits for Gumbel-Softmax)
            nn.Linear(hidden_dim, num_mixtures)
        )
        
        # Global mixture parameters (learnable)
        self.mixture_means = nn.Parameter(torch.randn(num_mixtures, z_dim))
        self.mixture_log_vars = nn.Parameter(torch.zeros(num_mixtures, z_dim))
        
    def forward(self, node_embeddings, edge_indices):
        # Concatenate node embeddings for each edge
        src_embeddings = node_embeddings[edge_indices[0]]  # Source nodes
        dst_embeddings = node_embeddings[edge_indices[1]]  # Target nodes
        edge_features = torch.cat([src_embeddings, dst_embeddings], dim=1)
        
        # Compute mixture assignments (local relation factors)
        mixture_logits = self.mlp(edge_features)
        
        # Gumbel softmax for discrete mixture selection with temperature
        temperature = 0.5  # Adjusted based on training needs
        mixture_weights = F.gumbel_softmax(mixture_logits, tau=temperature, hard=False)
        
        # Compute means and log_vars for edge embeddings
        batch_size = edge_features.size(0)
        means = torch.mm(mixture_weights, self.mixture_means)
        log_vars = torch.mm(mixture_weights, self.mixture_log_vars)
        
        # Reparameterization trick
        std = torch.exp(0.5 * log_vars)
        eps = torch.randn_like(std)
        z = means + eps * std
        
        return z, means, log_vars, mixture_weights, mixture_logits

class NetworkDecoder(nn.Module):
    """Decoder for network proximity"""
    def __init__(self, z_dim, hidden_dim):
        super(NetworkDecoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        return self.mlp(z).squeeze(-1)

class AttributeDecoder(nn.Module):
    """Decoder for node attributes"""
    def __init__(self, z_dim, hidden_dim, feature_dim):
        super(AttributeDecoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * feature_dim)  # Reconstructing concatenated node features
        )
        
    def forward(self, z):
        return self.mlp(z)

class ReLearnModel(nn.Module):
    """Simplified ReLearn model with two decoders"""
    def __init__(self, input_dim, hidden_dim, z_dim, num_mixtures=5):
        super(ReLearnModel, self).__init__()
        self.z_dim = z_dim
        
        # Encoders
        self.node_encoder = GCNEncoder(input_dim, hidden_dim, hidden_dim)
        self.edge_encoder = EdgeEncoder(hidden_dim, hidden_dim, z_dim, num_mixtures)
        
        # Decoders
        self.network_decoder = NetworkDecoder(z_dim, hidden_dim)
        self.attribute_decoder = AttributeDecoder(z_dim, hidden_dim, input_dim)
        
        # Prior distribution parameters
        self.prior_mean = nn.Parameter(torch.zeros(z_dim), requires_grad=False)
        self.prior_log_var = nn.Parameter(torch.zeros(z_dim), requires_grad=False)
        
    def encode(self, x, edge_index, edge_weight=None):
        # Encode nodes
        node_embeddings = self.node_encoder(x, edge_index, edge_weight)
        return node_embeddings
    
    def encode_edges(self, node_embeddings, edge_indices):
        # Encode edges
        z, means, log_vars, mixture_weights, mixture_logits = self.edge_encoder(node_embeddings, edge_indices)
        return z, means, log_vars, mixture_weights, mixture_logits
    
    def decode_network(self, z):
        # Decode network structure
        return self.network_decoder(z)
    
    def decode_attributes(self, z):
        # Decode node attributes
        return self.attribute_decoder(z)
    
    def forward(self, x, edge_index, edge_weight=None, all_edge_indices=None):
        # Encode nodes
        node_embeddings = self.encode(x, edge_index, edge_weight)
        
        # If all_edge_indices is not provided, use edge_index
        if all_edge_indices is None:
            all_edge_indices = edge_index
        
        # Encode edges
        z, means, log_vars, mixture_weights, mixture_logits = self.encode_edges(node_embeddings, all_edge_indices)
        
        # Decode
        edge_predictions = self.decode_network(z)
        attribute_reconstructions = self.decode_attributes(z)
        
        return edge_predictions, attribute_reconstructions, z, means, log_vars, mixture_weights, mixture_logits
    
    def kl_divergence(self, means, log_vars, mixture_weights, mixture_logits):
        """Compute KL divergence for the Gaussian mixture model"""
        # KL divergence for the Gaussian components
        kl_gauss = -0.5 * torch.sum(1 + log_vars - means.pow(2) - log_vars.exp(), dim=1)
        
        # KL divergence for the categorical distribution (with uniform prior)
        num_mixtures = mixture_logits.size(1)
        uniform_prior = torch.ones_like(mixture_logits) / num_mixtures
        kl_cat = torch.sum(mixture_weights * (torch.log(mixture_weights + 1e-8) - torch.log(uniform_prior + 1e-8)), dim=1)
        
        return kl_gauss + kl_cat

def train_relearn(model, optimizer, graph, latent_positions, num_epochs=100, lambda1=0.5, lambda2=0.5):
    """Train the ReLearn model"""
    # Convert graph to PyTorch Geometric format
    edge_list = list(graph.edges())
    edge_index = torch.tensor([[u, v] for u, v in edge_list] + [[v, u] for u, v in edge_list], dtype=torch.long).t()
    
    # Create edge weights (default to 1.0 if not weighted)
    edge_weights = []
    for u, v in edge_list:
        weight = graph[u][v].get('weight', 1.0)
        edge_weights.append(weight)
    edge_weights = edge_weights + edge_weights  # For both directions
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    
    # Convert latent positions to node features
    x = torch.tensor(latent_positions, dtype=torch.float)
    
    # Create all possible edges for negative sampling
    num_nodes = latent_positions.shape[0]
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        # Sample negative edges
        neg_edge_indices = sample_negative_edges(graph, num_samples=len(edge_list))
        neg_edge_index = torch.tensor([[u, v] for u, v in neg_edge_indices] + [[v, u] for u, v in neg_edge_indices], dtype=torch.long).t()
        
        # Combine positive and negative edges
        all_edge_index = torch.cat([edge_index, neg_edge_index], dim=1)
        edge_labels = torch.cat([torch.ones(edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])], dim=0)
        
        # Forward pass
        edge_predictions, attr_recon, z, means, log_vars, mixture_weights, mixture_logits = model(x, edge_index, edge_weights, all_edge_index)
        
        # Prepare node attribute targets (concatenated node features for each edge)
        src_nodes = all_edge_index[0]
        dst_nodes = all_edge_index[1]
        attr_targets = torch.cat([x[src_nodes], x[dst_nodes]], dim=1)
        
        # Compute losses
        # 1. Network proximity loss
        network_loss = F.binary_cross_entropy(edge_predictions, edge_labels)
        
        # 2. Attribute reconstruction loss
        attr_loss = F.mse_loss(attr_recon, attr_targets)
        
        # 3. KL divergence
        kl_loss = model.kl_divergence(means, log_vars, mixture_weights, mixture_logits).mean()
        
        # Total loss with weighting
        total_loss = lambda1 * network_loss + lambda2 * attr_loss + kl_loss
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}, "
                  f"Network Loss: {network_loss.item():.4f}, Attr Loss: {attr_loss.item():.4f}, "
                  f"KL Loss: {kl_loss.item():.4f}")
    
    return losses, z.detach()

def sample_negative_edges(graph, num_samples):
    """Sample negative edges that don't exist in the graph"""
    nodes = list(graph.nodes())
    num_nodes = len(nodes)
    existing_edges = set(graph.edges())
    
    # Build set of all possible edges
    all_possible_edges = set()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            all_possible_edges.add((i, j))
    
    available_edges = list(all_possible_edges - existing_edges)
    
    if num_samples > len(available_edges):
        #print(f"Warning: Requested {num_samples} negative edges but only {len(available_edges)} available. Sampling all.")
        num_samples = len(available_edges)
    
    neg_edges = list(np.random.choice(len(available_edges), num_samples, replace=False))
    neg_edges = [available_edges[i] for i in neg_edges]
    
    return neg_edges

def visualize_graph(graph, latent_positions, title="Graph Visualization"):
    """Visualize the graph with nodes colored by degree"""
    degrees = dict(nx.degree(graph))
    
    plt.figure(figsize=(10, 8))
    
    # Create a scatter plot of node positions
    plt.scatter(
        latent_positions[:, 0], 
        latent_positions[:, 1], 
        c=[degrees[i] for i in range(len(latent_positions))],
        cmap='viridis', 
        s=50, 
        alpha=0.8
    )
    
    # Add a colorbar to show degree scale
    cbar = plt.colorbar()
    cbar.set_label('Node Degree')
    
    # Add title and labels
    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    plt.tight_layout()
    plt.show()

def visualize_edge_embeddings(edge_embeddings, edge_index, graph, method='tsne', title="Edge Embeddings Visualization"):
    """Visualize edge embeddings using dimensionality reduction"""
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    # Convert to numpy for dimensionality reduction
    edge_embeddings_np = edge_embeddings.cpu().numpy()
    reduced_embeddings = reducer.fit_transform(edge_embeddings_np)
    
    # Extract edge information for coloring - ensure same length as reduced_embeddings
    edge_count = edge_embeddings.shape[0]
    
    # Get edge weights (assuming weighted graph)
    edge_weights = []
    for i in range(edge_count):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        try:
            weight = graph[u][v].get('weight', 1.0)
        except KeyError:
            # Handle case where edge might not exist in graph (e.g., during testing)
            weight = 0.0
        edge_weights.append(weight)
    
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(edge_embeddings_np)
    
    plt.figure(figsize=(16, 6))
    
    # Plot by edge weight
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(
        reduced_embeddings[:, 0], 
        reduced_embeddings[:, 1], 
        c=edge_weights, 
        cmap='plasma', 
        s=50, 
        alpha=0.8
    )
    plt.colorbar(scatter, label='Edge Weight')
    plt.title(f"{title} (Colored by Edge Weight)")
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    
    # Plot by cluster
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(
        reduced_embeddings[:, 0], 
        reduced_embeddings[:, 1], 
        c=clusters, 
        cmap='tab10', 
        s=50, 
        alpha=0.8
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title(f"{title} (Clustered)")
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    
    plt.tight_layout()
    plt.show()
    
    return clusters

def run_relearn_pipeline(graph, latent_positions, hidden_dim=64, z_dim=32, num_mixtures=5, num_epochs=100, latent_viz=True):
    """Run the complete ReLearn pipeline"""
    # Initialize model
    input_dim = latent_positions.shape[1]  # Node feature dimension
    model = ReLearnModel(input_dim, hidden_dim, z_dim, num_mixtures)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Convert graph to PyTorch Geometric format for visualization
    edge_list = list(graph.edges())
    edge_index = torch.tensor([[u, v] for u, v in edge_list] + [[v, u] for u, v in edge_list], dtype=torch.long).t()
    
    # 1. Visualize original graph and latent positions
    if latent_viz:
        print("Visualizing original graph...")
        visualize_graph(graph, latent_positions, title="Original Graph with Latent Positions")
    
    # 2. Initial random edge embeddings
    print("Generating initial edge embeddings...")
    model.eval()
    with torch.no_grad():
        x = torch.tensor(latent_positions, dtype=torch.float)
        node_embeddings = model.encode(x, edge_index)
        _, _, _, _, _, _, initial_edge_embeddings = model(x, edge_index)
    
    # 3. Visualize initial edge embeddings
    if latent_viz:
        print("Visualizing initial edge embeddings...")
        visualize_edge_embeddings(initial_edge_embeddings[:edge_index.shape[1]//2], edge_index[:, :edge_index.shape[1]//2], 
                                graph, method='tsne', title="Initial Edge Embeddings")
    
    # 4. Train the model
    print("\nTraining ReLearn model...")
    losses, learned_embeddings = train_relearn(model, optimizer, graph, latent_positions, num_epochs=num_epochs)
    
    # 5. Visualize training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('ReLearn Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # 6. Visualize learned edge embeddings
    if latent_viz:
        print("\nVisualizing learned edge embeddings...")
        model.eval()
        with torch.no_grad():
            x = torch.tensor(latent_positions, dtype=torch.float)
            node_embeddings = model.encode(x, edge_index)
            z, _, _, _, _ = model.encode_edges(node_embeddings, edge_index)
            clusters = visualize_edge_embeddings(z[:edge_index.shape[1]//2], edge_index[:, :edge_index.shape[1]//2], 
                                            graph, method='tsne', title="Learned Edge Embeddings")
    else:
        clusters= None
    
    # 7. Evaluate link prediction (optional)
    print("\nEvaluating link prediction...")
    model.eval()
    with torch.no_grad():
        # Sample some test edges
        test_pos_edges = sample_negative_edges(graph, num_samples=100)
        test_neg_edges = sample_negative_edges(graph, num_samples=100)
        
        # Add positive test edges to graph temporarily for embedding
        for u, v in test_pos_edges:
            graph.add_edge(u, v)
        
        # Prepare edge index with test edges
        all_edges = list(graph.edges())
        test_edge_index = torch.tensor([[u, v] for u, v in all_edges] + [[v, u] for u, v in all_edges], dtype=torch.long).t()
        
        # Generate embeddings and predictions
        x = torch.tensor(latent_positions, dtype=torch.float)
        node_embeddings = model.encode(x, test_edge_index)
        
        # Create index for test edges
        test_edges_index = torch.tensor([[u, v] for u, v in test_pos_edges + test_neg_edges], dtype=torch.long).t()
        z_test, _, _, _, _ = model.encode_edges(node_embeddings, test_edges_index)
        preds = model.decode_network(z_test)
        
        # Evaluate
        labels = torch.cat([torch.ones(len(test_pos_edges)), torch.zeros(len(test_neg_edges))], dim=0)
        accuracy = ((preds > 0.5).float() == labels).float().mean().item()
        
        # Remove temporary edges
        for u, v in test_pos_edges:
            graph.remove_edge(u, v)
            
        print(f"Link prediction accuracy: {accuracy:.4f}")
    
    return model, losses, clusters