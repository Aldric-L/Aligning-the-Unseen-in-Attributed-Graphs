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
    """Scalable GCN Encoder for node features processing"""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(GCNEncoder, self).__init__()
        
        # List to hold all convolutional layers
        self.convs = nn.ModuleList()
        
        # Create first layer from input_dim to first hidden dimension
        if len(hidden_dims) > 0:
            self.convs.append(GCNConv(input_dim, hidden_dims[0]))
            
            # Create intermediate layers
            for i in range(len(hidden_dims)-1):
                self.convs.append(GCNConv(hidden_dims[i], hidden_dims[i+1]))
            
            # Create output layer
            self.convs.append(GCNConv(hidden_dims[-1], output_dim))
        else:
            # If no hidden layers, connect input directly to output
            self.convs.append(GCNConv(input_dim, output_dim))
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_weight=None):
        # Apply all layers except the last with ReLU and dropout
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply last layer without activation (applied later if needed)
        x = self.convs[-1](x, edge_index, edge_weight)
        return x

class MLP(nn.Module):
    """Scalable MLP module for easy layer configuration"""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1, activation=nn.ReLU()):
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout
        
        # Input layer
        if len(hidden_dims) > 0:
            self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
            
            # Hidden layers
            for i in range(len(hidden_dims)-1):
                self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            
            # Output layer
            self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        else:
            # Direct mapping if no hidden layers
            self.layers.append(nn.Linear(input_dim, output_dim))
            
    def forward(self, x):
        # Process all layers except the last
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Last layer (no activation, handled by calling function if needed)
        x = self.layers[-1](x)
        return x

class EdgeEncoder(nn.Module):
    """Edge encoder for computing local relation factors"""
    def __init__(self, node_emb_dim, hidden_dims, z_dim, num_mixtures=5, dropout=0.1):
        super(EdgeEncoder, self).__init__()
        self.z_dim = z_dim
        self.num_mixtures = num_mixtures
        
        # Neural network for local relation factors using scalable MLP
        self.mlp = MLP(
            input_dim=2 * node_emb_dim,
            hidden_dims=hidden_dims,
            output_dim=num_mixtures,
            dropout=dropout
        )
        
        # Global mixture parameters (learnable)
        self.mixture_means = nn.Parameter(torch.randn(num_mixtures, z_dim))
        self.mixture_log_vars = nn.Parameter(torch.zeros(num_mixtures, z_dim))
        
    def forward(self, node_embeddings, edge_indices, temperature=0.5):
        # Concatenate node embeddings for each edge
        src_embeddings = node_embeddings[edge_indices[0]]  # Source nodes
        dst_embeddings = node_embeddings[edge_indices[1]]  # Target nodes
        edge_features = torch.cat([src_embeddings, dst_embeddings], dim=1)
        
        # Compute mixture assignments (local relation factors)
        mixture_logits = self.mlp(edge_features)
        
        # Gumbel softmax for discrete mixture selection with temperature
        mixture_weights = F.gumbel_softmax(mixture_logits, tau=temperature, hard=False)
        
        # Compute means and log_vars for edge embeddings
        means = torch.mm(mixture_weights, self.mixture_means)
        log_vars = torch.mm(mixture_weights, self.mixture_log_vars)
        
        # Reparameterization trick
        std = torch.exp(0.5 * log_vars)
        eps = torch.randn_like(std)
        z = means + eps * std
        
        return z, means, log_vars, mixture_weights, mixture_logits

class NetworkDecoder(nn.Module):
    """Decoder for network proximity"""
    def __init__(self, z_dim, hidden_dims, dropout=0.1):
        super(NetworkDecoder, self).__init__()
        self.mlp = MLP(
            input_dim=z_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=dropout
        )
        
    def forward(self, z):
        return torch.sigmoid(self.mlp(z)).squeeze(-1)

class AttributeDecoder(nn.Module):
    """Decoder for node attributes"""
    def __init__(self, z_dim, hidden_dims, feature_dim, dropout=0.1):
        super(AttributeDecoder, self).__init__()
        self.mlp = MLP(
            input_dim=z_dim,
            hidden_dims=hidden_dims,
            output_dim=2 * feature_dim,  # Reconstructing concatenated node features
            dropout=dropout
        )
        
    def forward(self, z):
        return self.mlp(z)

class ReLearnModel(nn.Module):
    """Improved ReLearn model with scalable architecture and KL annealing support"""
    def __init__(self, input_dim, gcn_hidden_dims, edge_mlp_hidden_dims, 
                 decoder_hidden_dims, z_dim, num_mixtures=5, dropout=0.1):
        super(ReLearnModel, self).__init__()
        self.z_dim = z_dim
        
        # Encoders
        self.node_encoder = GCNEncoder(
            input_dim=input_dim, 
            hidden_dims=gcn_hidden_dims, 
            output_dim=gcn_hidden_dims[-1] if len(gcn_hidden_dims) > 0 else 64,
            dropout=dropout
        )
        
        self.edge_encoder = EdgeEncoder(
            node_emb_dim=gcn_hidden_dims[-1] if len(gcn_hidden_dims) > 0 else 64, 
            hidden_dims=edge_mlp_hidden_dims, 
            z_dim=z_dim, 
            num_mixtures=num_mixtures,
            dropout=dropout
        )
        
        # Decoders
        self.network_decoder = NetworkDecoder(
            z_dim=z_dim, 
            hidden_dims=decoder_hidden_dims,
            dropout=dropout
        )
        
        self.attribute_decoder = AttributeDecoder(
            z_dim=z_dim, 
            hidden_dims=decoder_hidden_dims, 
            feature_dim=input_dim,
            dropout=dropout
        )
        
        # Prior distribution parameters
        self.prior_mean = nn.Parameter(torch.zeros(z_dim), requires_grad=False)
        self.prior_log_var = nn.Parameter(torch.zeros(z_dim), requires_grad=False)
        
    def encode(self, x, edge_index, edge_weight=None):
        # Encode nodes
        node_embeddings = self.node_encoder(x, edge_index, edge_weight)
        return node_embeddings
    
    def encode_edges(self, node_embeddings, edge_indices, temperature=0.5):
        # Encode edges
        z, means, log_vars, mixture_weights, mixture_logits = self.edge_encoder(
            node_embeddings, edge_indices, temperature=temperature
        )
        return z, means, log_vars, mixture_weights, mixture_logits
    
    def decode_network(self, z):
        # Decode network structure
        return self.network_decoder(z)
    
    def decode_attributes(self, z):
        # Decode node attributes
        return self.attribute_decoder(z)
    
    def forward(self, x, edge_index, edge_weight=None, all_edge_indices=None, temperature=0.5):
        # Encode nodes
        node_embeddings = self.encode(x, edge_index, edge_weight)
        
        # If all_edge_indices is not provided, use edge_index
        if all_edge_indices is None:
            all_edge_indices = edge_index
        
        # Encode edges
        z, means, log_vars, mixture_weights, mixture_logits = self.encode_edges(
            node_embeddings, all_edge_indices, temperature=temperature
        )
        
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

def train_relearn(model, optimizer, graph, latent_positions, num_epochs=100, 
                  lambda1=0.5, lambda2=0.5, kl_annealing=True, annealing_epochs=50,
                  kl_start=0.0, kl_end=1.0, temperature_start=1.0, temperature_end=0.1):
    """Train the ReLearn model with KL annealing"""
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
    
    # Training loop
    model.train()
    losses = []
    network_losses = []
    attr_losses = []
    kl_losses = []
    
    for epoch in range(num_epochs):
        # KL annealing factor (linearly increasing from kl_start to kl_end)
        if kl_annealing:
            kl_weight = kl_start + (kl_end - kl_start) * min(epoch / annealing_epochs, 1.0)
        else:
            kl_weight = 1.0
            
        # Temperature annealing for Gumbel-Softmax
        temperature = temperature_start - (temperature_start - temperature_end) * min(epoch / annealing_epochs, 1.0)
        
        # Sample negative edges
        neg_edge_indices = sample_negative_edges(graph, num_samples=len(edge_list))
        neg_edge_index = torch.tensor([[u, v] for u, v in neg_edge_indices] + [[v, u] for u, v in neg_edge_indices], dtype=torch.long).t()
        
        # Combine positive and negative edges
        all_edge_index = torch.cat([edge_index, neg_edge_index], dim=1)
        edge_labels = torch.cat([torch.ones(edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])], dim=0)
        
        # Forward pass with current temperature
        edge_predictions, attr_recon, z, means, log_vars, mixture_weights, mixture_logits = model(
            x, edge_index, edge_weights, all_edge_index, temperature=temperature)
        
        # Prepare node attribute targets (concatenated node features for each edge)
        src_nodes = all_edge_index[0]
        dst_nodes = all_edge_index[1]
        attr_targets = torch.cat([x[src_nodes], x[dst_nodes]], dim=1)
        
        # Compute losses
        # 1. Network proximity loss
        network_loss = F.binary_cross_entropy(edge_predictions, edge_labels)
        
        # 2. Attribute reconstruction loss
        attr_loss = F.mse_loss(attr_recon, attr_targets)
        
        # 3. KL divergence with annealing
        kl_loss = model.kl_divergence(means, log_vars, mixture_weights, mixture_logits).mean()
        
        # Total loss with weighting and KL annealing
        total_loss = lambda1 * network_loss + lambda2 * attr_loss + kl_weight * kl_loss
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Track losses
        losses.append(total_loss.item())
        network_losses.append(network_loss.item())
        attr_losses.append(attr_loss.item())
        kl_losses.append(kl_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}, "
                  f"Network Loss: {network_loss.item():.4f}, Attr Loss: {attr_loss.item():.4f}, "
                  f"KL Loss: {kl_loss.item():.4f} (weight: {kl_weight:.4f}), Temp: {temperature:.4f}")
    
    loss_data = {
        'total': losses,
        'network': network_losses,
        'attribute': attr_losses,
        'kl': kl_losses
    }
    
    return loss_data, z.detach()

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

def visualize_training_losses(loss_data):
    """Visualize the different loss components during training"""
    plt.figure(figsize=(12, 8))
    
    # Plot all losses
    plt.subplot(2, 2, 1)
    plt.plot(loss_data['total'], label='Total Loss')
    plt.title('Total Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(loss_data['network'], label='Network Loss')
    plt.title('Network Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(loss_data['attribute'], label='Attribute Loss')
    plt.title('Attribute Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(loss_data['kl'], label='KL Divergence')
    plt.title('KL Divergence Loss (Before Annealing)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def run_relearn_pipeline(graph, latent_positions, gcn_hidden_dims=[64, 128], 
                         edge_mlp_hidden_dims=[128, 64], decoder_hidden_dims=[64, 32],
                         z_dim=32, num_mixtures=5, num_epochs=100, kl_annealing=True,
                         annealing_epochs=50, kl_start=0.0, kl_end=1.0, 
                         temperature_start=1.0, temperature_end=0.1, 
                         lambda1=0.5, lambda2=0.5, lr=0.001, dropout=0.1,
                         latent_viz=True):
    """Run the complete ReLearn pipeline with configurable architecture and KL annealing"""
    # Initialize model
    input_dim = latent_positions.shape[1]  # Node feature dimension
    model = ReLearnModel(
        input_dim=input_dim, 
        gcn_hidden_dims=gcn_hidden_dims, 
        edge_mlp_hidden_dims=edge_mlp_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        z_dim=z_dim, 
        num_mixtures=num_mixtures,
        dropout=dropout
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
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
    
    # 4. Train the model with KL annealing
    print("\nTraining ReLearn model with KL annealing...")
    loss_data, learned_embeddings = train_relearn(
        model, optimizer, graph, latent_positions, 
        num_epochs=num_epochs,
        lambda1=lambda1, 
        lambda2=lambda2,
        kl_annealing=kl_annealing,
        annealing_epochs=annealing_epochs,
        kl_start=kl_start,
        kl_end=kl_end,
        temperature_start=temperature_start,
        temperature_end=temperature_end
    )
    
    # 5. Visualize training curves
    visualize_training_losses(loss_data)
    
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
        clusters = None
    
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
    
    return model, loss_data, clusters

def ensure_minimum_edges(graph, min_edges=50):
    """
    Ensure the graph has at least min_edges edges for visualization purposes.
    Returns the original graph if it already has enough edges,
    otherwise returns a modified graph with additional edges.
    """
    edge_count = len(list(graph.edges()))
    
    if edge_count >= min_edges:
        return graph
    
    # Make a copy to avoid modifying the original
    modified_graph = graph.copy()
    nodes = list(graph.nodes())
    num_nodes = len(nodes)
    
    # Calculate how many edges to add
    edges_to_add = min_edges - edge_count
    print(f"Adding {edges_to_add} edges to ensure minimum sample size for visualization")
    
    # Get existing edges to avoid duplicates
    existing_edges = set(graph.edges())
    
    # Add random edges
    added = 0
    attempts = 0
    max_attempts = edges_to_add * 10  # Avoid infinite loops
    
    while added < edges_to_add and attempts < max_attempts:
        attempts += 1
        i = np.random.randint(0, num_nodes)
        j = np.random.randint(0, num_nodes)
        
        if i != j and (i, j) not in existing_edges and (j, i) not in existing_edges:
            modified_graph.add_edge(i, j, weight=0.1)  # Add with low weight to indicate artificial
            existing_edges.add((i, j))
            added += 1
    
    print(f"Successfully added {added} edges")
    return modified_graph

# Modified run_relearn_pipeline function to handle small graphs
def run_relearn_pipeline(graph, latent_positions, gcn_hidden_dims=[64, 128], 
                         edge_mlp_hidden_dims=[128, 64], decoder_hidden_dims=[64, 32],
                         z_dim=32, num_mixtures=5, num_epochs=100, kl_annealing=True,
                         annealing_epochs=50, kl_start=0.0, kl_end=1.0, 
                         temperature_start=1.0, temperature_end=0.1, 
                         lambda1=0.5, lambda2=0.5, lr=0.001, dropout=0.1,
                         latent_viz=True):
    """Run the complete ReLearn pipeline with configurable architecture and KL annealing"""
    # Initialize model
    input_dim = latent_positions.shape[1]  # Node feature dimension
    model = ReLearnModel(
        input_dim=input_dim, 
        gcn_hidden_dims=gcn_hidden_dims, 
        edge_mlp_hidden_dims=edge_mlp_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        z_dim=z_dim, 
        num_mixtures=num_mixtures,
        dropout=dropout
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
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
        # Check if we have enough edges for meaningful visualization
        if edge_index.shape[1] // 2 < 10:
            print(f"WARNING: Only {edge_index.shape[1] // 2} edges available, skipping initial embedding visualization")
        else:
            try:
                visualize_edge_embeddings(initial_edge_embeddings[:edge_index.shape[1]//2], 
                                        edge_index[:, :edge_index.shape[1]//2], 
                                        graph, method='tsne', title="Initial Edge Embeddings")
            except Exception as e:
                print(f"Error during edge embedding visualization: {e}")
                print("Falling back to PCA for visualization")
                visualize_edge_embeddings(initial_edge_embeddings[:edge_index.shape[1]//2], 
                                        edge_index[:, :edge_index.shape[1]//2], 
                                        graph, method='pca', title="Initial Edge Embeddings")
    
    # 4. Train the model with KL annealing
    print("\nTraining ReLearn model with KL annealing...")
    loss_data, learned_embeddings = train_relearn(
        model, optimizer, graph, latent_positions, 
        num_epochs=num_epochs,
        lambda1=lambda1, 
        lambda2=lambda2,
        kl_annealing=kl_annealing,
        annealing_epochs=annealing_epochs,
        kl_start=kl_start,
        kl_end=kl_end,
        temperature_start=temperature_start,
        temperature_end=temperature_end
    )
    
    # 5. Visualize training curves
    visualize_training_losses(loss_data)
    
    # 6. Visualize learned edge embeddings
    clusters = None
    if latent_viz:
        print("\nVisualizing learned edge embeddings...")
        model.eval()
        with torch.no_grad():
            x = torch.tensor(latent_positions, dtype=torch.float)
            node_embeddings = model.encode(x, edge_index)
            z, _, _, _, _ = model.encode_edges(node_embeddings, edge_index)
            
            # Check if we have enough edges for visualization
            if edge_index.shape[1] // 2 < 10:
                print(f"WARNING: Only {edge_index.shape[1] // 2} edges available, skipping learned embedding visualization")
            else:
                try:
                    clusters = visualize_edge_embeddings(z[:edge_index.shape[1]//2], 
                                                    edge_index[:, :edge_index.shape[1]//2], 
                                                    graph, method='tsne', title="Learned Edge Embeddings")
                except Exception as e:
                    print(f"Error during edge embedding visualization: {e}")
                    print("Falling back to PCA for visualization")
                    clusters = visualize_edge_embeddings(z[:edge_index.shape[1]//2], 
                                                    edge_index[:, :edge_index.shape[1]//2], 
                                                    graph, method='pca', title="Learned Edge Embeddings")
    
    # 7. Evaluate link prediction (optional)
    print("\nEvaluating link prediction...")
    model.eval()
    with torch.no_grad():
        # Sample test edges based on graph size
        test_size = min(100, max(5, len(edge_list) // 4))  # Make test size adaptive
        
        # Sample some test edges
        test_pos_edges = sample_negative_edges(graph, num_samples=test_size)
        test_neg_edges = sample_negative_edges(graph, num_samples=test_size)
        
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
    
    return model, loss_data, clusters

# Modified function to handle small sample sizes 
def visualize_edge_embeddings(edge_embeddings, edge_index, graph, method='tsne', title="Edge Embeddings Visualization"):
    """Visualize edge embeddings using dimensionality reduction"""
    # Apply dimensionality reduction
    edge_embeddings_np = edge_embeddings.cpu().numpy()
    edge_count = edge_embeddings.shape[0]
    
    # Check if we have sufficient samples for t-SNE with default perplexity
    if method.lower() == 'tsne':
        # Set appropriate perplexity based on sample size
        if edge_count < 5:
            print(f"WARNING: Only {edge_count} samples available, switching to PCA for visualization")
            method = 'pca'
            reducer = PCA(n_components=2, random_state=42)
        elif edge_count < 50:
            # Use smaller perplexity for small datasets
            perplexity = max(5, edge_count // 5)  # Reasonable heuristic
            print(f"WARNING: Using reduced perplexity of {perplexity} due to small sample size")
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        else:
            reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    # Convert to numpy for dimensionality reduction
    reduced_embeddings = reducer.fit_transform(edge_embeddings_np)
    
    # Extract edge information for coloring
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
    
    # Adjust clustering based on sample size
    n_clusters = min(5, max(2, edge_count // 10))  # Reasonable heuristic
    
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
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


def evaluate_link_prediction(model, graph, latent_positions, test_ratio=0.2):
    """Evaluate link prediction performance"""
    # Split edges into train and test
    all_edges = list(graph.edges())
    np.random.shuffle(all_edges)
    
    test_size = int(len(all_edges) * test_ratio)
    test_edges = all_edges[:test_size]
    train_edges = all_edges[test_size:]
    
    # Create a training graph
    train_graph = graph.copy()
    for u, v in test_edges:
        train_graph.remove_edge(u, v)
    
    # Create equal number of negative edges for testing
    negative_edges = sample_negative_edges(graph, num_samples=len(test_edges))
    
    # Convert to PyTorch tensors
    train_edge_index = torch.tensor([[u, v] for u, v in train_edges] + 
                                   [[v, u] for u, v in train_edges], dtype=torch.long).t()
    
    test_pos_edge_index = torch.tensor([[u, v] for u, v in test_edges], dtype=torch.long).t()
    test_neg_edge_index = torch.tensor([[u, v] for u, v in negative_edges], dtype=torch.long).t()
    
    # Train the model on training edges
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    _, _ = train_relearn(model, optimizer, train_graph, latent_positions, num_epochs=50)
    
    # Evaluate on test edges
    model.eval()
    with torch.no_grad():
        x = torch.tensor(latent_positions, dtype=torch.float)
        node_embeddings = model.encode(x, train_edge_index)
        
        # Positive edges
        z_pos, _, _, _, _ = model.encode_edges(node_embeddings, test_pos_edge_index)
        pos_preds = model.decode_network(z_pos)
        
        # Negative edges
        z_neg, _, _, _, _ = model.encode_edges(node_embeddings, test_neg_edge_index)
        neg_preds = model.decode_network(z_neg)
        
        # Combine predictions
        preds = torch.cat([pos_preds, neg_preds], dim=0)
        labels = torch.cat([torch.ones(len(test_edges)), torch.zeros(len(negative_edges))], dim=0)
        
        # Calculate metrics
        accuracy = ((preds > 0.5).float() == labels).float().mean().item()
        
        # Calculate AUC-ROC
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Sort by prediction scores
        sorted_indices = np.argsort(preds_np)[::-1]
        sorted_labels = labels_np[sorted_indices]
        
        # Calculate TPR and FPR
        tp = np.cumsum(sorted_labels)
        fp = np.cumsum(1 - sorted_labels)
        tpr = tp / np.sum(labels_np)
        fpr = fp / np.sum(1 - labels_np)
        
        # Calculate AUC
        auc = np.trapz(tpr, fpr)
        
    return accuracy, auc
