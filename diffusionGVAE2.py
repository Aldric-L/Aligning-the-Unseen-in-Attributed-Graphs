import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from sklearn.manifold import TSNE
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic dataset using Erdős–Rényi model
def generate_erdos_renyi_dataset(num_graphs=100, num_nodes=20, p=0.3, min_weight=0.1, max_weight=1.0):
    dataset = []
    for _ in range(num_graphs):
        # Generate random graph
        G = nx.erdos_renyi_graph(num_nodes, p)
        
        # Assign random weights to edges
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(min_weight, max_weight)
        
        # Create weighted adjacency matrix
        adj_matrix = nx.to_numpy_array(G)
        
        # Make sure it's symmetric (undirected)
        adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
        
        dataset.append(torch.FloatTensor(adj_matrix))
    
    return dataset

# Heat kernel (diffusion distance)
def heat_kernel_distance(adj_matrix1, adj_matrix2, t=1.0):
    """
    Compute the heat kernel distance between two graphs.
    t is the diffusion time parameter.
    """
    device = adj_matrix1.device
    
    # Normalize adjacency matrices
    def normalize_adj(adj):
        # Add self-loops
        adj_with_self_loops = adj + torch.eye(adj.size(0), device=device)
        # Compute degree matrix
        degree = torch.sum(adj_with_self_loops, dim=1)
        # Compute D^(-1/2)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        # Compute normalized adjacency
        return torch.mm(torch.mm(d_mat_inv_sqrt, adj_with_self_loops), d_mat_inv_sqrt)
    
    # Normalized adjacency matrices
    norm_adj1 = normalize_adj(adj_matrix1)
    norm_adj2 = normalize_adj(adj_matrix2)
    
    # Heat kernel: H(t) = exp(-t(I-A))
    heat_kernel1 = torch.matrix_exp(-t * (torch.eye(adj_matrix1.size(0), device=device) - norm_adj1))
    heat_kernel2 = torch.matrix_exp(-t * (torch.eye(adj_matrix2.size(0), device=device) - norm_adj2))
    
    # Compute Frobenius norm between heat kernels
    distance = torch.norm(heat_kernel1 - heat_kernel2, p='fro')
    
    return distance

# Improved Graph Convolutional Network VAE
class GraphVAE(nn.Module):
    def __init__(self, num_nodes, hidden_dim=32, latent_dim=16, node_feature_size=8):
        super(GraphVAE, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Use a larger node feature size instead of just degree
        self.node_feature_size = node_feature_size
        
        # Generate initial node features
        self.node_embedding = nn.Embedding(num_nodes, node_feature_size)
        
        # Encoder
        self.gc1 = GCNConv(node_feature_size, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, hidden_dim)
        self.gc3 = GCNConv(hidden_dim, hidden_dim)  # Extra GCN layer
        
        # Latent space projection
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder - improved edge prediction network
        self.edge_predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Apply sigmoid here to avoid numerical issues
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def encode(self, adj_matrix):
        # Ensure adj_matrix is 2D (for single graph)
        if adj_matrix.dim() == 1:
            adj_matrix = adj_matrix.view(self.num_nodes, self.num_nodes)
        
        device = adj_matrix.device
        
        # Convert dense adjacency matrix to sparse format for GCN
        edge_index, edge_attr = dense_to_sparse(adj_matrix)
        
        # Create node features using embeddings and adding node degrees
        node_indices = torch.arange(self.num_nodes, device=device)
        x_embeddings = self.node_embedding(node_indices)
        
        # Add node degrees as an additional feature
        node_degrees = adj_matrix.sum(dim=1, keepdim=True)
        x = x_embeddings  # Use embeddings as features
        
        # Apply GCN layers with residual connections
        h1 = F.relu(self.gc1(x, edge_index, edge_attr))
        h2 = F.relu(self.gc2(h1, edge_index, edge_attr)) + h1  # Residual connection
        h3 = F.relu(self.gc3(h2, edge_index, edge_attr)) + h2  # Another residual connection
        
        # Mean and log variance
        mu = self.fc_mu(h3)
        logvar = self.fc_logvar(h3)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        """
        Decode latent representations into graph structure.
        Uses a neural network to predict edge weights for all pairs of nodes.
        """
        device = z.device
        batch_size = z.size(0) // self.num_nodes
        adj_pred = torch.zeros(batch_size, self.num_nodes, self.num_nodes, device=device)
        
        # For each graph in the batch
        for b in range(batch_size):
            # Get node embeddings for current graph
            z_graph = z[b * self.num_nodes:(b + 1) * self.num_nodes]
            
            # Create all possible node pairs for more efficient prediction
            i_indices, j_indices = torch.triu_indices(self.num_nodes, self.num_nodes, 1, device=device)
            
            # Prepare pairs of node embeddings
            edge_features = torch.cat([
                z_graph[i_indices], 
                z_graph[j_indices]
            ], dim=1)
            
            # Predict all edge weights at once
            edge_weights = self.edge_predictor(edge_features).squeeze(-1)
            
            # Fill in upper triangular part of adjacency matrix
            adj_pred[b, i_indices, j_indices] = edge_weights
            
            # Mirror for symmetry (undirected graph)
            adj_pred[b, j_indices, i_indices] = edge_weights
        
        return adj_pred
    
    def forward(self, adj_matrix):
        # Handle different input dimensions
        if adj_matrix.dim() == 2:  # Single graph
            batch_size = 1
            adj_matrix_batch = adj_matrix.unsqueeze(0)  # Add batch dimension
        elif adj_matrix.dim() == 3:  # Batch of graphs
            batch_size = adj_matrix.size(0)
            adj_matrix_batch = adj_matrix
        else:
            raise ValueError(f"Adjacency matrix must be 2D or 3D (got {adj_matrix.dim()} dimensions)")
        
        all_mu, all_logvar = [], []
        
        # Process each graph in the batch
        for b in range(batch_size):
            mu, logvar = self.encode(adj_matrix_batch[b])
            all_mu.append(mu)
            all_logvar.append(logvar)
        
        # Stack results
        mu = torch.cat(all_mu, dim=0)
        logvar = torch.cat(all_logvar, dim=0)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        adj_pred = self.decode(z)
        
        # Remove batch dimension for single graph
        if batch_size == 1 and adj_matrix.dim() == 2:
            adj_pred = adj_pred.squeeze(0)
        
        return adj_pred, mu, logvar


# Enhanced training function with KL annealing and early stopping
def train_model(model, dataset, epochs=100, batch_size=16, learning_rate=0.001, 
                diffusion_loss_weight=0.0, t_diffusion=1.0, kl_annealing=True, 
                beta_min=0.0, beta_max=1.0, patience=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    
    # Prepare dataset
    dataset = [d.to(device) for d in dataset]
    
    # For tracking
    train_losses = []
    recon_losses = []
    kl_losses = []
    diffusion_losses = []
    kl_weights = []
    
    # Early stopping
    best_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    no_improvement = 0
    
    # Train loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        epoch_diffusion_loss = 0
        
        # KL annealing - Improved cyclical schedule
        if kl_annealing:
            # Cyclical annealing schedule with smoother transition
            cycle_size = epochs // 4
            cycle_pos = (epoch % cycle_size) / cycle_size
            kl_weight = beta_min + 0.5 * (beta_max - beta_min) * (1 - np.cos(cycle_pos * np.pi))
        else:
            kl_weight = beta_max
        
        kl_weights.append(kl_weight)
            
        # Shuffle dataset for each epoch
        indices = torch.randperm(len(dataset))
        
        # Batch training
        for start_idx in range(0, len(dataset), batch_size):
            batch_indices = indices[start_idx:start_idx+batch_size]
            batch = [dataset[i] for i in batch_indices]
            
            # Stack batch for parallel processing
            batch_adj = torch.stack(batch)
            
            optimizer.zero_grad()
            
            # Forward pass
            batch_adj_recon, batch_mu, batch_logvar = model(batch_adj)
            
            # Reconstruction loss 
            recon_loss = F.binary_cross_entropy(batch_adj_recon, batch_adj, reduction='sum') / batch_size
            
            # KL divergence
            kl_loss = (-0.5 * torch.sum(1 + batch_logvar - batch_mu.pow(2) - batch_logvar.exp())) / batch_size
            
            # Diffusion distance loss
            diff_loss = torch.tensor(0.0, device=device)
            if diffusion_loss_weight > 0:
                for i in range(batch_size):
                    diff_loss += heat_kernel_distance(batch_adj[i], batch_adj_recon[i], t=t_diffusion)
                diff_loss = diff_loss / batch_size
            
            # Total loss with annealed KL term
            loss = recon_loss + kl_weight * kl_loss + diffusion_loss_weight * diff_loss
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track losses
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_diffusion_loss += diff_loss.item()
        
        # Average losses for the epoch
        num_batches = (len(dataset) + batch_size - 1) // batch_size  # ceiling division
        avg_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        avg_diffusion_loss = epoch_diffusion_loss / num_batches
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Track losses
        train_losses.append(avg_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        diffusion_losses.append(avg_diffusion_loss)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            no_improvement = 0
        else:
            no_improvement += 1
            
        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            model.load_state_dict(best_model_state)
            break
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, "
                  f"KL Loss: {avg_kl_loss:.4f}, Diffusion Loss: {avg_diffusion_loss:.4f}, KL Weight: {kl_weight:.4f}")
    
    return train_losses, recon_losses, kl_losses, diffusion_losses, kl_weights


# Evaluation function
def evaluate_model(model, dataset, t_diffusion=1.0):
    device = next(model.parameters()).device
    model.eval()
    
    # Convert dataset to device
    dataset = [d.to(device) for d in dataset]
    
    total_recon_loss = 0
    total_kl_loss = 0
    total_diffusion_loss = 0
    
    with torch.no_grad():
        for adj_matrix in dataset:
            # Forward pass
            adj_recon, mu, logvar = model(adj_matrix)
            
            # Reconstruction loss
            recon_loss = F.binary_cross_entropy(adj_recon, adj_matrix, reduction='sum')
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Diffusion distance
            diff_loss = heat_kernel_distance(adj_matrix, adj_recon, t=t_diffusion)
            
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_diffusion_loss += diff_loss.item()
    
    avg_recon_loss = total_recon_loss / len(dataset)
    avg_kl_loss = total_kl_loss / len(dataset)
    avg_diffusion_loss = total_diffusion_loss / len(dataset)
    
    return avg_recon_loss, avg_kl_loss, avg_diffusion_loss


# Visualization functions
def visualize_graph(adj_matrix, title):
    # Convert to numpy for NetworkX
    if torch.is_tensor(adj_matrix):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)
    
    # Threshold to remove very weak edges
    threshold = 0.1
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < threshold]
    G.remove_edges_from(edges_to_remove)
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(8, 6))
    
    # Get edge weights for line thickness
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7)
    nx.draw_networkx_labels(G, pos)
    
    plt.title(title)
    plt.axis('off')
    return plt


def visualize_latent_space(model, dataset, num_samples=100):
    device = next(model.parameters()).device
    model.eval()
    latent_vectors = []
    
    # Convert dataset to the same device as model
    dataset = [d.to(device) for d in dataset]
    
    with torch.no_grad():
        for adj_matrix in dataset[:num_samples]:
            mu, _ = model.encode(adj_matrix)
            latent_vectors.append(mu.cpu().numpy())
    
    # Stack all latent vectors
    latent_vectors = np.vstack(latent_vectors)
    
    # Reduce to 2D using t-SNE if dimension > 2
    if latent_vectors.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_vectors)
    else:
        latent_2d = latent_vectors
    
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.7)
    plt.title('Latent Space Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    return plt


def plot_loss_curves(baseline_losses, diffusion_losses, kl_weights=None, title='Training Loss Comparison'):
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 2, 1)
    plt.plot(baseline_losses[0], label='Baseline')
    plt.plot(diffusion_losses[0], label='With Diffusion')
    plt.title('Total Loss')
    plt.legend()
    
    plt.subplot(3, 2, 2)
    plt.plot(baseline_losses[1], label='Baseline')
    plt.plot(diffusion_losses[1], label='With Diffusion')
    plt.title('Reconstruction Loss')
    plt.legend()
    
    plt.subplot(3, 2, 3)
    plt.plot(baseline_losses[2], label='Baseline')
    plt.plot(diffusion_losses[2], label='With Diffusion')
    plt.title('KL Divergence Loss')
    plt.legend()
    
    plt.subplot(3, 2, 4)
    plt.plot(baseline_losses[3], label='Baseline')
    plt.plot(diffusion_losses[3], label='With Diffusion')
    plt.title('Diffusion Loss')
    plt.legend()
    
    if kl_weights:
        plt.subplot(3, 2, 5)
        plt.plot(kl_weights, label='KL Weight')
        plt.title('KL Annealing Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Beta')
        plt.legend()
    
    plt.tight_layout()
    plt.suptitle(title)
    plt.subplots_adjust(top=0.9)
    return plt


def compare_reconstructions(model1, model2, test_graph, model1_name="Baseline", model2_name="Diffusion"):
    device = next(model1.parameters()).device
    test_graph = test_graph.to(device)
    
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        recon1, _, _ = model1(test_graph)
        recon2, _, _ = model2(test_graph)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original graph
    G_orig = nx.from_numpy_array(test_graph.cpu().numpy())
    pos = nx.spring_layout(G_orig, seed=42)  # Use same layout for all
    
    nx.draw_networkx(G_orig, pos=pos, ax=axes[0], 
                     node_color='skyblue', 
                     edge_color='gray',
                     width=[G_orig[u][v]['weight'] * 2 for u, v in G_orig.edges()],
                     with_labels=True)
    axes[0].set_title("Original Graph")
    axes[0].axis('off')
    
    # Model 1 reconstruction
    G_recon1 = nx.from_numpy_array(recon1.cpu().numpy())
    nx.draw_networkx(G_recon1, pos=pos, ax=axes[1], 
                     node_color='lightgreen', 
                     edge_color='gray',
                     width=[G_recon1[u][v]['weight'] * 2 for u, v in G_recon1.edges()],
                     with_labels=True)
    axes[1].set_title(f"{model1_name} Reconstruction")
    axes[1].axis('off')
    
    # Model 2 reconstruction
    G_recon2 = nx.from_numpy_array(recon2.cpu().numpy())
    nx.draw_networkx(G_recon2, pos=pos, ax=axes[2], 
                     node_color='lightcoral', 
                     edge_color='gray',
                     width=[G_recon2[u][v]['weight'] * 2 for u, v in G_recon2.edges()],
                     with_labels=True)
    axes[2].set_title(f"{model2_name} Reconstruction")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Plot adjacency matrices
    fig2, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    axs[0].imshow(test_graph.cpu().numpy(), cmap='viridis')
    axs[0].set_title("Original")
    
    axs[1].imshow(recon1.cpu().numpy(), cmap='viridis')
    axs[1].set_title("Baseline")
    
    axs[2].imshow(recon2.cpu().numpy(), cmap='viridis')
    axs[2].set_title("Diffusion")
    
    plt.tight_layout()
    
    return fig, fig2


# Main execution
def main():
    # Parameters
    num_nodes = 20
    num_graphs = 200
    hidden_dim = 64
    latent_dim = 16
    node_feature_size = 8  # Embedding dimension for nodes
    epochs = 150
    batch_size = 16
    learning_rate = 0.001
    t_diffusion = 1.0
    diffusion_loss_weight = 0.5
    patience = 15  # For early stopping
    
    # Generate dataset
    print("Generating synthetic dataset...")
    dataset = generate_erdos_renyi_dataset(num_graphs=num_graphs, num_nodes=num_nodes)
    
    # Split dataset into train/test
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    print(f"Dataset generated: {len(train_dataset)} training graphs, {len(test_dataset)} test graphs")
    
    # Initialize models
    baseline_model = GraphVAE(num_nodes=num_nodes, hidden_dim=hidden_dim, 
                              latent_dim=latent_dim, node_feature_size=node_feature_size)
    diffusion_model = GraphVAE(num_nodes=num_nodes, hidden_dim=hidden_dim, 
                               latent_dim=latent_dim, node_feature_size=node_feature_size)
    
    # Train baseline model with cyclical KL annealing
    print("\nTraining baseline Graph VAE...")
    start_time = time.time()
    baseline_losses = train_model(baseline_model, train_dataset, epochs=epochs, batch_size=batch_size, 
                                learning_rate=learning_rate, diffusion_loss_weight=0.0, kl_annealing=True,
                                beta_min=0.0, beta_max=1.0, patience=patience)
    baseline_time = time.time() - start_time
    print(f"Baseline training completed in {baseline_time:.2f} seconds")
    
    # Train diffusion distance model with cyclical KL annealing
    print("\nTraining Graph VAE with diffusion distance loss...")
    start_time = time.time()
    diffusion_losses = train_model(diffusion_model, train_dataset, epochs=epochs, batch_size=batch_size,
                                  learning_rate=learning_rate, diffusion_loss_weight=diffusion_loss_weight, 
                                  t_diffusion=t_diffusion, kl_annealing=True,
                                  beta_min=0.0, beta_max=1.0, patience=patience)
    diffusion_time = time.time() - start_time
    print(f"Diffusion model training completed in {diffusion_time:.2f} seconds")
    
    # Evaluate models
    print("\nEvaluating models on test dataset...")
    baseline_metrics = evaluate_model(baseline_model, test_dataset, t_diffusion)
    diffusion_metrics = evaluate_model(diffusion_model, test_dataset, t_diffusion)
    
    print("\nTest Results:")
    print(f"Baseline Model - Recon Loss: {baseline_metrics[0]:.4f}, KL Loss: {baseline_metrics[1]:.4f}, Diffusion Distance: {baseline_metrics[2]:.4f}")
    print(f"Diffusion Model - Recon Loss: {diffusion_metrics[0]:.4f}, KL Loss: {diffusion_metrics[1]:.4f}, Diffusion Distance: {diffusion_metrics[2]:.4f}")
    
    # Calculate improvement percentages
    recon_improvement = (baseline_metrics[0] - diffusion_metrics[0]) / baseline_metrics[0] * 100
    kl_change = (diffusion_metrics[1] - baseline_metrics[1]) / baseline_metrics[1] * 100
    diffusion_improvement = (baseline_metrics[2] - diffusion_metrics[2]) / baseline_metrics[2] * 100
    
    print("\nImprovement Analysis:")
    print(f"Reconstruction Loss: {recon_improvement:.2f}% {'improvement' if recon_improvement > 0 else 'degradation'}")
    print(f"KL Divergence: {abs(kl_change):.2f}% {'increase' if kl_change > 0 else 'decrease'}")
    print(f"Diffusion Distance: {diffusion_improvement:.2f}% {'improvement' if diffusion_improvement > 0 else 'degradation'}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Plot loss curves with KL weights
    kl_weights = baseline_losses[4]  # KL weights from baseline training
    loss_plt = plot_loss_curves(baseline_losses[:4], diffusion_losses[:4], kl_weights)
    loss_plt.savefig('loss_comparison.png')
    print("Loss curves saved as 'loss_comparison.png'")
    
    # Sample a test graph for visualization
    sample_idx = 0
    sample_graph = test_dataset[sample_idx]
    
    # Compare reconstructions
    recon_plt, matrix_plt = compare_reconstructions(baseline_model, diffusion_model, sample_graph)
    recon_plt.savefig('graph_reconstructions.png')
    matrix_plt.savefig('adjacency_matrices.png')
    print("Graph reconstructions saved as 'graph_reconstructions.png'")
    print("Adjacency matrices saved as 'adjacency_matrices.png'")
    
    # Visualize latent spaces
    baseline_latent_plt = visualize_latent_space(baseline_model, test_dataset)
    baseline_latent_plt.savefig('baseline_latent_space.png')
    
    diffusion_latent_plt = visualize_latent_space(diffusion_model, test_dataset)
    diffusion_latent_plt.savefig('diffusion_latent_space.png')
    print("Latent space visualizations saved")
    
    print("\nBenchmark completed!")

if __name__ == "__main__":
    main()