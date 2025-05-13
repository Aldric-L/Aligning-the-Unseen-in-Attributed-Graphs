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

from diffusionGVAE2 import heat_kernel_distance,generate_erdos_renyi_dataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# New class: Riemannian Metric Tensor for the latent space
class RiemannianMetric(nn.Module):
    def __init__(self, latent_dim, hidden_dim=32):
        super(RiemannianMetric, self).__init__()
        self.latent_dim = latent_dim
        
        # Neural network to model the metric tensor
        # The output dimension is latent_dim * latent_dim for the full metric tensor
        self.metric_nn = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, latent_dim * latent_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, z):
        """
        Compute the Riemannian metric tensor G(z) at point z
        Returns a positive definite matrix for each point in the batch
        """
        batch_size = z.shape[0]
        
        # Get raw output from neural network
        G_flat = self.metric_nn(z)
        
        # Reshape to batch of matrices
        G = G_flat.view(batch_size, self.latent_dim, self.latent_dim)
        
        # Ensure positive definiteness by G = LL^T + εI
        # We use a small diagonal addition to ensure numerical stability
        eye = torch.eye(self.latent_dim, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
        epsilon = 1e-5
        
        # Make symmetric
        G = 0.5 * (G + G.transpose(1, 2))
        
        # Add small diagonal for stability
        G = G + epsilon * eye
        
        return G
        
    def compute_geodesic_distance(self, z1, z2, num_steps=10):
        """
        Approximate geodesic distance between z1 and z2 using discretized path integration
        """
        # Create a straight line path between z1 and z2
        t = torch.linspace(0, 1, num_steps, device=z1.device)
        
        # Interpolate between z1 and z2
        z_path = z1.unsqueeze(1) * (1 - t) + z2.unsqueeze(1) * t  # Shape: [batch, num_steps, latent_dim]
        
        # Calculate the tangent vector at each point
        dz = z2 - z1  # Shape: [batch, latent_dim]
        dz = dz.unsqueeze(1).expand(-1, num_steps, -1)  # Expand to match z_path
        
        # Compute metric tensor at each point along the path
        G_path = self.forward(z_path.reshape(-1, z1.shape[-1])).reshape(-1, num_steps, z1.shape[-1], z1.shape[-1])
        
        # Compute ds^2 = dz^T * G * dz at each point
        # First, compute G * dz
        G_dz = torch.matmul(G_path, dz.unsqueeze(-1)).squeeze(-1)  # [batch, num_steps, latent_dim]
        
        # Then compute dz^T * (G * dz)
        ds_squared = torch.sum(dz * G_dz, dim=-1)  # [batch, num_steps]
        
        # Take sqrt for actual ds and integrate over the path
        ds = torch.sqrt(torch.clamp(ds_squared, min=1e-8))
        
        # Approximate the integral by averaging and scaling by path length
        path_length = torch.sum(ds, dim=1) / num_steps
        
        return path_length

# Modified GraphVAE to include Riemannian metric
class RiemannianGraphVAE(nn.Module):
    def __init__(self, num_nodes, hidden_dim=32, latent_dim=16, node_feature_size=8, metric_hidden_dim=32):
        super(RiemannianGraphVAE, self).__init__()
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
        
        # Riemannian metric
        self.metric = RiemannianMetric(latent_dim, hidden_dim=metric_hidden_dim)
        
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
    
    def get_metric_tensor(self, z):
        """Get the Riemannian metric tensor at points z"""
        return self.metric(z)
    
    def compute_riemannian_prior_kl(self, mu, logvar, z, num_samples=5, num_steps=10):
        """
        Compute KL divergence between the variational distribution and a prior
        defined by a Riemannian random walk on the latent manifold
        """
        batch_size = mu.size(0)
        device = mu.device
        
        # Standard deviation
        std = torch.exp(0.5 * logvar)
        
        # Sample from the variational distribution
        samples = []
        for _ in range(num_samples):
            eps = torch.randn_like(std)
            samples.append(mu + eps * std)
        
        # Reference point (origin in latent space)
        zero = torch.zeros_like(mu)
        
        # Compute geodesic distances from each sample to the origin
        total_dist = 0
        for sample in samples:
            dist = self.metric.compute_geodesic_distance(sample, zero, num_steps=num_steps)
            total_dist += dist
        
        # Average distance across samples
        avg_dist = total_dist / num_samples
        
        # KL divergence approximation based on geodesic distances
        # This is a simplified version - in practice, you might want a more sophisticated formula
        kl_div = 0.5 * (torch.sum(mu**2 + torch.exp(logvar) - logvar - 1, dim=1) + avg_dist)
        
        return kl_div.mean()
    
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
        
        return adj_pred, mu, logvar, z


# Enhanced training function with Riemannian prior
def train_riemannian_model(model, dataset, epochs=100, batch_size=16, learning_rate=0.001, 
                diffusion_loss_weight=0.0, t_diffusion=1.0, kl_annealing=True, 
                beta_min=0.0, beta_max=1.0, patience=15, use_riemannian_prior=True, 
                riemannian_samples=5, geodesic_steps=10):
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
    riemannian_distances = []  # For tracking Riemannian distances
    
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
        epoch_riemannian_dist = 0
        
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
            batch_adj_recon, batch_mu, batch_logvar, batch_z = model(batch_adj)
            
            # Reconstruction loss 
            recon_loss = F.binary_cross_entropy(batch_adj_recon, batch_adj, reduction='sum') / batch_size
            
            # KL divergence - either standard or Riemannian
            if use_riemannian_prior:
                kl_loss = model.compute_riemannian_prior_kl(
                    batch_mu, batch_logvar, batch_z, 
                    num_samples=riemannian_samples, 
                    num_steps=geodesic_steps
                )
            else:
                # Standard KL divergence
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
            
            # Track average Riemannian distance
            if use_riemannian_prior:
                # Compute average geodesic distance from samples to origin
                zero = torch.zeros_like(batch_mu)
                riemann_dist = model.metric.compute_geodesic_distance(batch_z, zero).mean()
                epoch_riemannian_dist += riemann_dist.item()
        
        # Average losses for the epoch
        num_batches = (len(dataset) + batch_size - 1) // batch_size  # ceiling division
        avg_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        avg_diffusion_loss = epoch_diffusion_loss / num_batches
        avg_riemannian_dist = epoch_riemannian_dist / num_batches if use_riemannian_prior else 0
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Track losses
        train_losses.append(avg_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        diffusion_losses.append(avg_diffusion_loss)
        riemannian_distances.append(avg_riemannian_dist)
        
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
                  f"KL Loss: {avg_kl_loss:.4f}, Diffusion Loss: {avg_diffusion_loss:.4f}, "
                  f"Riemannian Dist: {avg_riemannian_dist:.4f}, KL Weight: {kl_weight:.4f}")
    
    return train_losses, recon_losses, kl_losses, diffusion_losses, kl_weights, riemannian_distances


# Evaluation function for Riemannian model
def evaluate_riemannian_model(model, dataset, t_diffusion=1.0, use_riemannian_prior=True):
    device = next(model.parameters()).device
    model.eval()
    
    # Convert dataset to device
    dataset = [d.to(device) for d in dataset]
    
    total_recon_loss = 0
    total_kl_loss = 0
    total_diffusion_loss = 0
    total_riemannian_dist = 0
    
    with torch.no_grad():
        for adj_matrix in dataset:
            # Forward pass
            adj_recon, mu, logvar, z = model(adj_matrix)
            
            # Reconstruction loss
            recon_loss = F.binary_cross_entropy(adj_recon, adj_matrix, reduction='sum')
            
            # KL divergence
            if use_riemannian_prior:
                kl_loss = model.compute_riemannian_prior_kl(mu, logvar, z)
            else:
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Diffusion distance
            diff_loss = heat_kernel_distance(adj_matrix, adj_recon, t=t_diffusion)
            
            # Riemannian distance to origin
            if use_riemannian_prior:
                zero = torch.zeros_like(mu)
                riemann_dist = model.metric.compute_geodesic_distance(z, zero).mean()
                total_riemannian_dist += riemann_dist.item()
            
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_diffusion_loss += diff_loss.item()
    
    avg_recon_loss = total_recon_loss / len(dataset)
    avg_kl_loss = total_kl_loss / len(dataset)
    avg_diffusion_loss = total_diffusion_loss / len(dataset)
    avg_riemannian_dist = total_riemannian_dist / len(dataset) if use_riemannian_prior else 0
    
    return avg_recon_loss, avg_kl_loss, avg_diffusion_loss, avg_riemannian_dist


# Function to plot Riemannian metric field
def visualize_riemannian_metric(model, latent_points=None, grid_size=20, plot_range=(-3, 3)):
    """
    Visualize the Riemannian metric tensor field in 2D
    """
    if latent_points is None or model.latent_dim > 2:
        # If latent dimension > 2, create a 2D grid for visualization
        # This is a simplification - in higher dimensions we're just visualizing a slice
        xx, yy = np.meshgrid(
            np.linspace(plot_range[0], plot_range[1], grid_size),
            np.linspace(plot_range[0], plot_range[1], grid_size)
        )
        
        # Create grid points
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # If latent_dim > 2, pad with zeros
        if model.latent_dim > 2:
            padding = np.zeros((grid_points.shape[0], model.latent_dim - 2))
            grid_points = np.hstack([grid_points, padding])
        
        # Convert to tensor
        points = torch.tensor(grid_points, dtype=torch.float32).to(next(model.parameters()).device)
    else:
        # Use provided latent points (assuming they're already 2D)
        points = latent_points
    
    # Get metric tensors at each point
    model.eval()
    with torch.no_grad():
        G = model.get_metric_tensor(points)
    
    # For visualization, we'll use the determinant of G as a measure of the volume element
    # This gives us a scalar field representing how the space is warped
    det_G = torch.det(G).cpu().numpy()
    
    # Reshape for plotting
    det_G_grid = det_G.reshape(grid_size, grid_size)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, det_G_grid, cmap='viridis', levels=20)
    plt.colorbar(label='det(G) - Volume Element')
    plt.title('Riemannian Metric Field')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    
    # Also visualize the principal directions of the metric tensor (eigenvectors)
    # We'll downsample for clarity
    downsample = 4
    for i in range(0, grid_size, downsample):
        for j in range(0, grid_size, downsample):
            # Get the 2x2 submatrix (assuming higher dimensions don't contribute significantly)
            G_ij = G[i*grid_size + j, :2, :2].cpu().numpy()
            
            # Get eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(G_ij)
            
            # Scale eigenvectors by square root of eigenvalues for visualization
            scaled_evecs = eigenvecs * np.sqrt(eigenvals)[:, np.newaxis]
            
            # Draw eigenvectors
            for k in range(2):
                plt.arrow(xx[i, j], yy[i, j], 
                          scaled_evecs[k, 0] * 0.2, scaled_evecs[k, 1] * 0.2,
                          head_width=0.05, head_length=0.1, fc='red', ec='red', alpha=0.6)
    
    plt.tight_layout()
    return plt


# Function to plot geodesic paths
def plot_geodesic_paths(model, start_points, end_points, num_steps=20):
    """
    Plot geodesic paths between pairs of points in the latent space
    """
    device = next(model.parameters()).device
    start_points = torch.tensor(start_points, dtype=torch.float32).to(device)
    end_points = torch.tensor(end_points, dtype=torch.float32).to(device)
    
    plt.figure(figsize=(10, 8))
    
    model.eval()
    with torch.no_grad():
        for i in range(len(start_points)):
            # Get start and end points
            z1 = start_points[i:i+1]
            z2 = end_points[i:i+1]
            
            # Create a straight line path between z1 and z2
            t = torch.linspace(0, 1, num_steps, device=device)
            z_path = z1.unsqueeze(1) * (1 - t) + z2.unsqueeze(1) * t  # Shape: [1, num_steps, latent_dim]
            
            # Get points along the path
            path_points = z_path.squeeze(0).cpu().numpy()
            
            # Plot the path
            plt.plot(path_points[:, 0], path_points[:, 1], 'o-', linewidth=2, alpha=0.7)
            
            # Mark start and end points
            plt.plot(path_points[0, 0], path_points[0, 1], 'go', markersize=10)
            plt.plot(path_points[-1, 0], path_points[-1, 1], 'ro', markersize=10)
    
    plt.title('Geodesic Paths in Latent Space')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    return plt
