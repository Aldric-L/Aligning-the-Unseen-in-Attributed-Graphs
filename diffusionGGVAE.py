import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import scipy.linalg as sla

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class GraphDataset:
    def __init__(self, num_graphs=1000, min_nodes=20, max_nodes=50, p=0.2, weighted=True):
        """
        Generate synthetic graph dataset using Erdos-Renyi model
        
        Args:
            num_graphs: Number of graphs to generate
            min_nodes: Minimum number of nodes in each graph
            max_nodes: Maximum number of nodes in each graph
            p: Probability of edge creation
            weighted: Whether to create weighted edges
        """
        self.num_graphs = num_graphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.p = p
        self.weighted = weighted
        self.graphs = []
        self.generate_graphs()
        
    def generate_graphs(self):
        """Generate synthetic graphs"""
        for _ in range(self.num_graphs):
            n_nodes = np.random.randint(self.min_nodes, self.max_nodes + 1)
            
            # Generate Erdos-Renyi graph as adjacency matrix
            adj_matrix = np.random.rand(n_nodes, n_nodes)
            adj_matrix = (adj_matrix < self.p).astype(np.float32)
            
            # Make the graph undirected (symmetric adjacency matrix)
            adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
            
            # Set diagonal to 0 (no self-loops)
            np.fill_diagonal(adj_matrix, 0)
            
            # Generate weights if needed
            if self.weighted:
                # Only apply weights where edges exist
                weights = np.random.rand(n_nodes, n_nodes).astype(np.float32)
                adj_matrix = adj_matrix * weights
                # Make weights symmetric
                adj_matrix = (adj_matrix + adj_matrix.T) / 2
            
            # Convert to torch tensor and create graph
            adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
            edge_index, edge_attr = dense_to_sparse(adj_matrix_tensor)
            
            data = Data(
                x=torch.ones(n_nodes, 1),  # Node features (all ones for simplicity)
                edge_index=edge_index,
                edge_attr=edge_attr,
                adj=adj_matrix_tensor,
                num_nodes=n_nodes
            )
            
            self.graphs.append(data)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]


class Encoder(nn.Module):
    def __init__(self, hidden_dim=32, latent_dim=16):
        """
        Graph Encoder using GCN layers
        
        Args:
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of latent space
        """
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # GCN layers
        self.conv1 = GCNConv(1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Apply GCN layers
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        # Global pooling (mean of all node features)
        x = torch.mean(x, dim=0)
        
        # Get mean and log variance for the VAE
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=32, max_nodes=50):
        """
        Graph Decoder
        
        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Dimension of hidden layers
            max_nodes: Maximum number of nodes in a graph
        """
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        
        # Fully connected layers
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, max_nodes * max_nodes)
        
    def forward(self, z, num_nodes):
        """
        Args:
            z: Latent vector
            num_nodes: Number of nodes in the graph
        """
        # Apply fully connected layers
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        
        # Reshape to get adjacency matrix
        adj_matrix = h.view(self.max_nodes, self.max_nodes)
        
        # Make the adjacency matrix symmetric (undirected graph)
        adj_matrix = (adj_matrix + adj_matrix.t()) / 2
        
        # Zero out the diagonal (no self-loops)
        adj_matrix = adj_matrix - torch.diag(torch.diag(adj_matrix))
        
        # Get only the submatrix corresponding to the actual number of nodes
        adj_matrix = adj_matrix[:num_nodes, :num_nodes]
        
        return adj_matrix


class GraphVAE(nn.Module):
    def __init__(self, hidden_dim=32, latent_dim=16, max_nodes=50):
        """
        Graph Variational Autoencoder
        
        Args:
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of latent space
            max_nodes: Maximum number of nodes in a graph
        """
        super(GraphVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, max_nodes)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, data):
        # Encode
        mu, logvar = self.encoder(data)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        adj_matrix_recon = self.decoder(z, data.num_nodes)
        
        return adj_matrix_recon, mu, logvar


class DiffusionDistance:
    """
    Compute diffusion distance (heat kernel) between two graphs
    """
    def __init__(self, t=1.0):
        """
        Args:
            t: Diffusion time (controls how far heat diffuses)
        """
        self.t = t
    
    def compute_heat_kernel(self, adj_matrix):
        """
        Compute heat kernel: exp(-t*L) where L is the graph Laplacian
        """
        # Create Laplacian matrix
        degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))
        laplacian = degree_matrix - adj_matrix
        
        # Convert to numpy for scipy operations
        laplacian_np = laplacian.detach().cpu().numpy()
        
        # Compute exp(-t*L)
        heat_kernel = sla.expm(-self.t * laplacian_np)
        
        # Convert back to torch tensor
        heat_kernel = torch.tensor(heat_kernel, dtype=torch.float32)
        
        return heat_kernel
    
    def __call__(self, adj_matrix_orig, adj_matrix_recon):
        """
        Compute diffusion distance between two graphs
        
        Args:
            adj_matrix_orig: Original adjacency matrix
            adj_matrix_recon: Reconstructed adjacency matrix
        """
        # Compute heat kernels
        heat_kernel_orig = self.compute_heat_kernel(adj_matrix_orig)
        heat_kernel_recon = self.compute_heat_kernel(adj_matrix_recon)
        
        # Compute Frobenius norm of difference
        diff = heat_kernel_orig - heat_kernel_recon
        distance = torch.norm(diff, p='fro')
        
        return distance


class KLAnnealer:
    """
    KL annealing schedule to gradually increase the weight of KL divergence term
    """
    def __init__(self, epochs, start=0.0, end=1.0, strategy='linear'):
        """
        Args:
            epochs: Total number of epochs
            start: Starting weight
            end: Ending weight
            strategy: Annealing strategy ('linear', 'sigmoid', 'cyclical')
        """
        self.epochs = epochs
        self.start = start
        self.end = end
        self.strategy = strategy
        
        if strategy == 'linear':
            self.weights = self._linear_schedule()
        elif strategy == 'sigmoid':
            self.weights = self._sigmoid_schedule()
        elif strategy == 'cyclical':
            self.weights = self._cyclical_schedule()
        else:
            raise ValueError(f"Unknown annealing strategy: {strategy}")
    
    def _linear_schedule(self):
        return np.linspace(self.start, self.end, self.epochs)
    
    def _sigmoid_schedule(self):
        # Sigmoid function for smoother transition
        x = np.linspace(-6, 6, self.epochs)
        y = 1 / (1 + np.exp(-x))
        return self.start + (self.end - self.start) * y
    
    def _cyclical_schedule(self):
        # Cyclical scheduling: gradually increases weight, then resets
        n_cycles = 4
        weights = []
        for i in range(n_cycles):
            cycle_len = self.epochs // n_cycles
            cycle_weights = np.linspace(self.start, self.end, cycle_len)
            weights.extend(cycle_weights)
        return np.array(weights[:self.epochs])
    
    def get_weight(self, epoch):
        """Get KL weight for current epoch"""
        return self.weights[epoch]



def train_vae(model, dataset, optimizer, kl_annealer, epochs=100, batch_size=32, 
              use_diffusion_loss=False, diffusion_weight=1.0):
    """
    Train the Graph VAE model
    
    Args:
        model: VAE model
        dataset: Graph dataset
        optimizer: Optimizer
        kl_annealer: KL annealing scheduler
        epochs: Number of epochs
        batch_size: Batch size
        use_diffusion_loss: Whether to use diffusion distance in loss
        diffusion_weight: Weight of diffusion distance term in loss
    """
    model.train()
    losses = []
    recon_losses = []
    kl_losses = []
    diffusion_losses = []
    
    diffusion_dist = DiffusionDistance(t=1.0)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_diff_loss = 0.0
        
        # Get KL weight for current epoch
        kl_weight = kl_annealer.get_weight(epoch)
        
        # Shuffle the dataset
        indices = torch.randperm(len(dataset))
        
        # Process in batches
        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_loss = 0.0
            batch_recon_loss = 0.0
            batch_kl_loss = 0.0
            batch_diff_loss = 0.0
            
            optimizer.zero_grad()
            
            # Process each graph in the batch
            for idx in batch_indices:
                data = dataset[idx]
                
                # Forward pass
                adj_matrix_recon, mu, logvar = model(data)
                
                # Reconstruction loss (MSE)
                recon_loss = F.mse_loss(adj_matrix_recon, data.adj)
                
                # KL divergence loss
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                # Total loss
                loss = recon_loss + kl_weight * kl_loss
                
                # Add diffusion distance if specified
                diff_loss = torch.tensor(0.0)
                if use_diffusion_loss:
                    diff_loss = diffusion_dist(data.adj, adj_matrix_recon)
                    loss += diffusion_weight * diff_loss
                
                # Accumulate batch loss
                batch_loss += loss
                batch_recon_loss += recon_loss
                batch_kl_loss += kl_loss
                batch_diff_loss += diff_loss
            
            # Average over batch
            batch_loss /= len(batch_indices)
            batch_recon_loss /= len(batch_indices)
            batch_kl_loss /= len(batch_indices)
            if use_diffusion_loss:
                batch_diff_loss /= len(batch_indices)
            
            # Backward pass and optimization
            batch_loss.backward()
            optimizer.step()
            
            # Accumulate epoch loss
            epoch_loss += batch_loss.item()
            epoch_recon_loss += batch_recon_loss.item()
            epoch_kl_loss += batch_kl_loss.item()
            if use_diffusion_loss:
                epoch_diff_loss += batch_diff_loss.item()
            else:
                epoch_diff_loss += batch_diff_loss  # Already a float value
        
        # Average over batches
        epoch_loss /= (len(dataset) // batch_size + 1)
        epoch_recon_loss /= (len(dataset) // batch_size + 1)
        epoch_kl_loss /= (len(dataset) // batch_size + 1)
        epoch_diff_loss /= (len(dataset) // batch_size + 1)
        
        # Store losses
        losses.append(epoch_loss)
        recon_losses.append(epoch_recon_loss)
        kl_losses.append(epoch_kl_loss)
        diffusion_losses.append(epoch_diff_loss)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}, " +
                  f"Recon: {epoch_recon_loss:.4f}, KL: {epoch_kl_loss:.4f}, " +
                  f"Diff: {epoch_diff_loss:.4f}, KL Weight: {kl_weight:.4f}")
    
    return {
        'loss': losses,
        'recon_loss': recon_losses,
        'kl_loss': kl_losses,
        'diffusion_loss': diffusion_losses
    }



def evaluate_model(model, dataset, sample_size=10):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained VAE model
        dataset: Test dataset
        sample_size: Number of samples to visualize
    """
    model.eval()
    mse_losses = []
    diffusion_losses = []
    
    diffusion_dist = DiffusionDistance(t=1.0)
    
    with torch.no_grad():
        for i in range(min(sample_size, len(dataset))):
            data = dataset[i]
            
            # Forward pass
            adj_matrix_recon, _, _ = model(data)
            
            # Calculate losses
            mse_loss = F.mse_loss(adj_matrix_recon, data.adj).item()
            diff_loss = diffusion_dist(data.adj, adj_matrix_recon).item()
            
            mse_losses.append(mse_loss)
            diffusion_losses.append(diff_loss)
    
    return {
        'mse': np.mean(mse_losses),
        'diffusion': np.mean(diffusion_losses),
        'mse_std': np.std(mse_losses),
        'diffusion_std': np.std(diffusion_losses)
    }


def visualize_graphs(model, dataset, num_samples=3):
    """
    Visualize original and reconstructed graphs
    
    Args:
        model: Trained VAE model
        dataset: Dataset
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    plt.figure(figsize=(15, 5 * num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            data = dataset[i]
            
            # Get original adjacency matrix
            adj_orig = data.adj.detach().cpu().numpy()
            
            # Get reconstructed adjacency matrix
            adj_recon, _, _ = model(data)
            adj_recon = adj_recon.detach().cpu().numpy()
            
            # Create NetworkX graphs
            G_orig = nx.from_numpy_array(adj_orig)
            G_recon = nx.from_numpy_array(adj_recon)
            
            # Plot original graph
            plt.subplot(num_samples, 2, 2*i + 1)
            plt.title(f"Original Graph {i+1}")
            pos = nx.spring_layout(G_orig, seed=42)
            nx.draw_networkx_nodes(G_orig, pos, node_color='b', node_size=100, alpha=0.8)
            
            # Use edge weights for line width if available
            weights = np.array([G_orig[u][v]['weight'] for u, v in G_orig.edges()])
            if len(weights) > 0:
                max_weight = weights.max()
                normalized_weights = weights / max_weight * 3
                nx.draw_networkx_edges(G_orig, pos, width=normalized_weights, alpha=0.5)
            else:
                nx.draw_networkx_edges(G_orig, pos, alpha=0.5)
            
            # Plot reconstructed graph
            plt.subplot(num_samples, 2, 2*i + 2)
            plt.title(f"Reconstructed Graph {i+1}")
            nx.draw_networkx_nodes(G_recon, pos, node_color='r', node_size=100, alpha=0.8)
            
            # Use edge weights for line width if available
            weights = np.array([G_recon[u][v]['weight'] for u, v in G_recon.edges()])
            if len(weights) > 0:
                max_weight = weights.max()
                normalized_weights = weights / max_weight * 3
                nx.draw_networkx_edges(G_recon, pos, width=normalized_weights, alpha=0.5)
            else:
                nx.draw_networkx_edges(G_recon, pos, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('graph_reconstruction.png')
    plt.close()


def visualize_losses(baseline_losses, diffusion_losses):
    """
    Visualize training losses
    
    Args:
        baseline_losses: Losses from baseline model
        diffusion_losses: Losses from diffusion model
    """
    plt.figure(figsize=(15, 10))
    
    # Plot total loss
    plt.subplot(2, 2, 1)
    plt.plot(baseline_losses['loss'], label='Baseline')
    plt.plot(diffusion_losses['loss'], label='Diffusion')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot reconstruction loss
    plt.subplot(2, 2, 2)
    plt.plot(baseline_losses['recon_loss'], label='Baseline')
    plt.plot(diffusion_losses['recon_loss'], label='Diffusion')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot KL loss
    plt.subplot(2, 2, 3)
    plt.plot(baseline_losses['kl_loss'], label='Baseline')
    plt.plot(diffusion_losses['kl_loss'], label='Diffusion')
    plt.title('KL Divergence Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot diffusion loss
    plt.subplot(2, 2, 4)
    plt.plot(diffusion_losses['diffusion_loss'], label='Diffusion')
    plt.title('Diffusion Distance Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_losses.png')
    plt.close()


def visualize_latent_space(baseline_model, diffusion_model, dataset, n_samples=100):
    """
    Visualize latent space using t-SNE
    
    Args:
        baseline_model: Baseline VAE model
        diffusion_model: Diffusion VAE model
        dataset: Dataset
        n_samples: Number of samples to visualize
    """
    from sklearn.manifold import TSNE
    
    baseline_model.eval()
    diffusion_model.eval()
    
    # Extract latent representations
    baseline_latents = []
    diffusion_latents = []
    
    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            data = dataset[i]
            
            # Baseline model
            mu_baseline, _ = baseline_model.encoder(data)
            baseline_latents.append(mu_baseline.detach().cpu().numpy())
            
            # Diffusion model
            mu_diffusion, _ = diffusion_model.encoder(data)
            diffusion_latents.append(mu_diffusion.detach().cpu().numpy())
    
    # Convert to numpy arrays
    baseline_latents = np.array(baseline_latents)
    diffusion_latents = np.array(diffusion_latents)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    baseline_tsne = tsne.fit_transform(baseline_latents)
    diffusion_tsne = tsne.fit_transform(diffusion_latents)
    
    # Visualize
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(baseline_tsne[:, 0], baseline_tsne[:, 1], c='blue', alpha=0.7)
    plt.title('Baseline Model Latent Space')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    plt.subplot(1, 2, 2)
    plt.scatter(diffusion_tsne[:, 0], diffusion_tsne[:, 1], c='red', alpha=0.7)
    plt.title('Diffusion Model Latent Space')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig('latent_space.png')
    plt.close()
