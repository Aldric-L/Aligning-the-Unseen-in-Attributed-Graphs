import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class RBFLayer(nn.Module):
    """
    Radial Basis Function layer for precision estimation
    """
    def __init__(self, latent_dim, n_centers, a=1.0, min_precision=1e-6):
        super(RBFLayer, self).__init__()
        
        self.latent_dim = latent_dim
        self.n_centers = n_centers
        self.a = a
        
        # Initialize centers randomly (will be set using KMeans during fit)
        self.centers = nn.Parameter(torch.randn(n_centers, latent_dim), requires_grad=False)
        
        # Initialize lambdas (bandwidths)
        self.lambdas = nn.Parameter(torch.ones(n_centers), requires_grad=False)
        
        # Learnable weights for the RBF outputs - initialized to ones
        self.weights = nn.Parameter(torch.ones(n_centers), requires_grad=True)
        
        # Minimum precision to avoid division by zero
        self.zeta = nn.Parameter(torch.tensor(min_precision), requires_grad=False)
        
    def fit_centers(self, latent_samples):
        """Fit the RBF centers and bandwidths using KMeans on latent samples"""
        # Convert to numpy for KMeans
        latent_np = latent_samples.detach().cpu().numpy()
        
        # Fit KMeans to find centers
        kmeans = KMeans(n_clusters=self.n_centers, random_state=42)
        cluster_assignments = kmeans.fit_predict(latent_np)
        centers = kmeans.cluster_centers_
        
        # Update centers
        self.centers.data = torch.tensor(centers, dtype=torch.float32)
        
        # Compute bandwidths (lambdas)
        lambdas = np.zeros(self.n_centers)
        for k in range(self.n_centers):
            # Get points in cluster k
            cluster_points = latent_np[cluster_assignments == k]
            if len(cluster_points) > 0:
                # Calculate distances to center
                dists = np.linalg.norm(cluster_points - centers[k], axis=1)
                # Calculate lambda according to the formula
                lambdas[k] = 0.5 * (self.a / len(cluster_points)) * np.sum(dists)
            else:
                # Default lambda for empty clusters
                lambdas[k] = 1.0
            
        # Ensure lambdas are positive
        lambdas = np.maximum(lambdas, 1e-3)
        self.lambdas.data = torch.tensor(lambdas, dtype=torch.float32)
        
    def forward(self, z):
        """Forward pass through the RBF layer"""
        # Compute squared distances to all centers: [batch_size, n_centers]
        z_expanded = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
        centers_expanded = self.centers.unsqueeze(0)  # [1, n_centers, latent_dim]
        squared_dists = torch.sum((z_expanded - centers_expanded) ** 2, dim=2)  # [batch_size, n_centers]
        
        # Apply RBF activation: v_k(z) = exp(-lambda_k * ||z - c_k||^2)
        activations = torch.exp(-self.lambdas.unsqueeze(0) * squared_dists)  # [batch_size, n_centers]
        
        # Calculate precision: β_ψ(z) = W*v(z) + ζ
        precision = torch.matmul(activations, nn.functional.relu(self.weights)) + self.zeta
        
        return precision


class PrecisionNetwork(nn.Module):
    """Network for estimating precision (1/variance) in the latent space"""
    def __init__(self, latent_dim, n_centers=10, a=1.0, min_precision=1e-6):
        super(PrecisionNetwork, self).__init__()
        self.rbf_layer = RBFLayer(latent_dim, n_centers, a, min_precision)
        
    def fit(self, latent_samples):
        """Fit the RBF centers using latent samples"""
        self.rbf_layer.fit_centers(latent_samples)
        
    def forward(self, z):
        """Forward pass to compute precision (1/variance) for latent points"""
        return self.rbf_layer(z)
    
    def get_variance(self, z):
        """Compute variance (1/precision) for latent points"""
        precision = self.forward(z)
        # Add small epsilon to avoid division by zero
        return 1.0 / (precision + 1e-10)


def optimize_rbf_weights(precision_net, latent_samples, n_epochs=100, lr=0.01):
    """
    Train the RBF weights to better fit the latent samples
    
    This is a simple optimization that maximizes precision in areas with data,
    and minimizes it in areas without data.
    """
    optimizer = torch.optim.Adam([precision_net.rbf_layer.weights], lr=lr)
    
    # Create a grid of points for the "no data" regions
    x_min, x_max = latent_samples[:, 0].min().item() - 2, latent_samples[:, 0].max().item() + 2
    y_min, y_max = latent_samples[:, 1].min().item() - 2, latent_samples[:, 1].max().item() + 2
    
    grid_size = 10
    x_grid = torch.linspace(x_min, x_max, grid_size)
    y_grid = torch.linspace(y_min, y_max, grid_size)
    
    grid_points = []
    for x in x_grid:
        for y in y_grid:
            grid_points.append([x, y])
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    
    # Create weights for each grid point based on distance to nearest latent sample
    weights = torch.ones(grid_tensor.shape[0])
    for i, point in enumerate(grid_tensor):
        # Calculate distance to each latent sample
        dists = torch.norm(latent_samples - point.unsqueeze(0), dim=1)
        min_dist = torch.min(dists).item()
        
        # Points far from any latent sample get lower weights
        # FIXED: Use float instead of trying to apply torch.exp to a scalar
        weights[i] = float(np.exp(-min_dist * 0.5))
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Compute precision for all points
        precision_data = precision_net(latent_samples)
        precision_grid = precision_net(grid_tensor)
        
        # Loss: maximize precision where there is data, and weight by distance for grid points
        loss = -torch.mean(precision_data) + torch.mean(precision_grid * weights)
        
        loss.backward()
        optimizer.step()
        
        # Apply ReLU to weights to ensure positivity
        with torch.no_grad():
            precision_net.rbf_layer.weights.data = nn.functional.relu(precision_net.rbf_layer.weights.data)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")


def visualize_2dlatent_space_precision(latent_samples, n_centers=5, a=1.0, optimize_weights=True, n_epochs=100):
    """
    Visualize precision and variance in a 2D latent space
    
    Args:
        latent_samples: Tensor of shape [n_samples, 2] containing latent representations
        n_centers: Number of RBF centers to use
        a: Hyperparameter for bandwidth calculation
        optimize_weights: Whether to optimize RBF weights
    """
    # Ensure input is a torch tensor
    if not isinstance(latent_samples, torch.Tensor):
        latent_samples = torch.tensor(latent_samples, dtype=torch.float32)
    
    # Create and fit the precision network
    precision_net = PrecisionNetwork(latent_dim=2, n_centers=n_centers, a=a)
    precision_net.fit(latent_samples)
    
    # Optionally optimize the weights
    if optimize_weights:
        optimize_rbf_weights(precision_net, latent_samples, n_epochs=n_epochs)
    
    # Create a grid for visualization
    x_min, x_max = latent_samples[:, 0].min().item() - 3, latent_samples[:, 0].max().item() + 3
    y_min, y_max = latent_samples[:, 1].min().item() - 3, latent_samples[:, 1].max().item() + 3
    
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid and convert to tensor
    grid_points = np.column_stack([X.flatten(), Y.flatten()])
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    
    # Compute precision and variance over the grid
    with torch.no_grad():
        precision_values = precision_net(grid_tensor).numpy()
        variance_values = precision_net.get_variance(grid_tensor).numpy()
    
    # Reshape for plotting
    precision_grid = precision_values.reshape(X.shape)
    variance_grid = variance_values.reshape(X.shape)
    
    # Create a 2x2 subplot figure
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Data points and RBF centers
    ax1 = fig.add_subplot(221)
    ax1.scatter(latent_samples[:, 0].numpy(), latent_samples[:, 1].numpy(), alpha=0.6, s=10, label='Latent Samples')
    ax1.scatter(precision_net.rbf_layer.centers[:, 0].numpy(), 
                precision_net.rbf_layer.centers[:, 1].numpy(), 
                c='red', s=100, marker='*', label='RBF Centers')
    
    # Add lambda values as text
    for i, (x, y) in enumerate(precision_net.rbf_layer.centers.numpy()):
        lambda_val = precision_net.rbf_layer.lambdas[i].item()
        weight_val = precision_net.rbf_layer.weights[i].item()
        ax1.annotate(f'λ={lambda_val:.2f}\nw={weight_val:.2f}', (x, y), xytext=(10, 5), textcoords='offset points')
    
    ax1.set_title('Latent Space Samples and RBF Centers', fontsize=14)
    ax1.legend()
    ax1.set_xlabel('Latent Dimension 1', fontsize=12)
    ax1.set_ylabel('Latent Dimension 2', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Precision as a contour plot (2D)
    ax2 = fig.add_subplot(222)
    contour = ax2.contourf(X, Y, precision_grid, levels=20, cmap='viridis')
    ax2.scatter(latent_samples[:, 0].numpy(), latent_samples[:, 1].numpy(), alpha=0.2, s=10, c='white')
    ax2.scatter(precision_net.rbf_layer.centers[:, 0].numpy(), 
                precision_net.rbf_layer.centers[:, 1].numpy(), 
                c='red', s=100, marker='*')
    ax2.set_title('Precision Estimate (1/σ²)', fontsize=14)
    ax2.set_xlabel('Latent Dimension 1', fontsize=12)
    ax2.set_ylabel('Latent Dimension 2', fontsize=12)
    fig.colorbar(contour, ax=ax2, label='Precision Value')
    
    # Plot 3: Variance as a contour plot (2D)
    ax3 = fig.add_subplot(223)
    contour = ax3.contourf(X, Y, variance_grid, levels=20, cmap='plasma')
    ax3.scatter(latent_samples[:, 0].numpy(), latent_samples[:, 1].numpy(), alpha=0.2, s=10, c='white')
    ax3.scatter(precision_net.rbf_layer.centers[:, 0].numpy(), 
                precision_net.rbf_layer.centers[:, 1].numpy(), 
                c='red', s=100, marker='*')
    ax3.set_title('Variance Estimate (σ²)', fontsize=14)
    ax3.set_xlabel('Latent Dimension 1', fontsize=12)
    ax3.set_ylabel('Latent Dimension 2', fontsize=12)
    fig.colorbar(contour, ax=ax3, label='Variance Value')
    
    # Plot 4: 3D surface of precision
    ax4 = fig.add_subplot(224, projection='3d')
    surf = ax4.plot_surface(X, Y, precision_grid, cmap='viridis', alpha=0.8, edgecolor='none')
    ax4.scatter(latent_samples[:, 0].numpy(), 
                latent_samples[:, 1].numpy(), 
                np.zeros_like(latent_samples[:, 0].numpy()), 
                alpha=0.2, s=10, c='black')
    ax4.set_title('Precision Estimate (3D View)', fontsize=14)
    ax4.set_xlabel('Latent Dimension 1', fontsize=12)
    ax4.set_ylabel('Latent Dimension 2', fontsize=12)
    ax4.set_zlabel('Precision Value', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('rbf_precision_visualization.png', dpi=300, bbox_inches='tight')
    
    return fig, precision_net
