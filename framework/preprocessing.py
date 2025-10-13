import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from tqdm import tqdm

from framework.jacobianCache import JacobianGridCache

class PreprocessingLayer(nn.Module, ABC):
    """
    Abstract base class for preprocessing layers that transform latent codes
    before passing them to decoders.
    """
    def __init__(self, name: str = "base_preprocessor"):
        super(PreprocessingLayer, self).__init__()
        self.name = name
    
    @abstractmethod
    def forward(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Transform the latent code
        
        Args:
            z: Input latent code [num_nodes, latent_dim]
            **kwargs: Additional arguments
            
        Returns:
            Transformed latent code
        """
        pass


class DistancePreprocessingLayer(PreprocessingLayer):
    """
    Preprocessing layer that computes pairwise distances using custom metrics
    """
    def __init__(
        self,
        latent_dim: int,
        metric_function: Optional[Callable] = None,
        num_integration_points: int = 100,
        distance_mode: str = "linear_interpolation",  # or "direct"
        output_dim: Optional[int] = None,
        name: str = "distance_preprocessor"
    ):
        super(DistancePreprocessingLayer, self).__init__(name)
        self.latent_dim = latent_dim
        self.num_integration_points = num_integration_points
        self.distance_mode = distance_mode
        self.output_dim = output_dim if output_dim is not None else latent_dim
        
        # Store metric function (can be None for Euclidean fallback)
        self.metric_function = metric_function
        
        # Optional learnable transformation after distance computation
        if self.output_dim != latent_dim:
            self.transform = nn.Linear(latent_dim, self.output_dim)
        else:
            self.transform = nn.Identity()
    
    def linear_interpolation_distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute distance using linear interpolation: ∫₀¹ √((u-v)ᵀg(x(t))(u-v))dt
        where x(t) = tu + (1-t)v
        
        Args:
            u, v: Latent vectors [latent_dim]
            
        Returns:
            Distance scalar
        """
        def integrand(t):
            x_t = t * u + (1 - t) * v
            if self.metric_function is not None:
                try:
                    # Convert to numpy for custom metric function
                    x_t_np = x_t.detach().cpu().numpy()
                    g = self.metric_function(x_t_np)
                    g = torch.tensor(g, device=u.device, dtype=u.dtype)
                    diff = u - v
                    return torch.sqrt(torch.sum(diff * (g @ diff)))
                except:
                    # Fallback to Euclidean
                    return torch.norm(u - v)
            else:
                # Direct Euclidean distance
                return torch.norm(u - v)
        
        # Numerical integration using trapezoidal rule
        t_vals = torch.linspace(0, 1, self.num_integration_points, device=u.device)
        dt = 1.0 / (self.num_integration_points - 1)
        
        integrand_vals = torch.stack([integrand(t) for t in t_vals])
        integral = torch.trapz(integrand_vals, dx=dt)
        
        return integral
    
    def compute_pairwise_distances(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between all nodes
        
        Args:
            z: Latent embeddings [num_nodes, latent_dim]
            
        Returns:
            Distance matrix [num_nodes, num_nodes]
        """
        num_nodes = z.size(0)
        distances = torch.zeros(num_nodes, num_nodes, device=z.device, dtype=z.dtype)
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # Only compute upper triangle
                if self.distance_mode == "linear_interpolation":
                    dist = self.linear_interpolation_distance(z[i], z[j])
                else:  # direct mode
                    dist = torch.norm(z[i] - z[j])
                
                distances[i, j] = dist
                distances[j, i] = dist  # Symmetric
        
        return distances
    
    def forward(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Transform latent code using distance computation
        
        Args:
            z: Input latent code [num_nodes, latent_dim]
            
        Returns:
            Transformed representation based on distances
        """
        # Compute pairwise distances
        distance_matrix = self.compute_pairwise_distances(z)
        
        # Option 1: Use distance matrix directly as features
        # Flatten upper triangle (excluding diagonal)
        num_nodes = z.size(0)
        mask = torch.triu(torch.ones(num_nodes, num_nodes, device=z.device), diagonal=1).bool()
        distance_features = distance_matrix[mask]
        
        # Reshape to match expected input format
        # Each node gets the same distance vector (could be modified)
        distance_features = distance_features.unsqueeze(0).repeat(num_nodes, 1)
        
        # Apply optional transformation
        transformed = self.transform(distance_features)
        
        return transformed


class JacobianMetricPreprocessingLayer(PreprocessingLayer):
    """
    Preprocessing layer that computes pairwise distances using metrics derived
    from the Jacobian of another decoder (typically NodeAttributeDecoder).
    """
    def __init__(
        self,
        latent_dim: int,
        reference_decoder: "DecoderBase",
        reference_model: "GraphVAE",
        num_integration_points: int = 50,
        metric_regularization: float = 1e-6,
        distance_mode: str = "linear_interpolation",
        output_dim: Optional[int] = None,
        cache_jacobians: bool = True,
        name: str = "jacobian_metric_preprocessor", *args, **kwargs
    ):
        super(JacobianMetricPreprocessingLayer, self).__init__(name)
        self.latent_dim = latent_dim
        
        # CRITICAL FIX: Store references without registering as submodules
        # This prevents circular references during .to(device) calls
        object.__setattr__(self, '_reference_model', reference_model)
        object.__setattr__(self, '_reference_decoder', reference_decoder)
        
        self.num_integration_points = num_integration_points
        self.metric_regularization = metric_regularization
        self.distance_mode = distance_mode
        self.output_dim = output_dim if output_dim is not None else latent_dim
        self.cache_jacobians = cache_jacobians
        
        # Cache for Jacobians to avoid recomputation
        self._jacobian_cache = {}
        self._cache_valid = False
        
        # Optional learnable transformation after distance computation
        if self.output_dim != latent_dim:
            self.transform = nn.Linear(latent_dim, self.output_dim)
        else:
            self.transform = nn.Identity()

        self.grid_cache = JacobianGridCache(
            grid_resolution=kwargs.get('grid_resolution', 32),
            cache_radius=kwargs.get('cache_radius', 0.1),
            max_cache_size=kwargs.get('max_cache_size', 1000)
        )
    
    def compute_metric_tensor(self, z: torch.Tensor, node_idx: int) -> torch.Tensor:
        """
        Compute the metric tensor at a given point in latent space using the 
        Jacobian of the reference decoder.
        
        The metric tensor G is computed as: G = J^T @ J
        where J is the Jacobian of the reference decoder output w.r.t. latent variables.
        
        Args:
            z: Latent variables [num_nodes, latent_dim]
            node_idx: Index of the node to compute metric for
            
        Returns:
            Metric tensor [latent_dim, latent_dim]
        """
        # Check cache first
        cache_key = f"{node_idx}_{hash(z.data.tobytes()) if hasattr(z.data, 'tobytes') else id(z)}"
        if self.cache_jacobians and cache_key in self._jacobian_cache:
            return self._jacobian_cache[cache_key]
        
        try:
            # Compute Jacobian of reference decoder at this point        
            jacobian = self._reference_decoder.compute_jacobian(z, node_idx)
            
            # Handle different Jacobian shapes based on decoder type
            if jacobian.dim() == 2:
                # Shape: [output_dim, latent_dim]
                # Metric tensor: G = J^T @ J
                metric_tensor = torch.mm(jacobian.t(), jacobian)
            elif jacobian.dim() == 3:
                # Shape: [num_nodes, output_dim, latent_dim] - take the specific node
                node_jacobian = jacobian[node_idx]  # [output_dim, latent_dim]
                metric_tensor = torch.mm(node_jacobian.t(), node_jacobian)
            else:
                raise ValueError(f"Unexpected Jacobian shape: {jacobian.shape}")
            
            # Add regularization for numerical stability
            metric_tensor += self.metric_regularization * torch.eye(
                self.latent_dim, device=z.device, dtype=z.dtype
            )
            
            # Cache the result
            if self.cache_jacobians:
                self._jacobian_cache[cache_key] = metric_tensor
            
            return metric_tensor
            
        except Exception as e:
            print(f"Warning: Failed to compute Jacobian metric, falling back to identity: {e}")
            # Fallback to identity metric (Euclidean)
            return torch.eye(self.latent_dim, device=z.device, dtype=z.dtype)
    
    def linear_interpolation_distance_with_jacobian(
        self, 
        u: torch.Tensor, 
        v: torch.Tensor, 
        z_full: torch.Tensor,
        node_idx_u: int,
        node_idx_v: int
    ) -> torch.Tensor:
        """
        Compute distance using linear interpolation with Jacobian-based metric:
        ∫₀¹ √((u-v)ᵀG(x(t))(u-v))dt where x(t) = tu + (1-t)v
        and G(x) is computed from the Jacobian at point x.
        
        Args:
            u, v: Latent vectors [latent_dim]
            z_full: Full latent matrix [num_nodes, latent_dim] (for Jacobian computation)
            node_idx_u, node_idx_v: Node indices for u and v
            
        Returns:
            Distance scalar
        """
        def integrand(t):
            # Interpolated point
            x_t = t * u + (1 - t) * v
            
            # Create temporary z matrix with interpolated point
            # We'll use the midpoint node index for metric computation
            mid_idx = (node_idx_u + node_idx_v) // 2
            z_temp = z_full.clone()
            z_temp[mid_idx] = x_t
            
            try:
                # Compute metric tensor at interpolated point
                G = self.compute_metric_tensor(z_temp, mid_idx)
                
                # Compute distance element: sqrt((u-v)^T @ G @ (u-v))
                diff = u - v
                quadratic_form = torch.sum(diff * (G @ diff))
                return torch.sqrt(torch.clamp(quadratic_form, min=1e-8))
                
            except Exception as e:
                # Fallback to Euclidean distance
                return torch.norm(u - v)
        
        # Numerical integration using trapezoidal rule
        t_vals = torch.linspace(0, 1, self.num_integration_points, device=u.device)
        dt = 1.0 / (self.num_integration_points - 1)
        
        integrand_vals = torch.stack([integrand(t) for t in t_vals])
        integral = torch.trapz(integrand_vals, dx=dt)
        
        return integral
    
    def compute_pairwise_distances(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between all nodes using Jacobian-based metric
        
        Args:
            z: Latent embeddings [num_nodes, latent_dim]
            
        Returns:
            Distance matrix [num_nodes, num_nodes]
        """
        num_nodes = z.size(0)
        distances = torch.zeros(num_nodes, num_nodes, device=z.device, dtype=z.dtype)
        
        # Clear cache at the beginning of each computation
        if not self._cache_valid:
            self._jacobian_cache.clear()
            self._cache_valid = True
        
        # for i in range(num_nodes):
        #     for j in range(i + 1, num_nodes):  # Only compute upper triangle
        #         if self.distance_mode == "linear_interpolation":
        #             dist = self.linear_interpolation_distance_with_jacobian(
        #                 z[i], z[j], z, i, j
        #             )
        #         else:  # direct mode with metric at midpoint
        #             mid_point = (z[i] + z[j]) / 2
        #             z_temp = z.clone()
        #             mid_idx = (i + j) // 2
        #             z_temp[mid_idx] = mid_point
                    
        #             G = self.compute_metric_tensor(z_temp, mid_idx)
        #             diff = z[i] - z[j]
        #             dist = torch.sqrt(torch.sum(diff * (G @ diff)))
                
        #         distances[i, j] = dist
        #         distances[j, i] = dist  # Symmetric


        # Calculate the total number of (i, j) pairs
        # This is equivalent to num_nodes * (num_nodes - 1) / 2 for an upper triangle
        total_pairs = num_nodes * (num_nodes - 1) // 2

        # Initialize the tqdm progress bar
        with tqdm(total=total_pairs, desc="Calculating Distances (Pairs)") as pbar:
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if self.distance_mode == "linear_interpolation":
                        dist = self.linear_interpolation_distance_with_jacobian(
                            z[i], z[j], z, i, j
                        )
                    else:  # direct mode with metric at midpoint
                        mid_point = (z[i] + z[j]) / 2
                        z_temp = z.clone()
                        mid_idx = (i + j) // 2
                        z_temp[mid_idx] = mid_point
                        
                        G = self.compute_metric_tensor(z_temp, mid_idx)
                        diff = z[i] - z[j]
                        dist = torch.sqrt(torch.sum(diff * (G @ diff)))
                    
                    distances[i, j] = dist
                    distances[j, i] = dist  # Symmetric
                    
                    # Manually update the progress bar for each (i, j) pair
                    pbar.update(1)

        
        return distances
    
    def compute_metric_tensor_cached(self, z: torch.Tensor, point: torch.Tensor, node_idx: int) -> torch.Tensor:
        """
        Gradient-safe metric tensor computation with spatial optimization.
        
        The cache is used only to determine if we should skip expensive computations
        in nearby regions, but we always recompute to preserve gradients.
        """
        # Initialize grid bounds on first use
        if self.grid_cache.bounds_min is None:
            self.grid_cache.initialize_bounds(z)
        
        # Check if we have a very recent nearby computation
        cached_template = self.grid_cache.find_cached_metric(point)
        
        # Always compute fresh to preserve gradients
        try:
            # Ensure z_temp requires gradients for Jacobian computation
            z_temp = z.clone()
            if not z_temp.requires_grad:
                z_temp.requires_grad_(True)
            
            # Ensure the specific point also requires gradients
            if point.requires_grad:
                z_temp[node_idx] = point
            else:
                # If point doesn't have gradients, create one that does
                point_with_grad = point.clone().requires_grad_(True)
                z_temp[node_idx] = point_with_grad
            
            jacobian = self._reference_decoder.compute_jacobian(z_temp, node_idx)
            
            if jacobian.dim() == 2:
                metric_tensor = torch.mm(jacobian.t(), jacobian)
            elif jacobian.dim() == 3:
                node_jacobian = jacobian[node_idx]
                metric_tensor = torch.mm(node_jacobian.t(), node_jacobian)
            else:
                raise ValueError(f"Unexpected Jacobian shape: {jacobian.shape}")
            
            # Add regularization
            metric_tensor += self.metric_regularization * torch.eye(
                self.latent_dim, device=z.device, dtype=z.dtype
            )
            
            # Store in cache (detached version for future spatial queries)
            self.grid_cache.store_metric(point, metric_tensor)
            
            return metric_tensor
            
        except Exception as e:
            print(f"Warning: Failed to compute Jacobian metric, falling back to identity: {e}")
            print(f"Debug - z.requires_grad: {z.requires_grad}, point.requires_grad: {point.requires_grad}")
            
            # Create identity tensor that matches gradient requirements
            identity = torch.eye(self.latent_dim, device=z.device, dtype=z.dtype)
            if z.requires_grad or point.requires_grad:
                # Make identity participate in gradient computation
                identity = identity * torch.ones(1, device=z.device, requires_grad=True)
            
            return identity

    def linear_interpolation_distance_optimized(
        self, 
        u: torch.Tensor, 
        v: torch.Tensor, 
        z_full: torch.Tensor,
        node_idx_u: int,
        node_idx_v: int
    ) -> torch.Tensor:
        """
        Optimized distance computation with reduced integration points and gradient safety.
        """
        # Ensure vectors have gradients
        if not u.requires_grad and z_full.requires_grad:
            u = u.clone().requires_grad_(True)
        if not v.requires_grad and z_full.requires_grad:
            v = v.clone().requires_grad_(True)
        
        # Use fewer integration points for speed
        num_points = max(5, self.num_integration_points // 4)  # Reduce integration points
        
        def integrand(t):
            x_t = t * u + (1 - t) * v
            mid_idx = (node_idx_u + node_idx_v) // 2
            
            try:
                G = self.compute_metric_tensor_cached(z_full, x_t, mid_idx)
                diff = u - v
                quadratic_form = torch.sum(diff * (G @ diff))
                return torch.sqrt(torch.clamp(quadratic_form, min=1e-8))
            except Exception as e:
                print(f"Integration point failed: {e}")
                return torch.norm(u - v)
        
        # Reduced integration points
        t_vals = torch.linspace(0, 1, num_points, device=u.device)
        dt = 1.0 / (num_points - 1)
        
        integrand_vals = torch.stack([integrand(t) for t in t_vals])
        integral = torch.trapz(integrand_vals, dx=dt)
        
        return integral

    def compute_pairwise_distances_optimized(self, z: torch.Tensor) -> torch.Tensor:
        """
        Gradient-safe optimized pairwise distance computation.
        
        Uses spatial caching for computational optimization while preserving
        all gradient flows through the actual computations.
        """
        num_nodes = z.size(0)
        distances = torch.zeros(num_nodes, num_nodes, device=z.device, dtype=z.dtype)
        
        # Ensure input tensor has gradients for Jacobian computation
        if not z.requires_grad:
            z = z.requires_grad_(True)
        
        # Initialize grid cache
        self.grid_cache.initialize_bounds(z)
        
        # Pre-seed cache with some metric computations (detached for cache only)
        print("Seeding cache with spatial reference points...")
        with torch.no_grad():
            z_detached = z.detach()
            for i in range(0, min(num_nodes, 50), 5):
                # Store spatial reference points in cache
                try:
                    temp_point = z_detached[i].requires_grad_(True)
                    temp_z = z_detached.clone().requires_grad_(True)
                    temp_z[i] = temp_point
                    
                    # This will fail but populate cache with spatial info
                    self.compute_metric_tensor_cached(temp_z, temp_point, i)
                except:
                    pass  # Expected to fail, just for spatial cache seeding
        
        # Compute distances with progress bar - ALL with gradients preserved
        total_pairs = num_nodes * (num_nodes - 1) // 2
        
        with tqdm(total=total_pairs, desc="Computing Jacobian-based distances") as pbar:
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if self.distance_mode == "linear_interpolation":
                        dist = self.linear_interpolation_distance_optimized(
                            z[i], z[j], z, i, j
                        )
                    else:  # direct mode
                        mid_point = (z[i] + z[j]) / 2  # Preserves gradients
                        mid_idx = (i + j) // 2
                        
                        G = self.compute_metric_tensor_cached(z, mid_point, mid_idx)
                        diff = z[i] - z[j]  # Preserves gradients
                        dist = torch.sqrt(torch.sum(diff * (G @ diff)))
                    
                    distances[i, j] = dist
                    distances[j, i] = dist
                    pbar.update(1)
        
        print(f"Spatial optimization: {len(self.grid_cache.cache)} grid cells cached")
        return distances
        
    def forward(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Transform latent code using Jacobian-based distance computation
        
        Args:
            z: Input latent code [num_nodes, latent_dim]
            
        Returns:
            Transformed representation based on Jacobian-derived distances
        """
        # Invalidate cache for new forward pass
        self._cache_valid = False
        
        # Compute pairwise distances using Jacobian-based metric
        #distance_matrix = self.compute_pairwise_distances(z)
        distance_matrix = self.compute_pairwise_distances_optimized(z)
        
        # Transform distance matrix into features
        num_nodes = z.size(0)
        
        # Option 1: Use distance statistics as features
        distance_features = []
        for i in range(num_nodes):
            node_distances = distance_matrix[i]
            # Create feature vector from distance statistics
            features = torch.stack([
                torch.mean(node_distances),      # Average distance to all nodes
                torch.std(node_distances),       # Standard deviation
                torch.min(node_distances[node_distances > 0]),  # Min non-zero distance
                torch.max(node_distances),       # Max distance
                torch.median(node_distances),    # Median distance
            ])
            distance_features.append(features)
        
        distance_features = torch.stack(distance_features)
        
        # Pad or truncate to match expected dimension
        if distance_features.size(1) < self.latent_dim:
            padding = torch.zeros(num_nodes, self.latent_dim - distance_features.size(1), 
                                device=z.device, dtype=z.dtype)
            distance_features = torch.cat([distance_features, padding], dim=1)
        elif distance_features.size(1) > self.latent_dim:
            distance_features = distance_features[:, :self.latent_dim]
        
        # Apply optional transformation
        transformed = self.transform(distance_features)
        
        return transformed

# Usage example:
"""
# Example of how to integrate this into GraphVAE:

# Define your custom metric function
def my_metric_function(x):
    # Your custom metric tensor computation
    # Should return a matrix g such that the distance is sqrt((u-v)^T @ g @ (u-v))
    return np.eye(len(x))  # Identity for Euclidean (example)

# Create your encoder (assuming you have this)
encoder = YourEncoder(input_dim=node_feature_dim, latent_dim=latent_dim)

# Create regular decoders
node_attr_decoder = NodeAttributeDecoder(latent_dim, output_dim=node_feature_dim)

# Create the distance-based adjacency decoder
distance_adj_decoder = create_distance_based_adjacency_decoder(
    latent_dim=latent_dim,
    metric_function=my_metric_function
)

# Create GraphVAE with both regular and preprocessed decoders
model = GraphVAE(
    encoder=encoder,
    decoders=[node_attr_decoder, distance_adj_decoder],
    kl_scheduler=KLAnnealingScheduler()
)

# Training loop example
for epoch in range(num_epochs):
    for batch in dataloader:
        x, edge_index = batch.x, batch.edge_index
        
        # Forward pass
        outputs = model(x, edge_index)
        
        # Prepare targets
        targets = {
            "node_attr_decoder": {"node_features": x},
            "distance_based_adjacency": {
                "edge_labels": batch.edge_attr,  # or however you define edge weights
                "edge_index": edge_index
            }
        }
        
        # Compute loss
        loss_dict = model.compute_loss(outputs, targets)
        total_loss = loss_dict["total_loss"]
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Step KL scheduler
        model.kl_scheduler.step()
"""
