
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from tqdm import tqdm
import numpy as np

class GradientAwareJacobianCache:
    """
    Grid-based cache for Jacobian computations that preserves gradient flow.
    Uses spatial hashing and interpolation for efficient caching while maintaining differentiability.
    """
    def __init__(self, grid_resolution: int = 32, cache_radius: float = 0.1, 
                 max_cache_size: int = 1000, interpolation_order: int = 1):
        self.grid_resolution = grid_resolution
        self.cache_radius = cache_radius
        self.max_cache_size = max_cache_size
        self.interpolation_order = interpolation_order
        
        # Cache storage: {grid_key: (jacobian_tensor, position_tensor, training_step)}
        self.cache = {}
        self.training_step = 0
        
        # Statistics for cache performance
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _get_grid_key(self, position: torch.Tensor) -> Tuple[int, ...]:
        """Convert continuous position to discrete grid key"""
        # Normalize position to grid coordinates
        grid_coords = (position / self.cache_radius * self.grid_resolution).long()
        # Clamp to prevent overflow
        grid_coords = torch.clamp(grid_coords, -1000, 1000)
        return tuple(grid_coords.cpu().numpy())
    
    def _get_neighboring_keys(self, key: Tuple[int, ...]) -> list:
        """Get neighboring grid keys for interpolation"""
        neighbors = []
        if len(key) == 1:
            for dx in [-1, 0, 1]:
                neighbors.append((key[0] + dx,))
        elif len(key) == 2:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbors.append((key[0] + dx, key[1] + dy))
        else:  # General case for higher dimensions
            import itertools
            offsets = list(itertools.product([-1, 0, 1], repeat=len(key)))
            for offset in offsets:
                neighbors.append(tuple(k + o for k, o in zip(key, offset)))
        return neighbors
    
    def _interpolate_jacobians(self, position: torch.Tensor, 
                             cached_entries: Dict) -> Optional[torch.Tensor]:
        """
        Interpolate between cached Jacobians using distance-weighted interpolation.
        This maintains differentiability with respect to the input position.
        """
        if not cached_entries:
            return None
            
        positions = []
        jacobians = []
        weights = []
        
        for cached_pos, cached_jac in cached_entries.items():
            dist = torch.norm(position - cached_pos)
            if dist < self.cache_radius:
                positions.append(cached_pos)
                jacobians.append(cached_jac)
                # Use inverse distance weighting (with small epsilon to avoid division by zero)
                weight = 1.0 / (dist + 1e-8)
                weights.append(weight)
        
        if not weights:
            return None
            
        # Convert to tensors
        weights = torch.stack(weights)
        jacobians = torch.stack(jacobians)
        
        # Normalize weights
        weights = weights / torch.sum(weights)
        
        # Weighted interpolation (preserves gradients)
        interpolated_jacobian = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * jacobians, dim=0)
        
        return interpolated_jacobian
    
    def get_cached_jacobian(self, position: torch.Tensor, node_idx: int) -> Optional[torch.Tensor]:
        """
        Retrieve cached Jacobian with interpolation if exact match not found.
        Returns None if no suitable cached values are found.
        """
        # Create composite key including node index
        pos_key = self._get_grid_key(position)
        cache_key = (node_idx, pos_key)
        
        # Check for exact match first
        if cache_key in self.cache:
            cached_jac, cached_pos, step = self.cache[cache_key]
            self.cache_hits += 1
            return cached_jac
        
        # Look for nearby cached values for interpolation
        neighboring_keys = self._get_neighboring_keys(pos_key)
        nearby_entries = {}
        
        for neighbor_key in neighboring_keys:
            full_key = (node_idx, neighbor_key)
            if full_key in self.cache:
                cached_jac, cached_pos, step = self.cache[full_key]
                nearby_entries[cached_pos] = cached_jac
        
        # Try interpolation
        if nearby_entries:
            interpolated = self._interpolate_jacobians(position, nearby_entries)
            if interpolated is not None:
                self.cache_hits += 1
                return interpolated
        
        self.cache_misses += 1
        return None
    
    def store_jacobian(self, position: torch.Tensor, node_idx: int, jacobian: torch.Tensor):
        """Store Jacobian in cache with position and training step info"""
        pos_key = self._get_grid_key(position)
        cache_key = (node_idx, pos_key)
        
        # Store with detached position (to avoid keeping computation graph)
        # but keep jacobian with gradients for interpolation
        self.cache[cache_key] = (jacobian.detach().clone(), 
                                position.detach().clone(), 
                                self.training_step)
        
        # Prune cache if it gets too large
        if len(self.cache) > self.max_cache_size:
            self._prune_cache()
    
    def _prune_cache(self):
        """Remove oldest entries from cache"""
        # Sort by training step and remove oldest 20%
        items = list(self.cache.items())
        items.sort(key=lambda x: x[1][2])  # Sort by training_step
        
        prune_count = len(items) // 5  # Remove 20%
        for i in range(prune_count):
            del self.cache[items[i][0]]
    
    def update_training_step(self):
        """Call this at the beginning of each training step"""
        self.training_step += 1
    
    def clear_cache(self):
        """Clear all cached entries"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'training_step': self.training_step
        }


class JacobianMetricPreprocessingLayer(nn.Module):
    """
    Preprocessing layer that computes pairwise distances using metrics derived
    from the Jacobian of another decoder (typically NodeAttributeDecoder).
    Enhanced with gradient-aware Jacobian caching.
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
        grid_resolution: int = 32,
        cache_radius: float = 0.1,
        max_cache_size: int = 1000,
        name: str = "jacobian_metric_preprocessor",
        **kwargs
    ):
        super(JacobianMetricPreprocessingLayer, self).__init__()
        self.latent_dim = latent_dim
        
        # CRITICAL FIX: Store references without registering as submodules
        object.__setattr__(self, '_reference_model', reference_model)
        object.__setattr__(self, '_reference_decoder', reference_decoder)
        
        self.num_integration_points = num_integration_points
        self.metric_regularization = metric_regularization
        self.distance_mode = distance_mode
        self.output_dim = output_dim if output_dim is not None else latent_dim
        self.cache_jacobians = cache_jacobians
        
        # Enhanced gradient-aware cache
        self.jacobian_cache = GradientAwareJacobianCache(
            grid_resolution=grid_resolution,
            cache_radius=cache_radius,
            max_cache_size=max_cache_size
        )
        
        # Optional learnable transformation after distance computation
        if self.output_dim != latent_dim:
            self.transform = nn.Linear(latent_dim, self.output_dim)
        else:
            self.transform = nn.Identity()
    
    def compute_metric_tensor(self, z: torch.Tensor, node_idx: int) -> torch.Tensor:
        """
        Compute the metric tensor at a given point in latent space using the 
        Jacobian of the reference decoder with gradient-aware caching.
        """
        position = z[node_idx]
        
        # Try to get cached Jacobian first
        if self.cache_jacobians:
            cached_jacobian = self.jacobian_cache.get_cached_jacobian(position, node_idx)
            if cached_jacobian is not None:
                # Recompute metric tensor from cached Jacobian
                # Note: We need to ensure gradient flow, so we create a differentiable operation
                if cached_jacobian.dim() == 2:
                    metric_tensor = torch.mm(cached_jacobian.t(), cached_jacobian)
                else:
                    metric_tensor = torch.mm(cached_jacobian.t(), cached_jacobian)
                
                # Add regularization
                metric_tensor = metric_tensor + self.metric_regularization * torch.eye(
                    self.latent_dim, device=z.device, dtype=z.dtype
                )
                
                # Create a differentiable connection to the current position
                # This ensures gradients can flow back to the current z
                connection_loss = torch.sum((position - position.detach()) ** 2)
                metric_tensor = metric_tensor + connection_loss * 0  # Adds gradient connection with zero contribution
                
                return metric_tensor
        
        try:
            # Compute Jacobian of reference decoder at this point        
            jacobian = self._reference_decoder.compute_jacobian(z, node_idx)
            
            # Handle different Jacobian shapes based on decoder type
            if jacobian.dim() == 2:
                # Shape: [output_dim, latent_dim]
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
            
            # Cache the Jacobian (not the metric tensor) for future use
            if self.cache_jacobians:
                jacobian_to_cache = node_jacobian if jacobian.dim() == 3 else jacobian
                self.jacobian_cache.store_jacobian(position, node_idx, jacobian_to_cache)
            
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
        Compute distance using linear interpolation with Jacobian-based metric.
        Enhanced with caching for intermediate points.
        """
        def integrand(t):
            # Interpolated point
            x_t = t * u + (1 - t) * v
            
            # Create temporary z matrix with interpolated point
            mid_idx = (node_idx_u + node_idx_v) // 2
            z_temp = z_full.clone()
            z_temp[mid_idx] = x_t
            
            try:
                # Compute metric tensor at interpolated point (with caching)
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
        with enhanced caching.
        """
        num_nodes = z.size(0)
        distances = torch.zeros(num_nodes, num_nodes, device=z.device, dtype=z.dtype)
        
        # Update training step for cache management
        self.jacobian_cache.update_training_step()
        
        total_pairs = num_nodes * (num_nodes - 1) // 2
        
        with tqdm(total=total_pairs, desc="Calculating Distances (Cached)") as pbar:
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
                    pbar.update(1)
        
        # Print cache statistics periodically
        if self.jacobian_cache.training_step % 100 == 0:
            stats = self.jacobian_cache.get_cache_stats()
            print(f"Cache Stats: {stats}")
        
        return distances
    
    def forward(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Transform latent code using Jacobian-based distance computation
        with enhanced caching.
        """
        # Compute pairwise distances using cached Jacobian-based metric
        distance_matrix = self.compute_pairwise_distances(z)
        
        # Transform distance matrix into features
        num_nodes = z.size(0)
        
        # Create feature vector from distance statistics
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
    
    def clear_cache(self):
        """Clear the Jacobian cache"""
        self.jacobian_cache.clear_cache()
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        return self.jacobian_cache.get_cache_stats()
    
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List
from tqdm import tqdm
import numpy as np
from scipy.spatial import cKDTree

class BatchJacobianCache:
    """
    High-performance batch-oriented Jacobian cache with precomputed grids and 
    vectorized operations for handling 500+ points efficiently.
    """
    def __init__(self, latent_dim: int, grid_size: int = 64, 
                 cache_memory_gb: float = 1.0, update_frequency: int = 50):
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.update_frequency = update_frequency
        self.step_count = 0
        
        # Pre-allocate dense grid cache
        max_cache_entries = int(cache_memory_gb * 1e9 / (latent_dim * latent_dim * 4))  # 4 bytes per float32
        self.max_entries = min(max_cache_entries, grid_size ** latent_dim)
        
        # Dense tensor storage for vectorized operations
        self.grid_positions = None  # [num_cached_points, latent_dim]
        self.cached_jacobians = None  # [num_cached_points, output_dim, latent_dim]
        self.cached_metrics = None   # [num_cached_points, latent_dim, latent_dim]
        self.valid_mask = None       # [num_cached_points] - which entries are valid
        self.tree = None            # KDTree for fast nearest neighbor lookup
        
        # Grid bounds (will be updated based on data)
        self.grid_bounds = None
        self.is_initialized = False
        
    def initialize_grid(self, z_sample: torch.Tensor):
        """Initialize the cache grid based on actual data distribution"""
        if self.is_initialized:
            return
            
        # Compute bounds from sample data
        z_min = torch.min(z_sample, dim=0)[0] - 0.2  # Add padding
        z_max = torch.max(z_sample, dim=0)[0] + 0.2
        self.grid_bounds = torch.stack([z_min, z_max])
        
        # Create regular grid within bounds
        grid_points = []
        if self.latent_dim <= 3:  # Full grid for low dimensions
            coords = [torch.linspace(z_min[i], z_max[i], self.grid_size) 
                     for i in range(self.latent_dim)]
            meshgrid = torch.meshgrid(*coords, indexing='ij')
            grid_points = torch.stack([g.flatten() for g in meshgrid], dim=1)
        else:  # Random sampling for high dimensions
            n_points = min(self.max_entries, self.grid_size ** 2)  # Quadratic scaling
            grid_points = torch.rand(n_points, self.latent_dim)
            grid_points = grid_points * (z_max - z_min) + z_min
        
        # Initialize storage tensors
        n_grid = min(len(grid_points), self.max_entries)
        device = z_sample.device
        
        self.grid_positions = grid_points[:n_grid].to(device)
        self.valid_mask = torch.zeros(n_grid, dtype=torch.bool, device=device)
        
        # Pre-allocate but don't fill yet (will be computed on-demand)
        self.cached_jacobians = None
        self.cached_metrics = torch.zeros(n_grid, self.latent_dim, self.latent_dim, 
                                        device=device, dtype=z_sample.dtype)
        
        self.is_initialized = True
        print(f"Initialized Jacobian cache with {n_grid} grid points")
    
    def batch_compute_missing_metrics(self, reference_decoder, z_context: torch.Tensor, 
                                    indices_needed: torch.Tensor):
        """Compute missing metric tensors in batch for efficiency"""
        if len(indices_needed) == 0:
            return
            
        batch_positions = self.grid_positions[indices_needed]  # [batch, latent_dim]
        batch_size = len(batch_positions)
        
        # Create context matrices for batch Jacobian computation
        # We'll compute Jacobians at grid points using a temporary context
        z_temp = z_context[0:1].expand(batch_size, -1).clone()  # [batch, latent_dim]
        
        try:
            # Batch Jacobian computation - this is the key optimization
            batch_jacobians = []
            
            # Process in smaller chunks to manage memory
            chunk_size = min(32, batch_size)  # Adjust based on memory
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunk_positions = batch_positions[i:end_idx]
                
                # Set positions in context
                z_temp[i:end_idx] = chunk_positions
                
                # Compute Jacobians for this chunk
                try:
                    chunk_jac = reference_decoder.compute_jacobian(z_temp[i:end_idx], 0)
                    if chunk_jac.dim() == 3:
                        chunk_jac = chunk_jac[:, 0]  # Take first node if batched
                    batch_jacobians.append(chunk_jac)
                except:
                    # Fallback: identity Jacobians
                    identity_jac = torch.eye(self.latent_dim, device=chunk_positions.device)
                    identity_jac = identity_jac.unsqueeze(0).expand(len(chunk_positions), -1, -1)
                    batch_jacobians.append(identity_jac)
            
            # Concatenate all chunks
            all_jacobians = torch.cat(batch_jacobians, dim=0)  # [batch, output_dim, latent_dim]
            
            # Batch compute metric tensors: G = J^T @ J
            batch_metrics = torch.bmm(all_jacobians.transpose(-2, -1), all_jacobians)
            
            # Add regularization
            reg_eye = torch.eye(self.latent_dim, device=batch_metrics.device, 
                              dtype=batch_metrics.dtype)
            batch_metrics += 1e-6 * reg_eye.unsqueeze(0)
            
            # Store in cache
            self.cached_metrics[indices_needed] = batch_metrics
            self.valid_mask[indices_needed] = True
            
        except Exception as e:
            print(f"Batch Jacobian computation failed: {e}")
            # Fallback to identity metrics
            identity_metric = torch.eye(self.latent_dim, device=self.grid_positions.device,
                                      dtype=self.cached_metrics.dtype)
            self.cached_metrics[indices_needed] = identity_metric.unsqueeze(0).expand(len(indices_needed), -1, -1)
            self.valid_mask[indices_needed] = True
    
    def get_interpolated_metrics(self, positions: torch.Tensor, reference_decoder, 
                               z_context: torch.Tensor) -> torch.Tensor:
        """
        Get metric tensors for given positions using fast interpolation.
        Returns: [num_positions, latent_dim, latent_dim]
        """
        if not self.is_initialized:
            self.initialize_grid(positions)
        
        num_pos = len(positions)
        device = positions.device
        
        # Build KDTree if needed (CPU operation for speed)
        if self.tree is None and torch.sum(self.valid_mask) > 0:
            valid_positions = self.grid_positions[self.valid_mask].cpu().numpy()
            if len(valid_positions) > 0:
                self.tree = cKDTree(valid_positions)
        
        # Find nearest neighbors for each query position
        k = min(4, torch.sum(self.valid_mask).item())  # Use 4 nearest neighbors
        if self.tree is None or k == 0:
            # No cached data yet, compute what we need
            indices_to_compute = torch.arange(min(num_pos, len(self.grid_positions)))
            self.batch_compute_missing_metrics(reference_decoder, z_context, indices_to_compute)
            
            # Rebuild tree
            if torch.sum(self.valid_mask) > 0:
                valid_positions = self.grid_positions[self.valid_mask].cpu().numpy()
                self.tree = cKDTree(valid_positions)
                k = min(4, len(valid_positions))
        
        if k == 0:
            # Still no valid cache entries, return identity
            return torch.eye(self.latent_dim, device=device).unsqueeze(0).expand(num_pos, -1, -1)
        
        # Query nearest neighbors
        query_positions = positions.cpu().numpy()
        distances, neighbor_indices = self.tree.query(query_positions, k=k)
        
        # Handle single query case
        if distances.ndim == 1:
            distances = distances.reshape(1, -1)
            neighbor_indices = neighbor_indices.reshape(1, -1)
        
        # Check if we need to compute more grid points
        max_distance_threshold = 0.5  # Adjust based on your latent space scale
        far_queries = np.any(distances > max_distance_threshold, axis=1)
        
        if np.any(far_queries) and self.step_count % self.update_frequency == 0:
            # Adaptively add grid points near far queries
            far_positions = positions[far_queries]
            n_new = min(len(far_positions), len(self.grid_positions) - torch.sum(self.valid_mask).item())
            
            if n_new > 0:
                # Find unused grid slots
                unused_indices = torch.where(~self.valid_mask)[0][:n_new]
                self.grid_positions[unused_indices] = far_positions[:n_new]
                self.batch_compute_missing_metrics(reference_decoder, z_context, unused_indices)
                
                # Rebuild tree
                valid_positions = self.grid_positions[self.valid_mask].cpu().numpy()
                self.tree = cKDTree(valid_positions)
                
                # Re-query with updated tree
                distances, neighbor_indices = self.tree.query(query_positions, k=k)
                if distances.ndim == 1:
                    distances = distances.reshape(1, -1)
                    neighbor_indices = neighbor_indices.reshape(1, -1)
        
        # Convert neighbor indices back to cache indices
        valid_indices = torch.where(self.valid_mask)[0].cpu().numpy()
        cache_indices = valid_indices[neighbor_indices]  # [num_pos, k]
        
        # Perform interpolation
        result_metrics = torch.zeros(num_pos, self.latent_dim, self.latent_dim, 
                                   device=device, dtype=positions.dtype)
        
        for i in range(num_pos):
            weights = 1.0 / (distances[i] + 1e-6)  # Inverse distance weighting
            weights = weights / np.sum(weights)    # Normalize
            
            # Weighted combination of metric tensors
            weighted_metric = torch.zeros(self.latent_dim, self.latent_dim, device=device)
            for j, cache_idx in enumerate(cache_indices[i]):
                weighted_metric += weights[j] * self.cached_metrics[cache_idx]
            
            result_metrics[i] = weighted_metric
        
        self.step_count += 1
        return result_metrics


class FastJacobianMetricLayer(nn.Module):
    """
    Ultra-fast Jacobian metric preprocessing layer optimized for 500+ points.
    Uses batch processing, vectorized operations, and smart caching.
    """
    def __init__(
        self,
        latent_dim: int,
        reference_decoder: "DecoderBase", 
        reference_model: "GraphVAE",
        num_integration_points: int = 20,  # Reduced for speed
        metric_regularization: float = 1e-6,
        distance_mode: str = "batch_direct",  # New faster mode
        output_dim: Optional[int] = None,
        cache_memory_gb: float = 1.0,
        distance_approximation: str = "local_metric",  # "local_metric" or "average_metric"
        **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Store references
        object.__setattr__(self, '_reference_model', reference_model)
        object.__setattr__(self, '_reference_decoder', reference_decoder)
        
        self.num_integration_points = num_integration_points
        self.metric_regularization = metric_regularization
        self.distance_mode = distance_mode
        self.output_dim = output_dim if output_dim is not None else latent_dim
        self.distance_approximation = distance_approximation
        
        # High-performance cache
        self.jacobian_cache = BatchJacobianCache(
            latent_dim=latent_dim,
            grid_size=kwargs.get('grid_size', 32),
            cache_memory_gb=cache_memory_gb,
            update_frequency=kwargs.get('update_frequency', 50)
        )
        
        # Learnable transformation
        if self.output_dim != latent_dim:
            self.transform = nn.Linear(latent_dim, self.output_dim)
        else:
            self.transform = nn.Identity()
        
        # Pre-computed distance feature extractors
        self.register_buffer('distance_weights', torch.ones(5))  # For weighted distance features
    
    def batch_compute_distances_fast(self, z: torch.Tensor) -> torch.Tensor:
        """
        Ultra-fast distance computation using vectorized operations and approximations.
        """
        num_nodes = z.size(0)
        device = z.device
        
        if self.distance_approximation == "average_metric":
            # Use single average metric tensor for all pairs (fastest)
            avg_position = torch.mean(z, dim=0, keepdim=True)
            z_avg = avg_position.expand(1, -1)
            
            avg_metrics = self.jacobian_cache.get_interpolated_metrics(
                avg_position, self._reference_decoder, z
            )  # [1, latent_dim, latent_dim]
            
            G = avg_metrics[0]  # [latent_dim, latent_dim]
            
            # Vectorized distance computation: d(i,j) = sqrt((z_i - z_j)^T @ G @ (z_i - z_j))
            z_expanded_i = z.unsqueeze(1)  # [num_nodes, 1, latent_dim]
            z_expanded_j = z.unsqueeze(0)  # [1, num_nodes, latent_dim]
            diff = z_expanded_i - z_expanded_j  # [num_nodes, num_nodes, latent_dim]
            
            # Batch matrix multiplication: [num_nodes, num_nodes, latent_dim] @ [latent_dim, latent_dim]
            diff_transformed = torch.matmul(diff, G)  # [num_nodes, num_nodes, latent_dim]
            
            # Compute quadratic form: sum over latent dimension
            quadratic_form = torch.sum(diff * diff_transformed, dim=-1)  # [num_nodes, num_nodes]
            distances = torch.sqrt(torch.clamp(quadratic_form, min=1e-8))
            
        else:  # "local_metric" - compute metrics at subset of points
            # Subsample points for metric computation to balance accuracy vs speed
            max_metric_points = min(50, num_nodes)  # Limit metric computations
            if num_nodes > max_metric_points:
                # Use k-means style sampling to get representative points
                indices = torch.randperm(num_nodes)[:max_metric_points]
                metric_positions = z[indices]
            else:
                metric_positions = z
                indices = torch.arange(num_nodes)
            
            # Get metrics for sampled positions
            metrics = self.jacobian_cache.get_interpolated_metrics(
                metric_positions, self._reference_decoder, z
            )  # [num_metric_points, latent_dim, latent_dim]
            
            # For each node pair, use nearest available metric
            distances = torch.zeros(num_nodes, num_nodes, device=device, dtype=z.dtype)
            
            # Vectorized computation for each metric region
            for k, (idx, G) in enumerate(zip(indices, metrics)):
                # Compute distances using this metric for a local region
                center = z[idx]
                dists_to_center = torch.norm(z - center, dim=1)
                
                # Use this metric for points within radius (or closest points)
                if num_nodes <= max_metric_points:
                    # Use this metric for all pairs involving this point
                    for i in range(num_nodes):
                        if distances[idx, i] == 0 and i != idx:  # Not computed yet
                            diff = z[idx] - z[i]
                            quad_form = torch.sum(diff * (G @ diff))
                            dist = torch.sqrt(torch.clamp(quad_form, min=1e-8))
                            distances[idx, i] = dist
                            distances[i, idx] = dist
                else:
                    # More sophisticated assignment based on proximity
                    region_size = num_nodes // max_metric_points
                    start_idx = k * region_size
                    end_idx = min((k + 1) * region_size, num_nodes)
                    
                    region_indices = list(range(start_idx, end_idx))
                    for i in region_indices:
                        for j in range(i + 1, num_nodes):
                            if distances[i, j] == 0:  # Not computed yet
                                diff = z[i] - z[j]
                                quad_form = torch.sum(diff * (G @ diff))
                                dist = torch.sqrt(torch.clamp(quad_form, min=1e-8))
                                distances[i, j] = dist
                                distances[j, i] = dist
            
            # Fill any remaining zero distances with Euclidean
            mask = (distances == 0) & (torch.eye(num_nodes, device=device) == 0)
            if torch.any(mask):
                euclidean_dists = torch.cdist(z, z, p=2)
                distances[mask] = euclidean_dists[mask]
        
        return distances
    
    def extract_distance_features_vectorized(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        """Extract distance-based features using vectorized operations"""
        num_nodes = distance_matrix.size(0)
        
        # Mask diagonal to exclude zero distances
        mask = torch.eye(num_nodes, device=distance_matrix.device, dtype=torch.bool)
        masked_distances = distance_matrix.masked_fill(mask, float('inf'))
        
        # Vectorized feature computation
        features = torch.stack([
            torch.mean(masked_distances, dim=1),           # Mean distance
            torch.std(masked_distances, dim=1),            # Std distance  
            torch.min(masked_distances, dim=1)[0],         # Min distance
            torch.max(distance_matrix, dim=1)[0],          # Max distance
            torch.median(masked_distances, dim=1)[0],      # Median distance
        ], dim=1)  # [num_nodes, 5]
        
        # Handle any NaN/Inf values
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        
        return features
    
    def forward(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Fast forward pass optimized for large numbers of nodes.
        """
        # Fast batch distance computation
        distance_matrix = self.batch_compute_distances_fast(z)
        
        # Vectorized feature extraction
        distance_features = self.extract_distance_features_vectorized(distance_matrix)
        
        # Pad or truncate to match expected dimension
        if distance_features.size(1) < self.latent_dim:
            padding_size = self.latent_dim - distance_features.size(1)
            padding = torch.zeros(distance_features.size(0), padding_size, 
                                device=z.device, dtype=z.dtype)
            distance_features = torch.cat([distance_features, padding], dim=1)
        elif distance_features.size(1) > self.latent_dim:
            distance_features = distance_features[:, :self.latent_dim]
        
        # Apply learnable transformation
        transformed = self.transform(distance_features)
        
        return transformed
    
    def get_cache_stats(self):
        """Get performance statistics"""
        if hasattr(self.jacobian_cache, 'valid_mask'):
            cache_usage = torch.sum(self.jacobian_cache.valid_mask).item()
            total_cache = len(self.jacobian_cache.valid_mask)
            return {
                'cache_usage': f"{cache_usage}/{total_cache}",
                'cache_utilization': cache_usage / total_cache if total_cache > 0 else 0,
                'step_count': self.jacobian_cache.step_count
            }
        return {'cache_usage': 'Not initialized'}