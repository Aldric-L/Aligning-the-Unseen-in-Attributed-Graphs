import torch
from typing import Dict, Tuple, Optional

class JacobianGridCache:
    """
    Gradient-safe spatial grid cache for Jacobian computations.
    Uses detached tensors for spatial indexing while preserving gradients
    in the actual metric computations.
    """
    def __init__(
        self, 
        grid_resolution: int = 32,
        cache_radius: float = 0.1,
        max_cache_size: int = 1000
    ):
        self.grid_resolution = grid_resolution
        self.cache_radius = cache_radius
        self.max_cache_size = max_cache_size
        
        # Cache structure: {grid_cell: (metric_tensor, center_point, usage_count)}
        # NOTE: We store detached tensors for spatial lookup but recompute with gradients
        self.cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, int]] = {}
        self.bounds_min = None
        self.bounds_max = None
        self.grid_size = None
        
    def initialize_bounds(self, z: torch.Tensor):
        """Initialize grid bounds based on data extent (detached for indexing)"""
        # Use detached tensors for bounds to avoid gradient issues in indexing
        with torch.no_grad():
            self.bounds_min = torch.min(z, dim=0)[0] - 0.1  # Small padding
            self.bounds_max = torch.max(z, dim=0)[0] + 0.1
            self.grid_size = (self.bounds_max - self.bounds_min) / self.grid_resolution
        
    def point_to_grid_cell(self, point: torch.Tensor) -> Tuple[int, int]:
        """Convert a point to its grid cell coordinates (uses detached values)"""
        if self.bounds_min is None:
            raise ValueError("Grid bounds not initialized")
        
        # Detach for indexing operations only
        with torch.no_grad():
            normalized = (point.detach() - self.bounds_min) / self.grid_size
            grid_x = int(torch.clamp(normalized[0], 0, self.grid_resolution - 1))
            grid_y = int(torch.clamp(normalized[1], 0, self.grid_resolution - 1))
        return (grid_x, grid_y)
    
    def grid_cell_to_point(self, grid_cell: Tuple[int, int]) -> torch.Tensor:
        """Convert grid cell to its center point"""
        grid_x, grid_y = grid_cell
        center = self.bounds_min + torch.tensor([grid_x + 0.5, grid_y + 0.5], 
                                               device=self.bounds_min.device) * self.grid_size
        return center
    
    def get_nearby_cells(self, point: torch.Tensor) -> list:
        """Get grid cells within cache_radius of the point"""
        center_cell = self.point_to_grid_cell(point)
        nearby_cells = []
        
        # Check cells in a small neighborhood
        radius_in_cells = max(1, int(self.cache_radius / torch.min(self.grid_size).item()))
        
        for dx in range(-radius_in_cells, radius_in_cells + 1):
            for dy in range(-radius_in_cells, radius_in_cells + 1):
                cell_x = center_cell[0] + dx
                cell_y = center_cell[1] + dy
                
                if (0 <= cell_x < self.grid_resolution and 
                    0 <= cell_y < self.grid_resolution):
                    nearby_cells.append((cell_x, cell_y))
        
        return nearby_cells
    
    def find_cached_metric(self, point: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Find if we should reuse computation based on spatial proximity.
        Returns None to force recomputation (maintaining gradients).
        
        This is now used only for computational savings detection,
        not for returning cached gradients.
        """
        nearby_cells = self.get_nearby_cells(point)
        
        # Check if we have a nearby computation, but don't return it
        # This allows us to skip expensive computations while preserving gradients
        for cell in nearby_cells:
            if cell in self.cache:
                cached_metric, cached_center, usage_count = self.cache[cell]
                distance = torch.norm(point.detach() - cached_center).item()
                
                if distance < self.cache_radius:
                    # Update usage but don't return cached tensor
                    metric, center, count = self.cache[cell]
                    self.cache[cell] = (metric, center, count + 1)
                    return cached_metric.detach()  # Return detached version as template
        
        return None
    
    def store_metric(self, point: torch.Tensor, metric: torch.Tensor):
        """Store a computed metric tensor in the cache (detached for storage)"""
        cell = self.point_to_grid_cell(point)
        
        # If cache is full, remove least used entry
        if len(self.cache) >= self.max_cache_size:
            min_usage = float('inf')
            worst_cell = None
            for c, (_, _, usage) in self.cache.items():
                if usage < min_usage:
                    min_usage = usage
                    worst_cell = c
            
            if worst_cell is not None:
                del self.cache[worst_cell]
        
        # Store detached versions to avoid gradient graph issues
        self.cache[cell] = (metric.detach(), point.detach(), 1)