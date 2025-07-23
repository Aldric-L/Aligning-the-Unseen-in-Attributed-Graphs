import torch
from typing import List, Tuple, Union, Callable
from itertools import product
try:
    from torchdiffeq import odeint
except ImportError:
    raise ImportError("Please install torchdiffeq: pip install torchdiffeq")
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


class BoundedManifold:
    """
    Manifold class that wraps the metric tensor and allows for caching.
    This version uses PyTorch tensors to be compatible with autograd pipelines.
    """
    def __init__(self, metric_tensor_func: callable, bounds: torch.Tensor, 
                 cache: bool = False, grid_points_per_dim: Union[int, List[int]] = 10, device: str = 'cpu'):
        """
        Initializes the BoundedManifold.

        Args:
            metric_tensor_func (callable): A callable that takes a torch.Tensor (point)
                                           and returns the metric tensor at that point.
                                           Expected signature: metric_tensor_func(point: torch.Tensor) -> torch.Tensor
            bounds (torch.Tensor): A 2D torch.Tensor defining the bounds for each dimension.
                                   Shape should be (n_dimensions, 2), where each row is [min_val, max_val].
            cache (bool): If True, metric tensor values will be cached and interpolation will be used.
            grid_points_per_dim (Union[int, List[int]]): The number of grid points per dimension.
            device (str): The device to run the computations on ('cpu' or 'cuda').
        """
        if not isinstance(bounds, torch.Tensor) or bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("Bounds must be a 2D torch.Tensor of shape (n_dimensions, 2).")

        self.device = device
        self._metric_tensor_func = metric_tensor_func
        self._bounds = bounds.to(self.device)
        self._n_dimensions = bounds.shape[0]
        self.cache_enabled = cache

        if isinstance(grid_points_per_dim, int):
            if grid_points_per_dim < 2 and cache:
                raise ValueError("grid_points_per_dim must be at least 2 for caching with interpolation.")
            self._grid_points_per_dim = [grid_points_per_dim] * self._n_dimensions
        elif isinstance(grid_points_per_dim, list):
            if len(grid_points_per_dim) != self._n_dimensions:
                raise ValueError(f"Length of grid_points_per_dim list ({len(grid_points_per_dim)}) must match manifold dimension ({self._n_dimensions}).")
            if any(n_points < 2 for n_points in grid_points_per_dim) and cache:
                raise ValueError("Each dimension in grid_points_per_dim must be at least 2 for caching with interpolation.")
            self._grid_points_per_dim = grid_points_per_dim
        else:
            raise TypeError("grid_points_per_dim must be an int or a list of ints.")

        # Generate the coordinates for each dimension of the grid
        self._grid_axes = [
            torch.linspace(self._bounds[i, 0], self._bounds[i, 1], self._grid_points_per_dim[i], device=self.device)
            for i in range(self._n_dimensions)
        ]

        # Shape of the cache: (grid_dim1, grid_dim2, ..., metric_tensor_rows, metric_tensor_cols)
        cache_shape = tuple(self._grid_points_per_dim) + (self._n_dimensions, self._n_dimensions)
        self._grid_cache = torch.full(cache_shape, torch.nan, device=self.device)
        self._is_filled = False

    @staticmethod
    def hypercube_bounds(
        vectors: torch.Tensor,
        margin: float = 0.0,
        relative: bool = True
    ) -> torch.Tensor:
        """
        Compute axis-aligned hypercube bounds around a set of N-dimensional vectors with an optional margin.

        Args:
            vectors (torch.Tensor): A tensor of shape (M, N) where M is the number of vectors and
                                    N is the dimensionality.
            margin (float): If `relative=True`, this is a fraction of the range on each axis
                            to pad the bounds. If `relative=False`, this is an absolute padding value.
            relative (bool): Whether the margin is relative (fractional) or absolute.

        Returns:
            torch.Tensor: A tensor of shape (N, 2), where each row corresponds to one axis,
                        the first column is the lower bound, and the second column is the upper bound.
        """
        if vectors.dim() != 2:
            raise ValueError(f"Expected a 2D tensor of shape (M, N), but got shape {vectors.shape}")

        # Compute minima and maxima along each dimension
        mins, _ = torch.min(vectors, dim=0)
        maxs, _ = torch.max(vectors, dim=0)
        ranges = maxs - mins

        # Apply margin
        if relative:
            pad = ranges * margin
        else:
            pad = torch.full_like(ranges, margin)

        lower = mins - pad
        upper = maxs + pad

        # Stack into (N, 2)
        bounds = torch.stack([lower, upper], dim=1)
        return bounds


    def get_bounds(self) -> torch.Tensor:
        return self._bounds

    def _is_within_bounds(self, point: torch.Tensor) -> bool:
        """
        Checks if a given point is within the defined manifold bounds.

        Args:
            point (torch.Tensor): The point to check, expected to be a 1D tensor.

        Returns:
            bool: True if the point is within bounds, False otherwise.
        """
        if not isinstance(point, torch.Tensor) or point.ndim != 1:
            raise ValueError("Point must be a 1D torch.Tensor.")
        if point.shape[0] != self._n_dimensions:
            raise ValueError(f"Point dimension ({point.shape[0]}) does not match manifold dimension ({self._n_dimensions}).")

        epsilon = 1e-9
        return torch.all((point >= self._bounds[:, 0] - epsilon) & (point <= self._bounds[:, 1] + epsilon))

    def _clamp_point_to_bounds(self, point: torch.Tensor) -> torch.Tensor:
        """
        Clamps a given point to the defined manifold bounds.
        """
        # torch.clamp requires min and max to be broadcastable with input.
        # point shape: (n_dim), bounds shape: (n_dim, 2)
        # We need to compare point[i] with bounds[i,0] and bounds[i,1]
        return torch.max(self._bounds[:, 0], torch.min(point, self._bounds[:, 1]))

    def _get_grid_cell_info(self, point: torch.Tensor) -> Tuple[List[int], List[int], List[float]]:
        """
        Calculates the lower corner indices, upper corner indices, and fractional parts
        of a point within the grid for interpolation.
        """
        lower_indices = []
        upper_indices = []
        fractional_parts = []

        for d in range(self._n_dimensions):
            coord = point[d]
            axis_coords = self._grid_axes[d]
            n_points_in_dim = self._grid_points_per_dim[d]

            idx = torch.searchsorted(axis_coords, coord, side='right') - 1

            if idx == n_points_in_dim - 1 and coord >= axis_coords[n_points_in_dim - 1]:
                lower_idx = n_points_in_dim - 1
                upper_idx = n_points_in_dim - 1
                frac = 0.0
            elif idx < 0:
                lower_idx = 0
                upper_idx = 0
                frac = 0.0
            else:
                lower_idx = int(idx)
                upper_idx = min(int(idx) + 1, n_points_in_dim - 1)
                if axis_coords[upper_idx] == axis_coords[lower_idx]:
                    frac = 0.0
                else:
                    frac = (coord - axis_coords[lower_idx]) / (axis_coords[upper_idx] - axis_coords[lower_idx])

            lower_indices.append(lower_idx)
            upper_indices.append(upper_idx)
            fractional_parts.append(frac)

        return lower_indices, upper_indices, fractional_parts

    def _interpolate_metric_tensor(self, point: torch.Tensor) -> torch.Tensor:
        """
        Performs multi-linear interpolation of the metric tensor at a given point.
        """
        lower_indices, upper_indices, fractional_parts = self._get_grid_cell_info(point)

        interpolated_tensor = torch.zeros((self._n_dimensions, self._n_dimensions), device=self.device)

        for i in range(2**self._n_dimensions):
            current_corner_indices = [0] * self._n_dimensions
            weight = 1.0
            for d in range(self._n_dimensions):
                if (i >> d) & 1:
                    current_corner_indices[d] = upper_indices[d]
                    weight *= fractional_parts[d]
                else:
                    current_corner_indices[d] = lower_indices[d]
                    weight *= (1 - fractional_parts[d])

            corner_value = self._grid_cache[tuple(current_corner_indices)]

            if torch.isnan(corner_value).any():
                raise RuntimeError(
                    f"Grid cache at {tuple(current_corner_indices)} contains uncomputed values (NaN). "
                    "Call `compute_full_grid_metric_tensor()` first to populate the cache."
                )

            interpolated_tensor = interpolated_tensor + weight * corner_value
        return interpolated_tensor

    def metric_tensor(self, point: torch.Tensor) -> torch.Tensor:
        """
        Computes or retrieves the metric tensor at a given point.
        """
        if not self._is_within_bounds(point):
            raise ValueError(f"Point {point} is outside the defined manifold bounds: {self._bounds}")

        if self.cache_enabled:
            return self._interpolate_metric_tensor(point)
        else:
            return self._metric_tensor_func(point)

    def compute_full_grid_metric_tensor(self, force=False):
        """
        Computes and caches the metric tensor for all points on the defined grid.
        Args:
            force (bool): If True, recomputes the metric tensor even if it has already been computed.
        """
        if not self.cache_enabled:
            print("Caching is not enabled for this manifold instance. Full grid computation skipped.")
            return

        if self._is_filled and not force:
            print("Full grid metric tensor has already been computed. Use 'force=True' to recompute.")
            return

        grid_indices_iter = product(*[range(n_points) for n_points in self._grid_points_per_dim])
        total_grid_points = np.prod(self._grid_points_per_dim)

        self._grid_cache = {} # Clear cache before recomputation if force is True
        with tqdm(total=total_grid_points, desc="Computing metric tensors") as pbar:
            for indices_tuple in grid_indices_iter:
                point_coords_list = [
                    self._grid_axes[d][indices_tuple[d]]
                    for d in range(self._n_dimensions)
                ]
                point_coords = torch.stack(point_coords_list)
                
                with torch.no_grad(): # Disable gradient computation for filling the cache
                    tensor = self._metric_tensor_func(point_coords)
                self._grid_cache[tuple(indices_tuple)] = tensor
                pbar.update(1)

        self._is_filled = True # Set the flag to True after successful computation
        #print("Full grid metric tensor computation complete.")

    def get_dimension(self) -> int:
        return self._n_dimensions

    def _numerical_derivative_metric_tensor(self, point: torch.Tensor, dim_idx: int, h: float) -> torch.Tensor:
        """
        Numerically computes the partial derivative of the metric tensor.
        """
        h_vec = torch.zeros_like(point)
        h_vec[dim_idx] = h
        
        point_plus = point + h_vec
        point_minus = point - h_vec

        g_plus = self.metric_tensor(self._clamp_point_to_bounds(point_plus))
        g_minus = self.metric_tensor(self._clamp_point_to_bounds(point_minus))

        return (g_plus - g_minus) / (2 * h)

    def _numerical_second_derivative_metric_tensor_component(self, point: torch.Tensor, g_row: int, g_col: int,
                                                              diff_dim_idx1: int, diff_dim_idx2: int, h: float) -> float:
        """
        Numerically computes the second partial derivative of a single metric tensor component.
        """
        h_vec1 = torch.zeros_like(point)
        h_vec1[diff_dim_idx1] = h
        
        if diff_dim_idx1 == diff_dim_idx2:
            point_plus_h = point + h_vec1
            point_minus_h = point - h_vec1

            val_plus = self.metric_tensor(self._clamp_point_to_bounds(point_plus_h))[g_row, g_col]
            val_center = self.metric_tensor(self._clamp_point_to_bounds(point))[g_row, g_col]
            val_minus = self.metric_tensor(self._clamp_point_to_bounds(point_minus_h))[g_row, g_col]
            return (val_plus - 2 * val_center + val_minus) / (h**2)
        else:
            h_vec2 = torch.zeros_like(point)
            h_vec2[diff_dim_idx2] = h

            point_pp = point + h_vec1 + h_vec2
            point_pm = point + h_vec1 - h_vec2
            point_mp = point - h_vec1 + h_vec2
            point_mm = point - h_vec1 - h_vec2

            val_pp = self.metric_tensor(self._clamp_point_to_bounds(point_pp))[g_row, g_col]
            val_pm = self.metric_tensor(self._clamp_point_to_bounds(point_pm))[g_row, g_col]
            val_mp = self.metric_tensor(self._clamp_point_to_bounds(point_mp))[g_row, g_col]
            val_mm = self.metric_tensor(self._clamp_point_to_bounds(point_mm))[g_row, g_col]

            return (val_pp - val_pm - val_mp + val_mm) / (4 * h**2)

    def compute_true_gaussian_curvature(self, point: torch.Tensor, h: float = 1e-5) -> float:
        """
        Computes the true Gaussian curvature K for a 2D manifold at a given point.
        """
        if self._n_dimensions != 2:
            raise ValueError("True Gaussian curvature is defined for 2-dimensional manifolds only.")
        if len(point) != self._n_dimensions:
            raise ValueError(f"Point dimension ({len(point)}) does not match manifold dimension ({self._n_dimensions}).")

        g = self.metric_tensor(point)
        det_g = torch.linalg.det(g)

        if abs(det_g) < 1e-12:
            print(f"Warning: Metric tensor is near-degenerate (det={det_g:.2e}) at {point}. Returning NaN.")
            return torch.nan

        Gamma = self.compute_christoffel(point, h)

        term_d2g01_d0d1 = self._numerical_second_derivative_metric_tensor_component(point, 0, 1, 0, 1, h)
        term_d2g00_d1d1 = self._numerical_second_derivative_metric_tensor_component(point, 0, 0, 1, 1, h)
        term_d2g11_d0d0 = self._numerical_second_derivative_metric_tensor_component(point, 1, 1, 0, 0, h)

        riemann_deriv_terms = term_d2g01_d0d1 - 0.5 * (term_d2g00_d1d1 + term_d2g11_d0d0)
        
        # Using the formula R_{0101} = ... + g_{ab} (Γ^a_{11}Γ^b_{00} - Γ^a_{10}Γ^b_{01})
        # This can be computed with matrix operations for efficiency
        gamma_prod_1 = torch.einsum('a,b->ab', Gamma[:, 1, 1], Gamma[:, 0, 0]) # Γ^a_{11}Γ^b_{00}
        gamma_prod_2 = torch.einsum('a,b->ab', Gamma[:, 1, 0], Gamma[:, 0, 1]) # Γ^a_{10}Γ^b_{01}
        
        riemann_christoffel_terms = torch.sum(g * (gamma_prod_1 - gamma_prod_2))

        R_0101 = riemann_deriv_terms + riemann_christoffel_terms
        K = R_0101 / det_g
        return K

    def compute_christoffel(self, point: torch.Tensor, h: float = 1e-6) -> torch.Tensor:
        """
        Compute the Christoffel symbols at a point.
        """
        if len(point) != self._n_dimensions:
            raise ValueError(f"Point dimension ({len(point)}) does not match manifold dimension ({self._n_dimensions}).")

        g = self.metric_tensor(point)
        n = self._n_dimensions
        g_inv = torch.linalg.inv(g)

        dg = torch.stack([self._numerical_derivative_metric_tensor(point, i, h) for i in range(n)])

        # Christoffel symbols: Γ^k_{ij} = 1/2 * g^{kℓ} ( ∂_i g_{jℓ} + ∂_j g_{iℓ} - ∂_ℓ g_{ij} )
        # Using einsum for efficient computation
        # dg has shape (n, n, n) where dg[l, i, j] = ∂_l g_ij
        term1 = dg.permute(1, 2, 0) # ∂_i g_{jl} -> shape (j, l, i)
        term2 = dg.permute(2, 0, 1) # ∂_j g_{il} -> shape (l, i, j)
        term3 = dg # ∂_l g_{ij} -> shape (l, i, j)
        
        christoffel_first_kind = 0.5 * (term1 + term2 - term3)
        Gamma = torch.einsum('kl,lij->kij', g_inv, christoffel_first_kind)
        return Gamma

    def _geodesic_equation_solver_wrapper(self, t, y):
        """
        Wrapper for the augmented geodesic equation for torchdiffeq.odeint.
        Computes the derivative of the state [position, velocity, distance].
        """
        dim = (y.shape[0] - 1) // 2
        position = y[:dim]
        velocity = y[dim:2*dim]

        clamped_position = self._clamp_point_to_bounds(position)
        
        # Acceleration: a^i = -Γ^i_{jk} v^j v^k
        gamma = self.compute_christoffel(clamped_position)
        v_outer = torch.outer(velocity, velocity)
        acceleration = -torch.einsum('ijk,jk->i', gamma, v_outer)

        # Speed (derivative of distance): dL/dt = sqrt(v^T * g(pos) * v)
        g = self.metric_tensor(clamped_position)
        speed_sq = velocity @ g @ velocity
        speed = torch.sqrt(torch.relu(speed_sq) + 1e-12)

        return torch.cat([velocity, acceleration, torch.tensor([speed], device=self.device)])

    def compute_geodesic(self, start_point: torch.Tensor, end_point: torch.Tensor, num_points: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a geodesic between two points using a differentiable ODE solver.
        Also computes the exact length of the geodesic path by augmenting the state.
        
        Returns:
            A tuple containing:
            - geodesic_path (torch.Tensor): The points along the geodesic.
            - total_distance (torch.Tensor): The total length of the path.
        """
        if not self._is_within_bounds(start_point) or not self._is_within_bounds(end_point):
            raise ValueError("Start or end point is outside the defined manifold bounds.")

        dim = self._n_dimensions
        initial_direction = end_point - start_point
        g_start = self.metric_tensor(start_point)

        speed_sq = initial_direction @ g_start @ initial_direction
        initial_velocity = initial_direction / torch.sqrt(speed_sq) if speed_sq > 1e-9 else initial_direction

        # Initial state for solver: [position, velocity, distance]
        initial_distance = torch.tensor([0.0], device=self.device)
        initial_state = torch.cat([start_point, initial_velocity, initial_distance])

        euclidean_distance = torch.linalg.norm(end_point - start_point)
        integration_time = 2.0 * euclidean_distance
        t_eval = torch.linspace(0, integration_time, num_points, device=self.device)
        
        solution = odeint(
            self._geodesic_equation_solver_wrapper,
            initial_state,
            t_eval,
            method='rk4',
            rtol=1e-5,
            atol=1e-5
        )
        
        geodesic_path = solution[:, :dim]
        total_distance = solution[-1, -1]

        return geodesic_path, total_distance

    def exact_geodesic_distance(self, p1: torch.Tensor, p2: torch.Tensor, num_points: int = 50) -> torch.Tensor:
        """
        Computes the exact geodesic distance between two points by solving the
        augmented geodesic equation. This is the recommended method for distance calculation.
        """
        _, distance = self.compute_geodesic(p1, p2, num_points=num_points)
        return distance

    def create_riemannian_distance_matrix(self, data_points: torch.Tensor, 
                                          distance_calculator: Callable = None, 
                                          **kwargs) -> torch.Tensor:
        """
        Creates a pairwise Riemannian distance matrix for the given data points.
        """
        n_points = data_points.shape[0]
        dist_matrix = torch.zeros((n_points, n_points), device=self.device)
        total_calculations = n_points * (n_points - 1) // 2
        
        actual_distance_calculator = distance_calculator if distance_calculator is not None else self.exact_geodesic_distance
        
        if n_points > 0 and data_points.shape[1] != self._n_dimensions:
            raise ValueError(f"Dimension of data_points ({data_points.shape[1]}) does not match manifold dimension ({self._n_dimensions}).")

        print(f"Calculating {total_calculations} pairwise Riemannian distances using {actual_distance_calculator.__name__}...")
        with tqdm(total=total_calculations, desc="Calculating distances") as pbar:
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    u_point, v_point = data_points[i], data_points[j]
                    dist = actual_distance_calculator(u_point, v_point, **kwargs)
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
                    pbar.update(1)
        print("Distance matrix calculation complete.")
        return dist_matrix

    @staticmethod
    def compute_gaussian_curvature(G: torch.Tensor) -> torch.Tensor:
        """
        Compute a proxy for Gaussian curvature from a 2x2 metric tensor.
        """
        if G.shape != (2, 2):
            print("Warning: Gaussian curvature proxy is defined for 2x2 metric tensors. Returning NaN.")
            return torch.tensor(torch.nan)
        
        det_g = torch.linalg.det(G)
        # Use log1p for numerical stability: log(1+x)
        return torch.log1p(torch.relu(det_g))

    # --- Visualization Methods ---
    # These methods detach tensors from the computation graph for plotting with Matplotlib.

    def visualize_manifold_curvature(self, z_range: Union[Tuple[float, float], torch.Tensor, None] = None, resolution: int = 30,
                                     data_points: Union[torch.Tensor, None] = None, labels: Union[torch.Tensor, None] = None,
                                     h_curvature: float = 1e-3, exact_mode: bool = False):
        if self._n_dimensions != 2:
            raise ValueError("Curvature visualization is supported only for 2D manifolds.")
        
        with torch.no_grad():
            if z_range is None:
                bounds_np = self._bounds.cpu().numpy()
                plot_z1 = np.linspace(bounds_np[0, 0], bounds_np[0, 1], resolution)
                plot_z2 = np.linspace(bounds_np[1, 0], bounds_np[1, 1], resolution)
            # ... (rest of z_range handling)

            Z1_np, Z2_np = np.meshgrid(plot_z1, plot_z2)
            Z1, Z2 = torch.from_numpy(Z1_np).to(self.device), torch.from_numpy(Z2_np).to(self.device)
            
            curvature = torch.zeros((resolution, resolution), device=self.device)
            
            for i in range(resolution):
                for j in range(resolution):
                    z = torch.stack([Z1[i, j], Z2[i, j]])
                    try:
                        clamped_z = self._clamp_point_to_bounds(z)
                        if exact_mode:
                            curv_val = self.compute_true_gaussian_curvature(clamped_z, h=h_curvature)
                            curvature[i, j] = torch.log(curv_val + 1e-8)
                        else:
                            curv_val = self.compute_gaussian_curvature(self.metric_tensor(clamped_z))
                            curvature[i, j] = curv_val
                    except (ValueError, RuntimeError) as e:
                        print(f"Error computing curvature at point {z}: {e}. Setting to NaN.")
                        curvature[i, j] = torch.nan
            
            # Convert to numpy for plotting
            curvature_np = curvature.cpu().numpy()
            
            # --- Plotting Code (using Matplotlib as before) ---
            fig = plt.figure(figsize=(18, 6))
            # ... (The plotting logic is identical, just using the _np versions of arrays)
            vmin, vmax = np.nanmin(curvature_np), np.nanmax(curvature_np)
            if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax: vmin, vmax = -1, 1
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            if vmin <= 0 and vmax >= 0:
                cmap = 'RdBu_r'
            elif vmin < 0 and vmax <= 0:
                cmap = 'Blues_r'
            else:
                cmap = 'Reds'

            ax1 = fig.add_subplot(131)
            im = ax1.pcolormesh(Z1_np, Z2_np, curvature_np, cmap=cmap, norm=norm, shading='auto')
            ax1.set_title('Manifold Gaussian Curvature (2D View)')
            plt.colorbar(im, ax=ax1, label='Curvature')

            ax2 = fig.add_subplot(132, projection='3d')
            ax2.plot_surface(Z1_np, Z2_np, curvature_np, cmap=cmap, norm=norm)
            ax2.set_title('Manifold Gaussian Curvature (3D Surface)')
            
            ax3 = fig.add_subplot(133)
            contour = ax3.contourf(Z1_np, Z2_np, curvature_np, 20, cmap=cmap, norm=norm)
            ax3.set_title('Manifold Gaussian Curvature (Contour)')
            plt.colorbar(contour, ax=ax3, label='Curvature')
            
            if data_points is not None:
                points_np = data_points.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy() if labels is not None else np.zeros(len(points_np))
                ax1.scatter(points_np[:, 0], points_np[:, 1], c=labels_np, cmap='viridis', edgecolors='k')
                ax3.scatter(points_np[:, 0], points_np[:, 1], c=labels_np, cmap='viridis', edgecolors='k')

            plt.tight_layout()
            plt.show()

