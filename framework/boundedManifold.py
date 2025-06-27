import numpy as np
from typing import List, Tuple, Union, Callable
from itertools import product
from scipy.integrate import solve_ivp # New import for solving differential equations
from tqdm import tqdm # New import for progress bars
import matplotlib.pyplot as plt # New import for visualization
import matplotlib.colors as colors # New import for color normalization in visualization

class BoundedManifold:
    """
    Manifold class that wraps the metric tensor and allows for caching.
    It is written to encapsulate manifolds learned using a VAE.
    """
    def __init__(self, metric_tensor_func: callable, bounds: np.ndarray, cache: bool = False, grid_points_per_dim: Union[int, List[int]] = 10):
        """
        Initializes the BoundedManifold.

        Args:
            metric_tensor_func (callable): A callable that takes a numpy array (point)
                                           and returns the metric tensor at that point.
                                           Expected signature: metric_tensor_func(point: np.ndarray) -> np.ndarray
            bounds (np.ndarray): A 2D numpy array defining the bounds for each dimension.
                                 Shape should be (n_dimensions, 2), where each row is [min_val, max_val].
                                 Example for a 2D manifold: np.array([[0, 1], [0, 1]])
            cache (bool): If True, metric tensor values will be cached and interpolation will be used.
                          Requires `grid_points_per_dim` to be > 1 for meaningful interpolation.
            grid_points_per_dim (Union[int, List[int]]): The number of grid points per dimension.
                                                         If int, applies to all dimensions.
                                                         If List, specifies for each dimension.
                                                         Must be >= 2 for interpolation to work when caching is enabled.
        """
        if not isinstance(bounds, np.ndarray) or bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("Bounds must be a 2D numpy array of shape (n_dimensions, 2).")

        self._metric_tensor_func = metric_tensor_func
        self._bounds = bounds
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
            np.linspace(self._bounds[i, 0], self._bounds[i, 1], self._grid_points_per_dim[i])
            for i in range(self._n_dimensions)
        ]

        # Shape of the cache: (grid_dim1, grid_dim2, ..., metric_tensor_rows, metric_tensor_cols)
        # Initialize with NaN to indicate not computed, will be filled by compute_full_grid_metric_tensor
        cache_shape = tuple(self._grid_points_per_dim) + (self._n_dimensions, self._n_dimensions)
        self._grid_cache = np.full(cache_shape, np.nan)

    def get_bounds(self) -> np.ndarray:
        return self._bounds

    def _is_within_bounds(self, point: np.ndarray) -> bool:
        """
        Checks if a given point is within the defined manifold bounds.

        Args:
            point (np.ndarray): The point to check, expected to be a 1D numpy array.

        Returns:
            bool: True if the point is within bounds, False otherwise.
        """
        if not isinstance(point, np.ndarray) or point.ndim != 1:
            raise ValueError("Point must be a 1D numpy array.")
        if point.shape[0] != self._n_dimensions:
            raise ValueError(f"Point dimension ({point.shape[0]}) does not match manifold dimension ({self._n_dimensions}).")

        # Check if each coordinate of the point is within its respective bounds
        # Use a small epsilon for floating point comparisons at boundaries
        epsilon = 1e-9 # A small tolerance for floating point comparisons
        for i in range(self._n_dimensions):
            min_val, max_val = self._bounds[i]
            # Check if point[i] is effectively within [min_val, max_val]
            if not (min_val - epsilon <= point[i] <= max_val + epsilon):
                return False
        return True

    def _clamp_point_to_bounds(self, point: np.ndarray) -> np.ndarray:
        """
        Clamps a given point to the defined manifold bounds.
        This is used internally by methods that might generate points
        slightly outside the strict bounds (e.g., numerical derivatives, ODE solvers).
        """
        clamped_point = point.copy()
        for i in range(self._n_dimensions):
            min_val, max_val = self._bounds[i]
            clamped_point[i] = np.clip(clamped_point[i], min_val, max_val)
        return clamped_point

    def _get_grid_cell_info(self, point: np.ndarray) -> Tuple[List[int], List[int], List[float]]:
        """
        Calculates the lower corner indices, upper corner indices, and fractional parts
        of a point within the grid for interpolation.

        Args:
            point (np.ndarray): The point for which to get grid cell info.

        Returns:
            Tuple[List[int], List[int], List[float]]:
                - lower_indices: List of integer indices for the lower corner of the cell.
                - upper_indices: List of integer indices for the upper corner of the cell.
                - fractional_parts: List of fractional parts of the point within the cell (0 to 1).
        """
        lower_indices = []
        upper_indices = []
        fractional_parts = []

        for d in range(self._n_dimensions):
            coord = point[d]
            axis_coords = self._grid_axes[d]
            n_points_in_dim = self._grid_points_per_dim[d]

            # Find the interval where the coordinate lies
            # np.searchsorted gives the index *before* which the value could be inserted
            idx = np.searchsorted(axis_coords, coord, side='right') - 1

            # Handle edge cases for index:
            # If coord is exactly the maximum value on this axis,
            # or if it's slightly beyond due to float precision, clamp it to the last valid index
            if idx == n_points_in_dim - 1 and coord >= axis_coords[n_points_in_dim - 1]:
                lower_idx = n_points_in_dim - 1
                upper_idx = n_points_in_dim - 1 # Point is exactly on the last grid point
                frac = 0.0 # No interpolation needed along this dimension, effectively
            elif idx < 0: # Point is below the first grid point (should be caught by _is_within_bounds, but as safeguard)
                lower_idx = 0
                upper_idx = 0
                frac = 0.0
            else:
                lower_idx = idx
                upper_idx = min(idx + 1, n_points_in_dim - 1)
                # Calculate fractional part
                if axis_coords[upper_idx] == axis_coords[lower_idx]: # Should not happen with linspace >= 2 points, but safeguard
                    frac = 0.0
                else:
                    frac = (coord - axis_coords[lower_idx]) / (axis_coords[upper_idx] - axis_coords[lower_idx])

            lower_indices.append(lower_idx)
            upper_indices.append(upper_idx)
            fractional_parts.append(frac)

        return lower_indices, upper_indices, fractional_parts

    def _interpolate_metric_tensor(self, point: np.ndarray) -> np.ndarray:
        """
        Performs multi-linear interpolation of the metric tensor at a given point
        using the pre-computed grid cache.

        Args:
            point (np.ndarray): The point at which to interpolate the metric tensor.

        Returns:
            np.ndarray: The interpolated metric tensor at the specified point.
        """
        lower_indices, upper_indices, fractional_parts = self._get_grid_cell_info(point)

        # Initialize interpolated tensor with zeros (shape: n_dimensions x n_dimensions)
        interpolated_tensor = np.zeros((self._n_dimensions, self._n_dimensions))

        # Iterate over all 2^N corners of the cell
        # This uses binary representation to generate all combinations of lower/upper indices for each dimension
        for i in range(2**self._n_dimensions):
            current_corner_indices = [0] * self._n_dimensions
            weight = 1.0
            for d in range(self._n_dimensions):
                # Check the d-th bit of i to decide between lower or upper index for this dimension
                if (i >> d) & 1: # If bit is 1, use upper index for this dimension
                    current_corner_indices[d] = upper_indices[d]
                    weight *= fractional_parts[d]
                else: # If bit is 0, use lower index for this dimension
                    current_corner_indices[d] = lower_indices[d]
                    weight *= (1 - fractional_parts[d])

            # Retrieve the metric tensor value from the pre-computed grid cache for the current corner
            corner_value = self._grid_cache[tuple(current_corner_indices)]

            # If any part of the corner value is NaN, it means the grid point hasn't been computed.
            # This indicates `compute_full_grid_metric_tensor` was likely not called or failed.
            # In such cases, linear interpolation cannot proceed reliably.
            # For this implementation, we will raise an error, as the expectation is a pre-filled cache.
            if np.isnan(corner_value).any():
                raise RuntimeError(
                    f"Grid cache at {tuple(current_corner_indices)} contains uncomputed values (NaN). "
                    "Call `compute_full_grid_metric_tensor()` first to populate the cache."
                )

            interpolated_tensor += weight * corner_value
        return interpolated_tensor

    def metric_tensor(self, point: np.ndarray) -> np.ndarray:
        """
        Computes or retrieves the metric tensor at a given point.
        If caching is enabled, it uses linear interpolation based on the grid cache.
        If caching is disabled, it computes it directly using the provided metric_tensor_func.

        Args:
            point (np.ndarray): The point (coordinates) at which to compute the metric tensor.

        Returns:
            np.ndarray: The metric tensor at the specified point.

        Raises:
            ValueError: If the point is outside the defined manifold bounds or has an incorrect dimension.
            RuntimeError: If caching is enabled but the grid cache is not fully populated.
        """
        # First, validate the point's bounds and dimension
        if not self._is_within_bounds(point):
            raise ValueError(f"Point {point} is outside the defined manifold bounds: {self._bounds}")

        if self.cache_enabled:
            # If caching is enabled, use interpolation from the grid
            return self._interpolate_metric_tensor(point)
        else:
            # If caching is not enabled, directly compute and return
            return self._metric_tensor_func(point)

    def compute_full_grid_metric_tensor(self):
        """
        Computes and caches the metric tensor for all points on the defined grid.
        This method should be called once to populate the cache for interpolation
        when caching is enabled.
        """
        if not self.cache_enabled:
            print("Caching is not enabled for this manifold instance. Full grid computation skipped.")
            return

        # Use itertools.product to iterate through all combinations of grid indices
        # Example for a 2D grid with 3 points per dimension: product(range(3), range(3))
        # gives (0,0), (0,1), (0,2), (1,0), ... (2,2)
        grid_indices_iter = product(*[range(n_points) for n_points in self._grid_points_per_dim])

        total_grid_points = np.prod(self._grid_points_per_dim)
        #print(f"Computing metric tensors for {total_grid_points} grid points...")

        with tqdm(total=total_grid_points, desc="Computing metric tensors") as pbar:
            for indices_tuple in grid_indices_iter:
                # Construct the actual point coordinates from the grid indices using _grid_axes
                point_coords = np.array([
                    self._grid_axes[d][indices_tuple[d]]
                    for d in range(self._n_dimensions)
                ])
                # Compute the metric tensor at this specific grid point
                tensor = self._metric_tensor_func(point_coords)
                # Store the computed tensor in the cache at the corresponding grid indices
                self._grid_cache[tuple(indices_tuple)] = tensor
                pbar.update(1) # Increment progress bar

        print("Full grid metric tensor computation complete.")

    def get_dimension(self) -> int:
        """
        Returns the dimension of the manifold.

        Returns:
            int: The number of dimensions of the manifold.
        """
        return self._n_dimensions

    def _numerical_derivative_metric_tensor(self, point: np.ndarray, dim_idx: int, h: float) -> np.ndarray:
        """
        Numerically computes the partial derivative of the metric tensor with respect
        to the specified dimension at a given point using central difference.

        Args:
            point (np.ndarray): The point at which to compute the derivative.
            dim_idx (int): The index of the dimension (0 to n_dimensions-1)
                           with respect to which to differentiate.
            h (float): Step size for numerical differentiation.

        Returns:
            np.ndarray: The derivative of the metric tensor (n x n matrix).
        """
        point_plus = point.copy()
        point_plus[dim_idx] += h
        point_minus = point.copy()
        point_minus[dim_idx] -= h

        # Clamp points to bounds before calling the metric_tensor method to avoid ValueErrors
        # when numerical differentiation steps go slightly out of bounds.
        g_plus = self.metric_tensor(self._clamp_point_to_bounds(point_plus))
        g_minus = self.metric_tensor(self._clamp_point_to_bounds(point_minus))

        return (g_plus - g_minus) / (2 * h)

    def _numerical_second_derivative_metric_tensor_component(self, point: np.ndarray, g_row: int, g_col: int,
                                                              diff_dim_idx1: int, diff_dim_idx2: int, h: float) -> float:
        """
        Numerically computes the second partial derivative of a single metric tensor component
        (g_row, g_col) with respect to two specified dimensions (diff_dim_idx1, diff_dim_idx2)
        at a given point using central difference approximations.

        Args:
            point (np.ndarray): The point at which to compute the derivative.
            g_row (int): Row index of the metric tensor component.
            g_col (int): Column index of the metric tensor component.
            diff_dim_idx1 (int): Index of the first dimension for differentiation.
            diff_dim_idx2 (int): Index of the second dimension for differentiation.
            h (float): Step size for numerical differentiation.

        Returns:
            float: The value of the second partial derivative.
        """
        if diff_dim_idx1 == diff_dim_idx2: # Unmixed second derivative ∂² / (∂x^i)²
            point_plus_h = point.copy()
            point_plus_h[diff_dim_idx1] += h
            point_minus_h = point.copy()
            point_minus_h[diff_dim_idx1] -= h

            # Clamp points to bounds for robustness
            val_plus = self.metric_tensor(self._clamp_point_to_bounds(point_plus_h))[g_row, g_col]
            val_center = self.metric_tensor(self._clamp_point_to_bounds(point))[g_row, g_col]
            val_minus = self.metric_tensor(self._clamp_point_to_bounds(point_minus_h))[g_row, g_col]
            return (val_plus - 2 * val_center + val_minus) / (h**2)
        else: # Mixed second derivative ∂² / (∂x^i ∂x^j)
            # Uses the formula: [f(x+h_i, y+h_j) - f(x+h_i, y-h_j) - f(x-h_i, y+h_j) + f(x-h_i, y-h_j)] / (4h^2)
            point_pp = point.copy() # x+h_i, y+h_j
            point_pp[diff_dim_idx1] += h
            point_pp[diff_dim_idx2] += h

            point_pm = point.copy() # x+h_i, y-h_j
            point_pm[diff_dim_idx1] += h
            point_pm[diff_dim_idx2] -= h

            point_mp = point.copy() # x-h_i, y+h_j
            point_mp[diff_dim_idx1] -= h
            point_mp[diff_dim_idx2] += h

            point_mm = point.copy() # x-h_i, y-h_j
            point_mm[diff_dim_idx1] -= h
            point_mm[diff_dim_idx2] -= h

            # Clamp points to bounds for robustness
            val_pp = self.metric_tensor(self._clamp_point_to_bounds(point_pp))[g_row, g_col]
            val_pm = self.metric_tensor(self._clamp_point_to_bounds(point_pm))[g_row, g_col]
            val_mp = self.metric_tensor(self._clamp_point_to_bounds(point_mp))[g_row, g_col]
            val_mm = self.metric_tensor(self._clamp_point_to_bounds(point_mm))[g_row, g_col]

            return (val_pp - val_pm - val_mp + val_mm) / (4 * h**2)

    def compute_true_gaussian_curvature(self, point: np.ndarray, h: float = 1e-5) -> float:
        """
        Computes the true Gaussian curvature K for a 2D manifold at a given point.
        Requires computing second derivatives of the metric tensor.

        Formula for R_{0101} (using 0-based indexing for coordinates x^0, x^1):
        R_{0101} = 0.5 * ( ∂^2 g_{01} / ∂x^1 ∂x^0 - ∂^2 g_{00} / ∂x^1 ∂x^1 - ∂^2 g_{11} / ∂x^0 ∂x^0 + ∂^2 g_{10} / ∂x^0 ∂x^1 )
                   + g_{ab} (Γ^a_{01}Γ^b_{00} - Γ^a_{00}Γ^b_{01})  <-- Incorrect from original request, using standard formula
        
        Using the standard formula R_{ijkl} = 1/2 (∂_j∂_k g_il - ∂_j∂_l g_ik - ∂_i∂_k g_jl + ∂_i∂_l g_jk)
                                            + g_mn (Γ^m_jl Γ^n_ik - Γ^m_jk Γ^n_il)
        For R_{0101} (i=0, j=1, k=0, l=1):
        R_{0101} = 1/2 (∂_1∂_0 g_01 - ∂_1∂_1 g_00 - ∂_0∂_0 g_11 + ∂_0∂_1 g_10)
                   + g_{ab} (Γ^a_{11}Γ^b_{00} - Γ^a_{10}Γ^b_{01})
        Assuming g_{01} = g_{10} and mixed partials commute:
        R_{0101} = ∂_0∂_1 g_01 - 1/2 (∂_1∂_1 g_00 + ∂_0∂_0 g_11)
                   + g_{ab} (Γ^a_{11}Γ^b_{00} - Γ^a_{10}Γ^b_{01})

        Args:
            point (np.ndarray): The point at which to compute the Gaussian curvature.
            h (float): Step size for numerical differentiation.

        Returns:
            float: The Gaussian curvature K at the specified point.

        Raises:
            ValueError: If the manifold is not 2-dimensional or point has incorrect dimension.
            RuntimeError: If metric tensor is degenerate or cannot be inverted.
        """
        if self._n_dimensions != 2:
            raise ValueError("True Gaussian curvature is defined for 2-dimensional manifolds only.")
        if len(point) != self._n_dimensions:
            raise ValueError(f"Point dimension ({len(point)}) does not match manifold dimension ({self._n_dimensions}).")

        g = self.metric_tensor(point)
        det_g = np.linalg.det(g)

        if abs(det_g) < 1e-12: # Check for near-degenerate metric to avoid division by zero
            print(f"Warning: Metric tensor is near-degenerate (det={det_g:.2e}) at {point}. Gaussian curvature may be unstable or undefined.")
            return np.nan # Return NaN for undefined or unstable cases

        Gamma = self.compute_christoffel(point, h) # Christoffel symbols Γ^k_{ij}

        # --- Compute components for the first part of R_{0101} (derivatives of metric) ---
        # ∂²g_01 / ∂x⁰∂x¹ (or ∂²g_10 / ∂x¹∂x⁰)
        term_d2g01_d0d1 = self._numerical_second_derivative_metric_tensor_component(point, 0, 1, 0, 1, h)
        # ∂²g_00 / ∂x¹∂x¹
        term_d2g00_d1d1 = self._numerical_second_derivative_metric_tensor_component(point, 0, 0, 1, 1, h)
        # ∂²g_11 / ∂x⁰∂x⁰
        term_d2g11_d0d0 = self._numerical_second_derivative_metric_tensor_component(point, 1, 1, 0, 0, h)

        riemann_deriv_terms = term_d2g01_d0d1 - 0.5 * (term_d2g00_d1d1 + term_d2g11_d0d0)

        # --- Compute components for the second part of R_{0101} (Christoffel symbol products) ---
        riemann_christoffel_terms = 0.0
        n = self._n_dimensions # which is 2 for 2D manifold
        # Sum over a and b (from 0 to 1)
        for a in range(n):
            for b in range(n):
                # Terms are g_{ab} (Γ^a_{11}Γ^b_{00} - Γ^a_{10}Γ^b_{01})
                # Note: original formulation of R_ijkl has different indices for Gamma terms based on j,k,l
                # Let's use the explicit R_0101 terms
                # R_{0101} = ∂_1 Γ^2_{10} - ∂_0 Γ^2_{11} + Γ^2_{1k} Γ^k_{10} - Γ^2_{0k} Γ^k_{11} (This is 3rd order derivative)

                # Back to the formula: R_{ijkl} = 1/2 (...) + g_{mn} (Γ^m_{jl} Γ^n_{ik} - Γ^m_{jk} Γ^n_{il})
                # For R_{0101}, i=0, j=1, k=0, l=1
                # The summation part is: g_{mn} (Γ^m_{11} Γ^n_{00} - Γ^m_{10} Γ^n_{01})
                term_prod_gamma1 = Gamma[a, 1, 1] * Gamma[b, 0, 0] # Γ^a_{11}Γ^b_{00}
                term_prod_gamma2 = Gamma[a, 1, 0] * Gamma[b, 0, 1] # Γ^a_{10}Γ^b_{01}
                riemann_christoffel_terms += g[a, b] * (term_prod_gamma1 - term_prod_gamma2)

        R_0101 = riemann_deriv_terms + riemann_christoffel_terms

        # Gaussian curvature K = R_{0101} / det(g)
        K = R_0101 / det_g
        return K

    @staticmethod
    def _geodesic_equation_solver_wrapper(t, y, manifold_instance):
        """
        Wrapper for the geodesic equation, designed to be passed to solve_ivp.
        It accesses the BoundedManifold instance's methods to compute Christoffel symbols.

        Args:
            t: Time parameter (required by solve_ivp, but not directly used here).
            y (np.ndarray): State vector [position, velocity].
            manifold_instance (BoundedManifold): The instance of BoundedManifold
                                                 to use for metric tensor and Christoffel computations.

        Returns:
            np.ndarray: Derivatives [velocity, acceleration].
        """
        dim = len(y) // 2
        position_raw = y[:dim]
        velocity = y[dim:]

        # Clamp the position to ensure it's within manifold bounds before computing Christoffel symbols.
        # This prevents solve_ivp from probing points too far out of bounds and raising errors.
        position = manifold_instance._clamp_point_to_bounds(position_raw)

        # Calculate Christoffel symbols at current position using the instance's method
        gamma = manifold_instance.compute_christoffel(position)

        # Calculate acceleration using the geodesic equation
        acceleration = np.zeros(dim)
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    acceleration[i] -= gamma[i, j, k] * velocity[j] * velocity[k]

        return np.concatenate([velocity, acceleration])
    

    def compute_christoffel(self, point: np.ndarray, h: float = 1e-6) -> np.ndarray:
        """
        Compute the Christoffel symbols at point.

        Args:
            point (np.ndarray): The point at which to compute the Christoffel symbols.
            h (float): Step size for numerical differentiation.

        Returns:
            np.ndarray: A 3-dimensional array Gamma of shape (n, n, n) where
                        Gamma[k, i, j] corresponds to \Gamma^k_{ij}.

        Raises:
            ValueError: If the point has an incorrect dimension. (Bounds are handled by internal calls).
        """
        if len(point) != self._n_dimensions:
            raise ValueError(f"Point dimension ({len(point)}) does not match manifold dimension ({self._n_dimensions}).")

        # The point itself should be within bounds. The metric_tensor method will
        # implicitly check this or raise an error if not for the main point.
        # However, _numerical_derivative_metric_tensor handles clamping for its intermediate points.
        g = self.metric_tensor(point)
        n = self._n_dimensions
        g_inv = np.linalg.inv(g)

        # Compute partial derivatives of the metric tensor
        dg = [self._numerical_derivative_metric_tensor(point, i, h) for i in range(n)]

        Gamma = np.zeros((n, n, n))
        # Christoffel symbols of the second kind formula:
        # Gamma^k_{ij} = 1/2 * g^{k\ell} ( ∂_i g_{jℓ} + ∂_j g_{iℓ} - ∂_ℓ g_{ij} )
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    sum_term = 0.0
                    for l in range(n):
                        term = (dg[i][j, l] + dg[j][i, l] - dg[l][i, j])
                        sum_term += g_inv[k, l] * term
                    Gamma[k, i, j] = 0.5 * sum_term
        return Gamma

    def compute_geodesic(self, start_point: np.ndarray, end_point: np.ndarray, num_points: int = 50) -> np.ndarray:
        """
        Compute a geodesic between two points by numerically solving the geodesic equation.

        Args:
            start_point (np.ndarray): Starting point.
            end_point (np.ndarray): Ending point.
            num_points (int): Number of points in the resulting geodesic.

        Returns:
            np.ndarray: Array of points along the geodesic.

        Raises:
            ValueError: If start or end point is outside bounds or has incorrect dimension.
        """
        # Ensure initial/final points are strictly within bounds before starting integration
        if not self._is_within_bounds(start_point) or not self._is_within_bounds(end_point):
            raise ValueError("Start or end point is outside the defined manifold bounds.")
        if len(start_point) != self._n_dimensions or len(end_point) != self._n_dimensions:
             raise ValueError("Start or end point dimension does not match manifold dimension.")

        dim = self._n_dimensions

        # Initial velocity is in the direction of end_point - start_point
        initial_direction = end_point - start_point
        g_start = self.metric_tensor(start_point)

        # Compute metric-normalized velocity
        speed = np.sqrt(initial_direction @ g_start @ initial_direction)
        if speed > 0:
            initial_velocity = initial_direction / speed
        else:
            # Default to Euclidean direction if points are nearly coincident or speed is zero
            initial_velocity = initial_direction

        # Initial state for solver: [position, velocity]
        initial_state = np.concatenate([start_point, initial_velocity])

        # Estimate the integration time based on Euclidean distance
        # Reverting to a general heuristic for integration time for flexibility.
        # This gives the solver enough "time" to potentially reach the endpoint,
        # but the final_distance check will still warn if it's too far.
        euclidean_distance = np.linalg.norm(end_point - start_point)
        integration_time = 2.0 * euclidean_distance # Reverted to 2.0x, can be adjusted further

        # Solve the geodesic equation
        t_eval = np.linspace(0, integration_time, num_points)
        solution = solve_ivp(
            fun=BoundedManifold._geodesic_equation_solver_wrapper, # Pass the static method
            t_span=[0, integration_time], # Time span for integration
            y0=initial_state, # Initial state
            args=(self,), # Pass 'self' as an argument to the wrapper function
            t_eval=t_eval, # Times at which to store the computed solution
            method='RK45', # Runge-Kutta method of order 4(5)
            rtol=1e-6 # Relative tolerance for the solver
        )

        # Extract positions from the solution (first 'dim' rows)
        geodesic_path = solution.y[:dim, :].T

        # Ensure the path ends near the desired end point
        final_distance = np.linalg.norm(geodesic_path[-1] - end_point)

        # Fallback heuristic for 2D manifolds when numerical integration is not accurate
        # The warning message is refined to provide more context for the user.
        if dim == 2 and final_distance > 0.2 * euclidean_distance:
            print(f"Warning: Geodesic calculation from {start_point} to {end_point} "
                  f"ended with a final distance of {final_distance:.4f} from the target. "
                  "Numerical integration (initial value problem) does not guarantee exact arrival at an endpoint. "
                  "Using a heuristic fallback for 2D. For improved accuracy, consider: "
                  "1. Increasing 'num_points' (more fine-grained path). "
                  "2. Adjusting 'integration_time' (e.g., trying a larger factor, though '2.0 * euclidean_distance' is a common start). "
                  "3. Reducing 'rtol' for the solver (more strict numerical precision). "
                  "This is especially relevant for highly curved manifolds."
            )

            t_bezier = np.linspace(0, 1, num_points)
            geodesic_path = np.zeros((num_points, dim))

            # Heuristic to find a suitable control point for a quadratic Bezier curve
            grid_size = 5
            midpoint = (start_point + end_point) / 2
            span = max(0.5, euclidean_distance / 4)

            # Create a small grid around the midpoint for curvature sampling
            x = np.linspace(midpoint[0] - span, midpoint[0] + span, grid_size)
            y = np.linspace(midpoint[1] - span, midpoint[1] + span, grid_size)
            X, Y = np.meshgrid(x, y)

            grid_curvature = np.zeros((grid_size, grid_size))
            for i_idx_grid in range(grid_size):
                for j_idx_grid in range(grid_size):
                    point_curr = np.array([X[i_idx_grid, j_idx_grid], Y[i_idx_grid, j_idx_grid]])
                    try:
                        g = self.metric_tensor(self._clamp_point_to_bounds(point_curr)) # Clamp for curvature sampling
                        grid_curvature[i_idx_grid, j_idx_grid] = BoundedManifold.compute_gaussian_curvature(g)
                    except (ValueError, RuntimeError):
                        # If point is out of bounds or cache is uncomputed, assume flat curvature for heuristic
                        grid_curvature[i_idx_grid, j_idx_grid] = 0

            # Find point with lowest absolute curvature as a control point
            min_idx = np.argmin(np.abs(grid_curvature))
            i_min, j_min = np.unravel_index(min_idx, grid_curvature.shape)
            control_point = np.array([X[i_min, j_min], Y[i_min, j_min]])

            # Add slight offset perpendicular to the direct path, to introduce a curve
            path_vector = end_point - start_point
            # Generate a perpendicular vector (only works reliably for 2D)
            perpendicular = np.array([-path_vector[1], path_vector[0]])
            perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-10) # Normalize to avoid division by zero
            control_point += perpendicular * (euclidean_distance * 0.2) # Offset magnitude

            # Create a simple quadratic Bezier curve for the path
            for i_pt, ti in enumerate(t_bezier):
                geodesic_path[i_pt] = (1-ti)**2 * start_point + 2*(1-ti)*ti * control_point + ti**2 * end_point

        return geodesic_path

    def _geodesic_distance(self, p1: np.ndarray, p2: np.ndarray, num_points: int = 50) -> float:
        """
        Computes the approximate geodesic distance between two points by integrating
        the length element along the computed geodesic path.

        Args:
            p1 (np.ndarray): The first point.
            p2 (np.ndarray): The second point.
            num_points (int): Number of points to use for computing the geodesic path.

        Returns:
            float: The approximate geodesic distance.
        """
        # Compute the geodesic path between the two points
        geodesic_path = self.compute_geodesic(p1, p2, num_points=num_points)
        distance = 0.0
        # Sum the lengths of the segments along the geodesic path
        for i in range(len(geodesic_path) - 1):
            segment_start = geodesic_path[i]
            segment_end = geodesic_path[i+1]
            segment_vector = segment_end - segment_start
            
            # Use the metric at the midpoint of the segment for a better approximation of length
            midpoint = (segment_start + segment_end) / 2
            try:
                # Get the metric tensor at the midpoint of the segment. Clamp it for safety.
                g_mid = self.metric_tensor(self._clamp_point_to_bounds(midpoint))
                # Compute the length of the segment using the metric
                # Length = sqrt(v^T * G * v)
                distance += np.sqrt(segment_vector @ g_mid @ segment_vector)
            except (ValueError, RuntimeError):
                # If midpoint is out of bounds or cache is uncomputed, fall back to Euclidean distance for this segment
                print(f"Warning: Cannot compute metric at midpoint {midpoint}. Falling back to Euclidean distance for segment.")
                distance += np.linalg.norm(segment_vector)
        return distance


    def create_riemannian_distance_matrix(self, data_points: np.ndarray, 
                                          distance_calculator: Callable[[np.ndarray, np.ndarray], float] = None, 
                                          **kwargs) -> np.ndarray:
        """
        Creates a pairwise Riemannian distance matrix for the given data points.
        Allows specifying a custom distance calculation method.

        Args:
            data_points (np.ndarray): Array of data points (each row is a point).
                                      Expected shape (N, n_dimensions).
            distance_calculator (Callable[[np.ndarray, np.ndarray], float], optional):
                A callable method from this BoundedManifold instance that takes two points
                (u, v) and returns a distance. Optional additional keyword arguments
                will be passed to this callable.
                If None, defaults to `self._geodesic_distance`.
                Examples: `self._geodesic_distance`, `self.linear_interpolation_distance`,
                `self.midpoint_approximation`, etc.
            **kwargs: Additional keyword arguments to pass to the `distance_calculator` method.
                      E.g., `num_points` for `_geodesic_distance` or `linear_interpolation_distance`.

        Returns:
            np.ndarray: The pairwise Riemannian distance matrix (N x N).
        """
        n_points = data_points.shape[0]
        dist_matrix = np.zeros((n_points, n_points))
        total_calculations = n_points * (n_points - 1) // 2
        
        # Set default distance calculator if not provided
        actual_distance_calculator = distance_calculator if distance_calculator is not None else self._geodesic_distance
        
        # Validate the dimensions of the input data_points
        if n_points > 0 and data_points.shape[1] != self._n_dimensions:
            raise ValueError(f"Dimension of data_points ({data_points.shape[1]}) does not match manifold dimension ({self._n_dimensions}).")

        print(f"Calculating {total_calculations} pairwise Riemannian distances using {actual_distance_calculator.__name__}...")
        # Use tqdm for a progress bar during computation
        with tqdm(total=total_calculations, desc="Calculating distances") as pbar:
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    u_point, v_point = data_points[i], data_points[j]
                    
                    try:
                        # Call the chosen distance calculator with the points and any extra kwargs
                        dist = actual_distance_calculator(u_point, v_point, **kwargs)
                    except TypeError as e:
                        print(f"Error calling distance calculator '{actual_distance_calculator.__name__}': {e}. "
                              "Check if the method accepts the provided kwargs. Falling back to Euclidean.")
                        dist = self.euclidean_distance(u_point, v_point)
                    except Exception as e:
                        print(f"An unexpected error occurred during distance calculation for pair ({i},{j}) "
                              f"using {actual_distance_calculator.__name__}: {e}. Falling back to Euclidean.")
                        dist = self.euclidean_distance(u_point, v_point)

                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist # Distance matrix is symmetric
                    pbar.update(1) # Increment progress bar
        print("Distance matrix calculation complete.")
        return dist_matrix

    @staticmethod
    def compute_gaussian_curvature(G: np.ndarray) -> float:
        """
        Compute a proxy for Gaussian curvature from a 2x2 metric tensor.
        NOTE: This function provides a simplified proxy for Gaussian curvature,
        using the determinant of the metric tensor as an indicator of local distortion.
        A true Gaussian curvature calculation typically involves derivatives of the metric tensor
        (Riemann curvature tensor components).

        Args:
            G (np.ndarray): 2x2 metric tensor.

        Returns:
            float: A value representing a proxy for distortion/curvature.
                   Returns -1.0 if the metric tensor is not 2x2 or is degenerate.
        """
        if G.shape != (2, 2):
            # Gaussian curvature is formally defined for 2D surfaces.
            # In higher dimensions, one typically looks at sectional curvatures.
            print("Warning: Gaussian curvature computation is defined for 2x2 metric tensors. Returning -1.0.")
            return -1.0
        
        det_g = np.linalg.det(G)
        if det_g > 0:
            # Using log1p (log(1+x)) to handle small positive determinants gracefully
            # and provide a value that increases with distortion.
            return np.log1p(det_g)
        else:
            # Handle the case of degenerate or ill-conditioned metric (det_g <= 0)
            return -1.0 # Return a distinct negative value for invalid cases

    def visualize_manifold_curvature(self, z_range: Union[Tuple[float, float], np.ndarray, None] = None, resolution: int = 30,
                                     data_points: Union[np.ndarray, None] = None, labels: Union[np.ndarray, None] = None,
                                     h_curvature: float = 1e-3, exact_mode: bool = False):
        """
        Visualize the **true Gaussian curvature** of a 2D manifold in latent space with optional data points.
        This function requires the BoundedManifold instance to have caching enabled
        and its cache populated using `compute_full_grid_metric_tensor()`.

        Args:
            z_range (Union[Tuple[float, float], np.ndarray, None]):
                Range of the latent space coordinates for plotting.
                If a tuple (min, max), applies to all dimensions.
                If a 2D np.ndarray ([[min1, max1], [min2, max2]]), specifies per-dimension range.
                If None, uses the manifold's intrinsic bounds (`_bounds`).
            resolution (int): Number of points in each dimension for the visualization grid.
            data_points (Optional[np.ndarray]): Optional array containing data points on the manifold.
                                                 Shape (N, n_dimensions). Will be truncated to 2D for plotting.
            labels (Optional[np.ndarray]): Optional array of integer labels for the data points.
            h_curvature (float): Step size for numerical differentiation when computing true Gaussian curvature.
        """
        if not self.cache_enabled:
            raise ValueError("Visualization requires caching to be enabled for the BoundedManifold instance.")
        if self._n_dimensions != 2:
            raise ValueError("Curvature visualization is currently supported only for 2D manifolds.")
        
        # Determine the z_range for plotting
        if z_range is None:
            plot_z1 = np.linspace(self._bounds[0, 0], self._bounds[0, 1], resolution)
            plot_z2 = np.linspace(self._bounds[1, 0], self._bounds[1, 1], resolution)
        elif isinstance(z_range, tuple) and len(z_range) == 2:
            plot_z1 = np.linspace(z_range[0], z_range[1], resolution)
            plot_z2 = np.linspace(z_range[0], z_range[1], resolution)
        elif isinstance(z_range, np.ndarray) and z_range.ndim == 2 and z_range.shape[1] == 2:
            plot_z1 = np.linspace(z_range[0, 0], z_range[0, 1], resolution)
            plot_z2 = np.linspace(z_range[1, 0], z_range[1, 1], resolution)
        else:
            raise ValueError("Invalid z_range format. Must be None, a tuple (min, max), or a 2D np.ndarray [[min1,max1],[min2,max2]].")

        Z1, Z2 = np.meshgrid(plot_z1, plot_z2)
        
        curvature = np.zeros((resolution, resolution))
        
        # Compute true Gaussian curvature at each point on the visualization grid
        for i in range(resolution):
            for j in range(resolution):
                z = np.array([Z1[i, j], Z2[i, j]])
                try:
                    # Clamp point for safety before computing curvature if it's slightly outside visualization range
                    clamped_z = self._clamp_point_to_bounds(z)
                    if exact_mode:
                        curvature[i, j] = np.log(self.compute_true_gaussian_curvature(clamped_z, h=h_curvature) + 1e-8)
                    else:
                        curvature[i, j] = self.compute_gaussian_curvature(self.metric_tensor(clamped_z))
                except (ValueError, RuntimeError) as e:
                    # Catch errors for points that might be outside defined bounds or uncomputed cache
                    # print(f"Error at grid point {z}: {e}. Setting curvature to NaN.")
                    curvature[i, j] = np.nan
        
        # Create plots
        fig = plt.figure(figsize=(18, 6))
        
        # Handle color normalization robustly
        vmin = np.nanmin(curvature)
        vmax = np.nanmax(curvature)
        
        # Make sure our color scaling works
        if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax:
            print("Warning: All curvature values are NaN or uniform. Check your manifold/metric function. Using fallback color scale.")
            vmin, vmax = -1, 1  # Fallback values
        
        # Create a robust color normalization
        if vmin >= 0:
            vmin = 0
            norm = colors.Normalize(vmin=vmin, vmax=max(vmax, 0.1))
            cmap = 'Reds'
        elif vmax <= 0:
            vmax = 0
            norm = colors.Normalize(vmin=min(vmin, -0.1), vmax=vmax)
            cmap = 'Blues_r'
        else:
            # We can use a diverging colormap with zero in the middle
            maxabs = max(abs(vmin), abs(vmax))
            norm = colors.Normalize(vmin=-maxabs, vmax=maxabs)
            cmap = 'RdBu_r'
        
        # 1. 2D heatmap
        ax1 = fig.add_subplot(131)
        im = ax1.pcolormesh(Z1, Z2, curvature, cmap=cmap, norm=norm, shading='auto')
        if exact_mode:
            ax1.set_title('Manifold True Gaussian Curvature (2D View)')
        else:
            ax1.set_title('Manifold approx. Gaussian Curvature (2D View)')
        ax1.set_xlabel('z1')
        ax1.set_ylabel('z2')
        plt.colorbar(im, ax=ax1, label='Curvature')
        
        # 2. 3D surface plot
        ax2 = fig.add_subplot(132, projection='3d')
        surf = ax2.plot_surface(Z1, Z2, curvature, cmap=cmap, 
                               linewidth=0, antialiased=True, norm=norm)
        ax2.set_title('Manifold True Gaussian Curvature (3D Surface)')
        ax2.set_xlabel('z1')
        ax2.set_ylabel('z2')
        ax2.set_zlabel('Curvature')
        fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='Curvature')
        
        # 3. Contour plot
        ax3 = fig.add_subplot(133)
        contour = ax3.contourf(Z1, Z2, curvature, 20, cmap=cmap, norm=norm)
        if exact_mode:
            ax3.set_title('Manifold True Gaussian Curvature (Contour Plot)')
        else:
            ax3.set_title('Manifold approx. Gaussian Curvature (Contour Plot)')
        ax3.set_xlabel('z1')
        ax3.set_ylabel('z2')
        plt.colorbar(contour, ax=ax3, label='Curvature')
        
        # Add zero contour line if it crosses zero
        if vmin < 0 and vmax > 0:
            ax3.contour(Z1, Z2, curvature, levels=[0], colors='k', linewidths=1.5)
        
        # Plot data points if provided
        if data_points is not None:
            # Verify we have labels for the data points
            if labels is None:
                print("Warning: Data points provided without labels. Using default label 0.")
                labels = np.zeros(len(data_points), dtype=int)
                
            # Check the shape of data_points and adjust if needed
            if len(data_points.shape) > 1 and data_points.shape[1] > 2:
                print(f"Warning: Data points have {data_points.shape[1]} dimensions, using only first 2 dimensions for plotting.")
                # Extract just the first two dimensions
                plot_points = data_points[:, :2]
            else:
                plot_points = data_points
                
            # Ensure points are correctly shaped for plotting
            if len(plot_points.shape) == 1:
                # If we have a single point with just a list of coordinates
                plot_points = plot_points.reshape(1, -1)

            # Get unique labels
            unique_labels = np.unique(labels)
            
            # Create a colormap for the labels
            label_cmap = plt.cm.get_cmap('tab10', len(unique_labels))
            
            # Scatter plot for each subplot
            for ax_plot in [ax1, ax3]: # Changed from `ax` to `ax_plot` to avoid conflict with `ax2`
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    ax_plot.scatter(plot_points[mask, 0], plot_points[mask, 1], 
                              c=[label_cmap(i)], edgecolors='k', s=50, 
                              label=f'Class {label}', zorder=10)
                # Use automatic legend placement with small font and maximum 10 items per column
                if len(unique_labels) > 10:
                    ncol = max(1, len(unique_labels) // 10)
                    ax_plot.legend(loc='best', fontsize='xx-small', ncol=ncol)
                else:
                    ax_plot.legend(loc='best', fontsize='small')
            
            # For 3D plot, we need to compute the curvature at each data point
            # and use that for z-coordinate
            data_curvatures = np.zeros(len(plot_points))
            for i, point in enumerate(plot_points):
                try:
                    # Clamp data point for safety before computing curvature
                    clamped_point = self._clamp_point_to_bounds(point)
                    if exact_mode:
                        data_curvatures[i] = self.compute_true_gaussian_curvature(clamped_point, h=h_curvature)
                    else:
                        data_curvatures[i] = self.compute_gaussian_curvature(self.metric_tensor(clamped_z))*1.05
                except (ValueError, RuntimeError) as e:
                    print(f"Error computing curvature for data point {i} ({point}): {e}. Setting to NaN.")
                    data_curvatures[i] = np.nan
            
            # Plot on 3D surface
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax2.scatter(plot_points[mask, 0], plot_points[mask, 1], 
                           data_curvatures[mask], c=[label_cmap(i)], 
                           edgecolors='k', s=50, label=f'Class {label}', zorder=10)
            # Use automatic legend placement with small font and maximum 10 items per column
            if len(unique_labels) > 10:
                ncol = max(1, len(unique_labels) // 10)
                ax2.legend(loc='best', fontsize='xx-small', ncol=ncol)
            else:
                ax2.legend(loc='best', fontsize='small')
        
        plt.tight_layout()
        plt.show()
        
        return curvature, Z1, Z2

    def plot_geodesics(self, start_point: np.ndarray, end_point: np.ndarray, curvature: np.ndarray, Z1: np.ndarray, Z2: np.ndarray, num_points_geodesic: int = 50):
        """
        Plot geodesics between points to show path avoidance around high curvature regions.
        This function requires the BoundedManifold instance to have caching enabled
        and its cache populated using `compute_full_grid_metric_tensor()`.

        Args:
            start_point (np.ndarray): Starting point.
            end_point (np.ndarray): Ending point.
            curvature (np.ndarray): Array of curvature values (e.g., from `visualize_manifold_curvature`).
            Z1 (np.ndarray): Meshgrid array for x-coordinates (e.g., from `visualize_manifold_curvature`).
            Z2 (np.ndarray): Meshgrid array for y-coordinates (e.g., from `visualize_manifold_curvature`).
            num_points_geodesic (int): Number of points to compute for the geodesic path.
        """
        if not self.cache_enabled:
            raise ValueError("Geodesic plotting requires caching to be enabled for the BoundedManifold instance.")
        if self._n_dimensions != 2:
            raise ValueError("Geodesic plotting is currently supported only for 2D manifolds.")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Handle color normalization robustly
        vmin = np.nanmin(curvature)
        vmax = np.nanmax(curvature)
        
        # Create a robust color normalization
        if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax:
            print("Warning: All curvature values are NaN or uniform. Using fallback color scale for geodesic plot.")
            vmin, vmax = -1, 1  # Fallback values
        
        if vmin >= 0:
            vmin = 0
            norm = colors.Normalize(vmin=vmin, vmax=max(vmax, 0.1))
            cmap = 'Reds'
        elif vmax <= 0:
            vmax = 0
            norm = colors.Normalize(vmin=min(vmin, -0.1), vmax=vmax)
            cmap = 'Blues_r'
        else:
            maxabs = max(abs(vmin), abs(vmax))
            norm = colors.Normalize(vmin=-maxabs, vmax=maxabs)
            cmap = 'RdBu_r'
        
        # Plot curvature heatmap
        im = ax.pcolormesh(Z1, Z2, curvature, cmap=cmap, norm=norm, shading='auto', alpha=0.7)
        
        # Plot straight line (Euclidean path)
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                'k--', linewidth=1.5, label='Euclidean Path')
        
        # Compute and plot true geodesic using the BoundedManifold instance
        geodesic_path = self.compute_geodesic(start_point, end_point, num_points=num_points_geodesic)
        ax.plot(geodesic_path[:, 0], geodesic_path[:, 1], 'r-', linewidth=2, label='Geodesic Path')
        
        # Mark start and end points
        ax.plot(start_point[0], start_point[1], 'go', markersize=8, label='Start')
        ax.plot(end_point[0], end_point[1], 'bo', markersize=8, label='End')
        
        ax.set_title('Geodesic vs Euclidean Path (True Gaussian Curvature Background)')
        ax.set_xlabel('z1')
        ax.set_ylabel('z2')
        plt.colorbar(im, label='True Gaussian Curvature')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def linear_interpolation_distance(self, u: np.ndarray, v: np.ndarray, num_points: int = 100) -> float:
        """
        Compute distance using linear interpolation of the path between u and v: ∫₀¹ √((u-v)ᵀg(x(t))(u-v))dt
        where x(t) = (1-t)u + t*v (linear path in coordinate space).

        Args:
            u (np.ndarray): The starting point.
            v (np.ndarray): The ending point.
            num_points (int): Number of integration points along the linear path.

        Returns:
            float: The approximate distance along the linearly interpolated path.
        """
        if len(u) != self._n_dimensions or len(v) != self._n_dimensions:
            raise ValueError("Point dimensions must match manifold dimension.")
        if not self._is_within_bounds(u) or not self._is_within_bounds(v):
            print("Warning: One or both points are outside manifold bounds. Results may be inaccurate or fall back to Euclidean.")

        def integrand(t):
            # Linearly interpolated point
            x_t = (1 - t) * u + t * v
            # Clamp x_t for metric tensor evaluation, as intermediate points might slightly go out of bounds
            clamped_x_t = self._clamp_point_to_bounds(x_t)
            
            diff = v - u # This is the "dx/dt" in the integral for a linear path
            
            try:
                # Use self.metric_tensor to leverage caching and bounds checking
                g = self.metric_tensor(clamped_x_t)
                # The integrand is sqrt( (dx/dt)^T * G * (dx/dt) )
                return np.sqrt(diff.T @ g @ diff)
            except (ValueError, RuntimeError):
                # Fallback to Euclidean distance if metric tensor cannot be computed (e.g., truly out of bounds)
                # This should be np.linalg.norm(diff) not (u-v)
                return np.linalg.norm(diff) # Integrate Euclidean length of the segment

        t_vals = np.linspace(0, 1, num_points)
        # Evaluate integrand at each t_val
        integrand_values = [integrand(t) for t in t_vals]
        # Use trapezoidal rule for numerical integration
        integral = np.trapz(integrand_values, x=t_vals)
        return integral
    
    def midpoint_approximation(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Distance approximation using metric at midpoint: √((v-u)ᵀg((u+v)/2)(v-u))
        This is a simple straight-line distance approximation using the metric tensor
        evaluated only at the midpoint.

        Args:
            u (np.ndarray): The starting point.
            v (np.ndarray): The ending point.

        Returns:
            float: The approximate distance.
        """
        if len(u) != self._n_dimensions or len(v) != self._n_dimensions:
            raise ValueError("Point dimensions must match manifold dimension.")
        if not self._is_within_bounds(u) or not self._is_within_bounds(v):
            print("Warning: One or both points are outside manifold bounds. Results may be inaccurate or fall back to Euclidean.")

        midpoint = (u + v) / 2
        clamped_midpoint = self._clamp_point_to_bounds(midpoint) # Clamp for metric evaluation
        diff = v - u
        
        try:
            # Use self.metric_tensor to leverage caching and bounds checking
            g = self.metric_tensor(clamped_midpoint)
            # Ensure the result of the quadratic form is non-negative before sqrt
            quadratic_form_result = diff.T @ g @ diff
            return np.sqrt(max(0, quadratic_form_result))
        except (ValueError, RuntimeError):
            # Fallback to Euclidean distance if metric tensor cannot be computed
            print(f"Warning: Cannot compute metric at midpoint {midpoint}. Falling back to Euclidean distance.")
            return np.linalg.norm(diff)
    
    def weighted_midpoint_approximation(self, u: np.ndarray, v: np.ndarray, weights: List[float] = None) -> float:
        """
        Distance approximation using weighted average of squared differential elements at multiple points.
        This is an approximation based on combining results from several points.
        Default weights correspond to Simpson's rule for the integral of f(t) = sqrt( (u-v)^T g(x(t)) (u-v) )
        over [0,1], where x(t) is linear.

        Args:
            u (np.ndarray): The starting point.
            v (np.ndarray): The ending point.
            weights (List[float], optional): Weights for each point. Defaults to Simpson's rule weights [1/6, 4/6, 1/6].

        Returns:
            float: The approximate distance.
        """
        if len(u) != self._n_dimensions or len(v) != self._n_dimensions:
            raise ValueError("Point dimensions must match manifold dimension.")
        if not self.cache_enabled:
            # This method heavily benefits from caching, especially for multiple metric calls.
            print("Warning: Caching is not enabled for this manifold instance. Weighted midpoint approximation will be slower.")
        
        # Check initial point bounds for a warning
        if not self._is_within_bounds(u) or not self._is_within_bounds(v):
            print("Warning: One or both points are outside manifold bounds. Results may be inaccurate or fall back to Euclidean.")

        if weights is None:
            weights = [1/6, 4/6, 1/6]  # Simpson's rule weights for 3 points
        
        # Points along the linear path (t=0, t=0.5, t=1)
        points = [u, (u + v) / 2, v]
        
        if len(points) != len(weights):
            raise ValueError("Number of points must match number of weights provided.")

        total_weighted_sum_of_lengths = 0.0
        diff = v - u # The vector difference (dx/dt)

        for i, (point, weight) in enumerate(zip(points, weights)):
            clamped_point = self._clamp_point_to_bounds(point) # Clamp for metric evaluation
            
            try:
                # Use self.metric_tensor to leverage caching and bounds checking
                g = self.metric_tensor(clamped_point)
                quadratic_form_result = diff.T @ g @ diff
                # Take sqrt for the length contribution, ensure non-negative
                length_contribution = np.sqrt(max(0, quadratic_form_result))
                total_weighted_sum_of_lengths += weight * length_contribution
            except (ValueError, RuntimeError):
                # Fallback to Euclidean length contribution if metric cannot be computed
                print(f"Warning: Cannot compute metric at point {point}. Falling back to Euclidean length contribution for this segment.")
                total_weighted_sum_of_lengths += weight * np.linalg.norm(diff)
        
        return total_weighted_sum_of_lengths
    
    def endpoint_average_approximation(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Distance approximation using average of metrics at endpoints: √((v-u)ᵀ((g(u)+g(v))/2)(v-u))
        This approximates the integral of the metric along a straight line by averaging the metric
        at the two endpoints.

        Args:
            u (np.ndarray): The starting point.
            v (np.ndarray): The ending point.

        Returns:
            float: The approximate distance.
        """
        if len(u) != self._n_dimensions or len(v) != self._n_dimensions:
            raise ValueError("Point dimensions must match manifold dimension.")
        if not self._is_within_bounds(u) or not self._is_within_bounds(v):
            print("Warning: One or both points are outside manifold bounds. Results may be inaccurate or fall back to Euclidean.")

        diff = v - u
        try:
            # Use self.metric_tensor to leverage caching and bounds checking
            g_u = self.metric_tensor(self._clamp_point_to_bounds(u))
            g_v = self.metric_tensor(self._clamp_point_to_bounds(v))
            g_avg = (g_u + g_v) / 2
            
            # Ensure the result of the quadratic form is non-negative before sqrt
            quadratic_form_result = diff.T @ g_avg @ diff
            return np.sqrt(max(0, quadratic_form_result))
        except (ValueError, RuntimeError):
            # Fallback to Euclidean distance if metric tensor cannot be computed
            print(f"Warning: Cannot compute metric at endpoints {u} or {v}. Falling back to Euclidean distance.")
            return np.linalg.norm(diff)
    
    def euclidean_distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Standard Euclidean distance for comparison.

        Args:
            u (np.ndarray): The first point.
            v (np.ndarray): The second point.

        Returns:
            float: The Euclidean distance between u and v.
        """
        if len(u) != self._n_dimensions or len(v) != self._n_dimensions:
            raise ValueError("Point dimensions must match manifold dimension.")
            
        return np.linalg.norm(u - v)
