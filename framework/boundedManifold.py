import torch
from tqdm import tqdm
from typing import List, Tuple, Union, Callable
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from torchdiffeq import odeint

from framework.GridGraph import GridGraph

class BoundedManifold:
    """
    Manifold class that wraps the metric tensor and allows for caching.
    This version uses PyTorch tensors to be compatible with autograd pipelines.
    """
    def __init__(self, metric_tensor_func: callable, bounds : torch.tensor = None, 
                 grid_size : int = 50, device : str = None):
        """
        Initializes the BoundedManifold.

        Args:
            metric_tensor_func (callable): A callable that takes a torch.Tensor (point)
                                           and returns the metric tensor at that point.
                                           Expected signature: metric_tensor_func(point: torch.Tensor) -> torch.Tensor
            bounds (torch.Tensor): A 2D torch.Tensor defining the bounds for each dimension.
                                   Shape should be (n_dimensions, 2), where each row is [min_val, max_val].
            grid_size (int): The number of grid points per dimension.
            device (str): The device to run the computations on ('cpu' or 'cuda').
        """
        self._metric_tensor_func = metric_tensor_func
        self.device = torch.device(device) if device is not None else torch.device('cpu')

        # Grid config
        self.grid_size = grid_size
        if bounds is None:
            b = torch.tensor([[-1.5, 1.5], [-1.5, 1.5]], dtype=torch.float32)
        else:
            if not isinstance(bounds, torch.Tensor) or bounds.ndim != 2 or bounds.shape[1] != 2:
                raise ValueError("Bounds must be a 2D torch.Tensor of shape (n_dimensions, 2).")
            b = bounds.clone().float()
        self.bounds = b.to(self.device)
        self.n_dims = self.bounds.size(0)

        # Precompute steps and strides for flattening
        self.steps = (self.bounds[:,1] - self.bounds[:,0]) / self.grid_size
        self.inv_steps = 1.0 / self.steps
        # strides for a (grid_size+1)^n indexing
        sizes = [self.grid_size + 1] * self.n_dims
        self._strides = [1]
        for s in reversed(sizes[:-1]):
            self._strides.insert(0, self._strides[0] * s)
        self._strides = torch.tensor(self._strides, dtype=torch.long, device=self.device)
        self._total_grid = torch.prod(torch.tensor(sizes, device=self.device)).item()

        # Cache storage (flat)
        self.cache_flat = None   # shape [total_grid, *metric_shape]
        self.mask_flat = None    # shape [total_grid], bool
        self.metric_shape = None
        self.current_hash = None

        # Stats
        self.hits = self.misses = self.total_requests = 0
        self._full_computed = False

        self.grid_graph = None

    def get_bounds(self) -> torch.Tensor:
        return self.bounds
    
    def get_dimension(self) -> int:
        return self.n_dims
    
    def _is_within_bounds(self, points: torch.Tensor) -> Union[bool, torch.Tensor]:
        """
        Checks if point(s) are within the defined manifold bounds.

        Args:
            points (torch.Tensor): 
                - If shape (D,), returns a single bool.
                - If shape (B, D), returns a (B,) bool tensor.

        Returns:
            bool or torch.Tensor:
                - For a single point, a Python bool.
                - For a batch, a Boolean tensor of length B.
        """
        if not isinstance(points, torch.Tensor):
            raise ValueError("points must be a torch.Tensor.")

        epsilon = 1e-9
        # bounds is (D, 2): [:,0]=minima, [:,1]=maxima
        mins = self.bounds[:, 0].to(points.device)  # (D,)
        maxs = self.bounds[:, 1].to(points.device)  # (D,)

        if points.dim() == 1:
            if points.shape[0] != self.n_dims:
                raise ValueError(
                    f"Point dim ({points.shape[0]}) ≠ manifold dim ({self.n_dims})."
                )
            in_low  = points >= (mins - epsilon)
            in_high = points <= (maxs + epsilon)
            return bool((in_low & in_high).all())

        elif points.dim() == 2:
            B, D = points.shape
            if D != self.n_dims:
                raise ValueError(
                    f"Points second dim ({D}) ≠ manifold dim ({self.n_dims})."
                )
            # (B, D) mask
            in_low  = points >= (mins - epsilon)
            in_high = points <= (maxs + epsilon)
            # (B,) mask: each row must satisfy all dims
            return (in_low & in_high).all(dim=1)

        else:
            raise ValueError(
                "points must be 1D (D,) or 2D (B, D) torch.Tensor."
            )

    def _clamp_point_to_bounds(self, point: torch.Tensor) -> torch.Tensor:
        """
        Clamps a given point to the defined manifold bounds.
        """
        # torch.clamp requires min and max to be broadcastable with input.
        # point shape: (n_dim), bounds shape: (n_dim, 2)
        # We need to compare point[i] with bounds[i,0] and bounds[i,1]
        return torch.max(self.bounds[:, 0], torch.min(point, self.bounds[:, 1]))

    def _ravel_index(self, points: torch.Tensor):
        """
        Vectorized ravel index that preserves retro‐compatibility:
        - If `points` is shape (D,), returns a Python int.
        - If `points` is shape (B, D), returns a torch.LongTensor of shape (B,).
        """
        # Move to correct device/dtype
        pts = points.to(self.device).float()  # (..., D)

        # Compute relative position in grid coordinates
        # assumes self.bounds[:,0] is (D,), self.inv_steps is (D,)
        rel = (pts - self.bounds[:, 0]) * self.inv_steps  # (..., D)

        # Round to nearest integer grid index and clamp to [0, grid_size]
        idx = torch.clamp(torch.round(rel), 0, self.grid_size).long()  # (..., D)

        # Compute flat index: dot with strides (D,) → sum over last axis → shape (...)
        flat = (idx * self._strides.to(idx.device)).sum(dim=-1)          # (...)

        # If it’s a single point, return a Python int (exactly as before)
        if flat.dim() == 0:
            return int(flat.item())

        # Otherwise return the batched tensor
        return flat

    def metric_tensor(self,
                    points: torch.Tensor,
                    force: bool = False
    ) -> torch.Tensor:
        """
        If `points` is (D,), run the original single‐point logic unchanged.
        If `points` is (B, D), do the batched logic for speed.
        """
        # --- Single‐point fallback ------------------------------------------------
        if points.dim() == 1:
            # exactly your original code, unchanged
            point = points
            if not force and not self._is_within_bounds(point):
                raise ValueError(f"Point {point} is outside the defined manifold bounds: {self.get_bounds()}")
            self.total_requests += 1

            flat_idx = self._ravel_index(point)
            if self.cache_flat is not None and self.mask_flat[flat_idx]:
                self.hits += 1
                return self.cache_flat[flat_idx].clone()

            self.misses += 1
            g = self._metric_tensor_func(point.to(self.device))

            # Lazy init
            if self.cache_flat is None:
                self.metric_shape = tuple(g.shape)
                shape = [self._total_grid] + list(self.metric_shape)
                self.cache_flat = torch.empty(*shape, device=self.device)
                self.mask_flat = torch.zeros(self._total_grid, dtype=torch.bool, device=self.device)

            self.cache_flat[flat_idx] = g
            self.mask_flat[flat_idx] = True
            return g.clone()

        # --- Batched path --------------------------------------------------------
        # Normalize and batchify
        pts = points.float().to(self.device)
        B, D = pts.shape

        if not force:
            # # cheap Python check per‐row
            # for pt in pts:
            #     if not self._is_within_bounds(pt):
            #         raise ValueError(f"Point {pt} is outside the defined manifold bounds: {self.get_bounds()}")
            in_bounds = self._is_within_bounds(pts)     # now a single kernel!
            if not in_bounds.all():
                bad = pts[~in_bounds]
                raise ValueError(f"Out-of-bounds points: {bad}")

        self.total_requests += B

        # Compute flat indices
        #flat_idxs = torch.tensor(self._ravel_index(pts),
        #                        dtype=torch.long,
        #                        device=self.device)
        flat_idxs = self._ravel_index(pts).detach().to(self.device).long()

        # Lazy init cache arrays
        if self.cache_flat is None:
            self.metric_shape = (D, D)
            cache_shape = (self._total_grid, D, D)
            self.cache_flat = torch.empty(*cache_shape, device=self.device)
            self.mask_flat = torch.zeros(self._total_grid, dtype=torch.bool, device=self.device)

        hits_mask = self.mask_flat[flat_idxs]   # (B,)
        miss_mask = ~hits_mask

        out = torch.empty((B, D, D), device=self.device, dtype=torch.float)

        # Serve cache hits
        if hits_mask.any():
            hit_idxs = flat_idxs[hits_mask]
            out[hits_mask] = self.cache_flat[hit_idxs].clone()
            self.hits += hit_idxs.numel()

        # Compute all misses in one go
        if miss_mask.any():
            miss_pts = pts[miss_mask]               # (M, D)
            Gmiss = self._metric_tensor_func(miss_pts.to(self.device))  # (M, D, D)
            miss_idxs = flat_idxs[miss_mask]
            self.cache_flat[miss_idxs] = Gmiss
            self.mask_flat[miss_idxs] = True
            out[miss_mask] = Gmiss.clone()
            self.misses += miss_idxs.numel()

        return out  # shape (B, D, D)

    def clear(self):
        """Reset cache and stats"""
        self.cache_flat = None
        self.mask_flat = None
        self.metric_shape = None
        self.current_hash = None
        self.hits = self.misses = self.total_requests = 0
        self._full_computed = False

    def get_stats(self):
        hit_rate = self.hits / max(self.total_requests, 1)
        cached = int(self.mask_flat.sum().item()) if self.mask_flat is not None else 0
        return {
            'hit_rate': hit_rate,
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': self.total_requests,
            'cached_points': cached,
            'total_grid_points': self._total_grid
        }

    def compute_full_grid_metric_tensor(self, force=False, batch_size=None):
        if not force and self._full_computed:
            print("Full grid already computed for current epoch.")
            return

        # Generate all grid points
        coords = [torch.linspace(self.bounds[d,0], self.bounds[d,1], self.grid_size+1, device=self.device)
                  for d in range(self.n_dims)]
        meshes = torch.meshgrid(*coords, indexing='ij')
        flat_pts = torch.stack([m.flatten() for m in meshes], dim=1)
        total = flat_pts.size(0)
        print(f"Computing metric tensors for {total} grid points...")

        # If self._metric_tensor_func supports batching, do so
        if batch_size and hasattr(self._metric_tensor_func, '__call__'):
            for start in tqdm(range(0, total, batch_size), desc="Batched grid compute"):
                end = min(start + batch_size, total)
                batch = flat_pts[start:end]
                gs = self._metric_tensor_func(batch)  # expect [B, *metric_shape]
                for i, g in enumerate(gs):
                    idx = self._ravel_index(batch[i])
                    self.cache_flat[idx] = g
                    self.mask_flat[idx] = True
        else:
            for p in tqdm(flat_pts, desc="Grid compute"):
                idx = self._ravel_index(p)
                if self.mask_flat is not None and self.mask_flat[idx] and not force:
                    continue
                g = self._metric_tensor_func(p)
                if self.cache_flat is None:
                    self.metric_shape = tuple(g.shape)
                    shape = [self._total_grid] + list(self.metric_shape)
                    self.cache_flat = torch.empty(*shape, device=self.device)
                    self.mask_flat = torch.zeros(self._total_grid, dtype=torch.bool, device=self.device)
                self.cache_flat[idx] = g
                self.mask_flat[idx] = True

        self._full_computed = True
        print(f"Completed. Cached {int(self.mask_flat.sum().item())} / {total} points.")

    def create_riemannian_distance_matrix(self, data_points: torch.Tensor, 
                                          distance_calculator: Callable = None, 
                                          batch_size: int = 8,
                                          **kwargs) -> torch.Tensor:
        """
        Creates a pairwise Riemannian distance matrix for the given data points.
        """
        n_points = data_points.shape[0]
        dist_matrix = torch.zeros((n_points, n_points), device=self.device)
        total_calculations = n_points * (n_points - 1) // 2
        
        actual_distance_calculator = distance_calculator if distance_calculator is not None else self.exact_geodesic_distance
        
        if n_points > 0 and data_points.shape[1] != self.get_dimension():
            raise ValueError(f"Dimension of data_points ({data_points.shape[1]}) does not match manifold dimension ({self.get_dimension()}).")

        if batch_size == 0:
            print(f"Calculating {total_calculations} pairwise Riemannian distances using {actual_distance_calculator.__name__} (single mode)...")
            with tqdm(total=total_calculations, desc="Calculating distances") as pbar:
                for i in range(n_points):
                    for j in range(i + 1, n_points):
                        u_point, v_point = data_points[i], data_points[j]
                        if "manifold" in actual_distance_calculator.__code__.co_varnames:
                            dist = actual_distance_calculator(self, u_point, v_point, **kwargs)
                        else:
                            dist = actual_distance_calculator(u_point, v_point, **kwargs)
                        dist_matrix[i, j] = dist
                        dist_matrix[j, i] = dist
                        pbar.update(1)
        else:
            print(f"Calculating {total_calculations} pairwise Riemannian distances using {actual_distance_calculator.__name__} (batch mode)...")
            num_nodes = data_points.shape[0]
            row_indices, col_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
            total_pairs = len(row_indices)

            # Process in mini-batches
            for i in tqdm(range(0, total_pairs, batch_size), desc="Computing distances in batches"):
                batch_start = i
                batch_end = min(i + batch_size, total_pairs)

                # Get the current batch of indices
                current_row_indices = row_indices[batch_start:batch_end]
                current_col_indices = col_indices[batch_start:batch_end]

                # Extract the corresponding node embeddings for these pairs
                u_batch = data_points[current_row_indices]
                v_batch = data_points[current_col_indices]

                # Compute distances for this mini-batch
                if "manifold" in actual_distance_calculator.__code__.co_varnames:
                    current_batch_distances = actual_distance_calculator(self, u_batch, v_batch, **kwargs)
                else:
                    current_batch_distances = actual_distance_calculator(u_batch, v_batch, **kwargs)

                # Populate the distance matrix with the results of this batch
                dist_matrix[current_row_indices, current_col_indices] = current_batch_distances
                dist_matrix[current_col_indices, current_row_indices] = current_batch_distances # Symmetric

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
    
    @staticmethod
    def _plot_manifold_grid(values, Z1_np, Z2_np, latent_points=None, labels=None, name="Gaussian Curvature",
                            graphs=2):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()    
           
        with plt.rc_context({'font.family': 'sans-serif', 
                     'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans']}):
            fig = plt.figure(figsize=(18 if graphs > 2 else 12, 6), dpi=300)
            vmin, vmax = np.nanmin(values), np.nanmax(values)
            if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax: vmin, vmax = -1, 1
            if vmin < 0 and vmax > 0:
                absmax = np.max([np.abs(vmax), np.abs(vmin)])
                norm = colors.TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)
                cmap = 'RdBu_r'
            elif vmin < 0 and vmax <= 0:
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
                cmap = 'Blues_r'
            else:
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
                cmap = 'Reds'

            if graphs > 2:
                ax1 = fig.add_subplot(131)
                im = ax1.pcolormesh(Z1_np, Z2_np, values, cmap=cmap, norm=norm, shading='auto')
                ax1.set_title(f'Manifold {name} (2D View)')
                plt.colorbar(im, ax=ax1, label=f'{name}')

            ax2 = fig.add_subplot(132 if graphs > 2 else 121, projection='3d')
            ax2.plot_surface(Z1_np, Z2_np, values, cmap=cmap, norm=norm)
            ax2.set_title(f'Manifold {name} (3D Surface)')

            ax3 = fig.add_subplot(133 if graphs > 2 else 122)
            contour = ax3.contourf(Z1_np, Z2_np, values, 20, cmap=cmap, norm=norm)
            ax3.set_title(f'Manifold {name} (Contour)')
            plt.colorbar(contour, ax=ax3, label=f'{name}')

            if latent_points is not None:
                points_np = latent_points.detach().cpu().numpy()                
                labels_np = labels.detach().cpu().numpy() if labels is not None else np.zeros(len(points_np))
                if graphs > 2:
                    ax1.scatter(points_np[:, 0], points_np[:, 1], c=labels_np, cmap='viridis', edgecolors='k')
                ax3.scatter(points_np[:, 0], points_np[:, 1], c=labels_np, cmap='viridis', edgecolors='k')

            plt.tight_layout()
            plt.show()

    def plot_on_manifold_grid(self, function: Callable[[torch.Tensor], float], name: str,
            data_points: Union[torch.Tensor, None] = None, labels: Union[torch.Tensor, None] = None,
            z_range: Union[Tuple[float, float], torch.Tensor, None] = None, 
            resolution: int = 30, require_full_grid: bool=True, plot: bool=True):
        
        if self.get_dimension() != 2:
            raise ValueError("Visualization is supported only for 2D manifolds.")
        
        if not self._full_computed and require_full_grid:
            print("Full grid is not computed, we trigger compute_full_grid_metric_tensor...")
            self.compute_full_grid_metric_tensor()

        with torch.no_grad():
            if z_range is None:
                bounds_np = self.get_bounds().cpu().numpy()
                plot_z1 = np.linspace(bounds_np[0, 0], bounds_np[0, 1], resolution)
                plot_z2 = np.linspace(bounds_np[1, 0], bounds_np[1, 1], resolution)

            Z1_np, Z2_np = np.meshgrid(plot_z1, plot_z2)
            Z1, Z2 = torch.from_numpy(Z1_np).to(self.device), torch.from_numpy(Z2_np).to(self.device)
            
            values = torch.zeros((resolution, resolution), device=self.device)
            
            for i in range(resolution):
                for j in range(resolution):
                    z = torch.stack([Z1[i, j], Z2[i, j]])
                    try:
                        values[i, j] = function(self._clamp_point_to_bounds(z).float())
                    except (ValueError, RuntimeError) as e:
                        print(f"Error computing point {z}: {e}. Setting to NaN.")
                        values[i, j] = torch.nan
            
            # Convert to numpy for plotting
            curvature_np = values.cpu().numpy()
            if plot:
                self._plot_manifold_grid(curvature_np, Z1_np, Z2_np, latent_points=data_points, labels=labels, name=name)
            return curvature_np
        
    def visualize_manifold_curvature(self, z_range: Union[Tuple[float, float], torch.Tensor, None] = None, resolution: int = 30,
                                     data_points: Union[torch.Tensor, None] = None, labels: Union[torch.Tensor, None] = None,
                                     h_curvature: float = 1e-3, exact_mode: bool = False,
                                     return_data: bool = False, plot: bool = True):
        
        def curvature_func(z: torch.Tensor) -> float:
            return torch.log(self.compute_true_gaussian_curvature(z, h=h_curvature) +1e-8) if exact_mode else self.compute_gaussian_curvature(self.metric_tensor(z))

        r = self.plot_on_manifold_grid(curvature_func, "Gaussian Curvature",
                                          data_points=data_points, labels=labels, z_range=z_range,
                                          resolution=resolution, require_full_grid=True, plot=plot)
        if return_data:
            return r

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
    
    def linear_interpolation_distance(self,
            u: torch.Tensor,
            v: torch.Tensor,
            num_points: int = 50,
        ) -> torch.Tensor:
        """
        Per-pair linear interpolation distance fully in float precision,
        now batched over inputs of shape (D,) or (B, D).
        Returns a scalar if inputs are (D,), or a tensor of shape (B,) if inputs are (B, D).
        """
        # ensure float
        u = u.float()
        v = v.float()

        # make sure we have a batch dimension
        single = (u.dim() == 1)
        if single:
            # from (D,) to (1, D)
            u = u.unsqueeze(0)
            v = v.unsqueeze(0)

        # now u, v are (B, D)
        B, D = u.shape

        # time samples
        t_vals = torch.linspace(0.0, 1.0, num_points, device=self.device, dtype=torch.float)
        T = num_points
        # shape (T, 1, 1) for broadcasting over (B, D)
        t = t_vals.view(T, 1, 1)

        # build all interpolation points: shape (T, B, D)
        # u.unsqueeze(0) is (1, B, D), same for v
        X = (1.0 - t) * u.unsqueeze(0) + t * v.unsqueeze(0)
        # clamp them, but _clamp_point_to_bounds wants (..., D)
        X_flat = X.view(-1, D)  # (T*B, D)
        Xc_flat = self._clamp_point_to_bounds(X_flat)
        Xc = Xc_flat.view(T, B, D)

        # compute metric tensor at each of the T*B points in one go:
        # self.metric_tensor(Xc_flat, True) -> (T*B, D, D)
        G_flat = self.metric_tensor(Xc_flat, True)
        G = G_flat.view(T, B, D, D)  # (T, B, D, D)

        # difference v - u: (B, D)
        diff = (v - u)
        # replicate over T: (T, B, D)
        diff_rep = diff.unsqueeze(0).expand(T, B, D)

        # squared speed at each (t, b)
        # seg_sq[t,b] = diff_rep[t,b] @ G[t,b] @ diff_rep[t,b]
        seg_sq = torch.einsum("tbi,tbij,tbj->tb", diff_rep, G, diff_rep)

        # speed = sqrt(seg_sq)
        integrand = torch.sqrt(torch.relu(seg_sq) + 1e-12)  # (T, B)

        # trapezoidal integration over t: returns shape (B,)
        dists = torch.trapz(integrand, x=t_vals, dim=0)  # (B,)

        # if original inputs were single points, return scalar
        if single:
            return dists.squeeze(0)
        return dists
    
    def get_grid_as_graph(self):
        if not self._full_computed:
            print("Full grid is not computed, we trigger compute_full_grid_metric_tensor...")
            self.compute_full_grid_metric_tensor()

        if self.grid_graph is None:
            self.grid_graph = GridGraph(self, include_diagonal=True)
            
        return self.grid_graph
    
    def _numerical_derivative_metric_tensor(self, point: torch.Tensor, dim_idx: int, h: float) -> torch.Tensor:
        """
        Numerically computes the partial derivative ∂_{dim_idx} g_{ij}(point) using central difference.
        Returns a tensor of shape (n, n) on the same device/dtype as `point`.
        """
        device = point.device
        dtype = point.dtype
        h_t = torch.tensor(h, device=device, dtype=dtype)

        h_vec = torch.zeros_like(point)
        h_vec[dim_idx] = h_t

        point_plus = self._clamp_point_to_bounds(point + h_vec)
        point_minus = self._clamp_point_to_bounds(point - h_vec)

        g_plus = self.metric_tensor(point_plus)
        g_minus = self.metric_tensor(point_minus)

        return (g_plus - g_minus) / (2.0 * h_t)

    def _numerical_second_derivative_metric_tensor_component(self, point: torch.Tensor,
                                                          g_row: int, g_col: int,
                                                          diff_dim_idx1: int, diff_dim_idx2: int,
                                                          h: float) -> float:
        """
        Numerically computes the second partial derivative of a single metric component
        ∂_{diff_dim_idx1} ∂_{diff_dim_idx2} g_{g_row,g_col}(point).
        Returns a scalar (torch scalar on same device/dtype).
        """
        device = point.device
        dtype = point.dtype
        h_t = torch.tensor(h, device=device, dtype=dtype)

        if diff_dim_idx1 == diff_dim_idx2:
            # second derivative along same axis: central second difference
            h_vec = torch.zeros_like(point); h_vec[diff_dim_idx1] = h_t
            v_plus = self.metric_tensor(self._clamp_point_to_bounds(point + h_vec))[g_row, g_col]
            v_0    = self.metric_tensor(self._clamp_point_to_bounds(point))[g_row, g_col]
            v_minus= self.metric_tensor(self._clamp_point_to_bounds(point - h_vec))[g_row, g_col]
            return (v_plus - 2.0 * v_0 + v_minus) / (h_t ** 2)
        else:
            # mixed second derivative ∂_i ∂_j f = (f(x+hi+hj) - f(x+hi-hj) - f(x-hi+hj) + f(x-hi-hj)) / (4 h^2)
            h_vec1 = torch.zeros_like(point); h_vec1[diff_dim_idx1] = h_t
            h_vec2 = torch.zeros_like(point); h_vec2[diff_dim_idx2] = h_t

            p_pp = self._clamp_point_to_bounds(point + h_vec1 + h_vec2)
            p_pm = self._clamp_point_to_bounds(point + h_vec1 - h_vec2)
            p_mp = self._clamp_point_to_bounds(point - h_vec1 + h_vec2)
            p_mm = self._clamp_point_to_bounds(point - h_vec1 - h_vec2)

            v_pp = self.metric_tensor(p_pp)[g_row, g_col]
            v_pm = self.metric_tensor(p_pm)[g_row, g_col]
            v_mp = self.metric_tensor(p_mp)[g_row, g_col]
            v_mm = self.metric_tensor(p_mm)[g_row, g_col]

            return (v_pp - v_pm - v_mp + v_mm) / (4.0 * (h_t ** 2))
        
    def compute_christoffel(self, point: torch.Tensor, h: float = 1e-6) -> torch.Tensor:
        """
        Compute Christoffel symbols Γ^i_{jk} at `point` and return them with index order (i, j, k),
        so that acceleration a^i = - Γ^i_{jk} v^j v^k can be computed with
            torch.einsum('ijk,jk->i', Gamma, v_outer)
        """
        if len(point) != self.n_dims:
            raise ValueError(f"Point dimension ({len(point)}) does not match manifold dimension ({self.n_dims}).")

        device = point.device
        dtype = point.dtype

        g = self.metric_tensor(point).to(device=device, dtype=dtype)   # (n, n)
        n = self.n_dims
        g_inv = torch.linalg.inv(g)

        # dg[l, i, j] = ∂_l g_{ij}
        dg = torch.stack([self._numerical_derivative_metric_tensor(point, l, h) for l in range(n)])  # shape (n, n, n)

        # Build Christoffel symbols of the first kind: Γ_{i j l} = 1/2 ( ∂_i g_{j l} + ∂_j g_{i l} - ∂_l g_{i j} )
        # We'll construct it with indices (i, j, l) for clarity.
        christoffel_first_kind = torch.zeros((n, n, n), device=device, dtype=dtype)  # (i, j, l)
        for i in range(n):
            for j in range(n):
                for l in range(n):
                    # dg[deriv, a, b] = ∂_{deriv} g_{ab}
                    term = 0.5 * (dg[i, j, l] + dg[j, i, l] - dg[l, i, j])
                    christoffel_first_kind[i, j, l] = term

        # Now raise the last index using g^{i l} to obtain Gamma^k_{ij}.
        # Γ^k_{ij} = g^{k l} Γ_{i j l}
        # We'll compute Gamma_kij first (k, i, j), then permute to (i, j, k).
        Gamma_kij = torch.einsum('kl,ijl->kij', g_inv, christoffel_first_kind)  # (k, i, j)
        Gamma_ijk = Gamma_kij.permute(1, 2, 0).contiguous()  # (i, j, k) where first axis is the upper index i

        return Gamma_ijk


    def compute_true_gaussian_curvature(self, point: torch.Tensor, h: float = 1e-5) -> float:
        """
        Computes the true Gaussian curvature K for a 2D manifold at a given point.
        """
        if self.n_dims != 2:
            raise ValueError("True Gaussian curvature is defined for 2-dimensional manifolds only.")
        if len(point) != self.n_dims:
            raise ValueError(f"Point dimension ({len(point)}) does not match manifold dimension ({self.n_dims}).")

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

    # def compute_geodesic(self, start_point: torch.Tensor, end_point: torch.Tensor, num_points: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Robust discrete geodesic finder:
    #      - quick diagnostic to detect effectively-constant metric on the straight segment
    #      - otherwise optimize interior points with L-BFGS (multiple restarts) minimizing discrete energy
    #      - returns (path (T, D), total_length scalar tensor)
    #     """
    #     device = self.device
    #     dtype = torch.float32

    #     # --- basic validation & setup -----------------------------------------
    #     if start_point.dim() != 1 or end_point.dim() != 1:
    #         raise ValueError("start_point and end_point must be 1D tensors of shape (D,).")
    #     if start_point.shape != end_point.shape:
    #         raise ValueError("start_point and end_point must have the same shape.")
    #     D = start_point.numel()
    #     if D != self.n_dims:
    #         raise ValueError(f"Point dimension ({D}) does not match manifold dimension ({self.n_dims}).")

    #     x0 = start_point.to(device=device, dtype=dtype).clone()
    #     x1 = end_point.to(device=device, dtype=dtype).clone()

    #     # trivial identical points
    #     if torch.allclose(x0, x1, atol=1e-12, rtol=0.0):
    #         path = x0.unsqueeze(0).expand(num_points, D).clone()
    #         return path, torch.tensor(0.0, device=device, dtype=dtype)

    #     T = max(2, int(num_points))
    #     # straight-line initialization
    #     t_vals = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)
    #     X_init = torch.stack([(1.0 - t) * x0 + t * x1 for t in t_vals], dim=0)  # (T, D)
    #     X_init = torch.stack([self._clamp_point_to_bounds(X_init[i]) for i in range(T)], dim=0)

    #     if T <= 2:
    #         lin_dist = self.linear_interpolation_distance(x0, x1, num_points=T)
    #         return X_init, lin_dist

    #     # --- quick diagnostic: does metric vary along straight path? ----------
    #     with torch.no_grad():
    #         mids_straight = 0.5 * (X_init[1:] + X_init[:-1])  # (T-1, D)
    #         mids_clamped = self._clamp_point_to_bounds(mids_straight.view(-1, D)).view(mids_straight.shape)
    #         try:
    #             Gs_on_line = self.metric_tensor(mids_clamped, force=True)  # (T-1, D, D)
    #             # compute a simple scalar variability measure (std over all entries)
    #             var_metric = torch.std(Gs_on_line.view(-1))
    #         except Exception:
    #             # if metric evaluation fails for some reason, skip diagnostic
    #             var_metric = torch.tensor(float('inf'), device=device)

    #     # If metric barely varies along the straight line, it's likely the straight line is the geodesic
    #     if var_metric < 1e-7:
    #         # short-circuit and return linear path
    #         print("Metric variation along straight line is negligible; returning linear interpolation as geodesic.")
    #         lin_dist = self.linear_interpolation_distance(x0, x1, num_points=T)
    #         return X_init, lin_dist

    #     # --- energy function helper ------------------------------------------
    #     eps = 1e-12
    #     smoothing_lambda = 1e-6

    #     def discrete_energy_from_interior(interior_tensor: torch.Tensor):
    #         """
    #         interior_tensor: (T-2, D) tensor (requires_grad True during optimization)
    #         returns scalar energy (torch scalar).
    #         """
    #         X = torch.cat([x0.unsqueeze(0), interior_tensor, x1.unsqueeze(0)], dim=0)  # (T, D)
    #         diffs = X[1:] - X[:-1]  # (T-1, D)
    #         mids = 0.5 * (X[1:] + X[:-1])  # (T-1, D)
    #         mids_clamped_loc = self._clamp_point_to_bounds(mids.view(-1, D)).view(mids.shape)
    #         Gs_loc = self.metric_tensor(mids_clamped_loc, force=True)  # (T-1, D, D)
    #         seg_sq_loc = torch.einsum('si,sij,sj->s', diffs, Gs_loc, diffs).clamp(min=0.0)  # (T-1,)
    #         energy_loc = 0.5 * seg_sq_loc.sum()
    #         # small smoothing on second differences
    #         if T > 3:
    #             second_diffs = X[2:] - 2.0 * X[1:-1] + X[:-2]
    #             smoothing = smoothing_lambda * (second_diffs.pow(2).sum())
    #             energy_loc = energy_loc + smoothing
    #         return energy_loc

    #     # --- optimization with L-BFGS and multiple restarts -------------------
    #     best_interior = None
    #     best_energy = float('inf')
    #     restarts = 3
    #     lbfgs_max_iter = 200

    #     # base interior (straight line)
    #     base_interior = X_init[1:-1].detach().clone()

    #     for r in range(restarts):
    #         # initialize interior: first restart uses straight line, others use perturbed versions
    #         if r == 0:
    #             interior = base_interior.clone().requires_grad_(True)
    #         else:
    #             noise_scale = 0.05 * torch.norm(x1 - x0).item()
    #             perturb = (torch.randn_like(base_interior) * noise_scale).to(device)
    #             interior = (base_interior + perturb).requires_grad_(True)

    #         optimizer = torch.optim.LBFGS([interior], lr=1.0, max_iter=lbfgs_max_iter, line_search_fn='strong_wolfe')

    #         def closure():
    #             optimizer.zero_grad()
    #             loss = discrete_energy_from_interior(interior)
    #             # LBFGS requires a scalar tensor to be returned
    #             loss.backward()
    #             return loss

    #         try:
    #             optimizer.step(closure)
    #         except Exception:
    #             # if LBFGS fails (rare), try a short Adam run
    #             try:
    #                 interior_adam = interior.detach().clone().requires_grad_(True)
    #                 adam = torch.optim.Adam([interior_adam], lr=1e-2)
    #                 for _ in range(200):
    #                     adam.zero_grad()
    #                     loss_ad = discrete_energy_from_interior(interior_adam)
    #                     loss_ad.backward()
    #                     adam.step()
    #                     with torch.no_grad():
    #                         interior_adam.data = self._clamp_point_to_bounds(interior_adam.data.view(-1, D)).view(interior_adam.data.shape)
    #                 interior = interior_adam
    #             except Exception:
    #                 # give up on this restart
    #                 continue

    #         # evaluate final energy for this restart
    #         with torch.no_grad():
    #             e_final = float(discrete_energy_from_interior(interior).item())
    #             if e_final < best_energy:
    #                 best_energy = e_final
    #                 best_interior = interior.detach().clone()

    #     # if optimization failed for all restarts, fall back to linear path
    #     if best_interior is None:
    #         path = X_init
    #         dist = self.linear_interpolation_distance(x0, x1, num_points=T)
    #         return path, dist

    #     # --- assemble final path and compute Riemannian length ----------------
    #     with torch.no_grad():
    #         X_final = torch.cat([x0.unsqueeze(0), best_interior, x1.unsqueeze(0)], dim=0)  # (T, D)
    #         diffs = X_final[1:] - X_final[:-1]  # (T-1, D)
    #         mids = 0.5 * (X_final[1:] + X_final[:-1])
    #         mids_clamped = self._clamp_point_to_bounds(mids.view(-1, D)).view(mids.shape)
    #         Gs_final = self.metric_tensor(mids_clamped, force=True)  # (T-1, D, D)
    #         seg_sq_final = torch.einsum('si,sij,sj->s', diffs, Gs_final, diffs).clamp(min=0.0)
    #         seg_lengths = torch.sqrt(seg_sq_final + eps)
    #         total_length = seg_lengths.sum()
    #         # final safety clamp of path points
    #         X_final = torch.stack([self._clamp_point_to_bounds(X_final[i]) for i in range(X_final.shape[0])], dim=0)

    #     return X_final, total_length
    def compute_geodesic(self, start_point: torch.Tensor, end_point: torch.Tensor, num_points: int = 50, 
                     avoid_high_metric: bool = True, metric_threshold: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Geodesic computation with optional obstacle avoidance.
        
        Args:
            avoid_high_metric: If True, penalize paths through regions with high metric values
            metric_threshold: Metric value above which to apply penalty (auto-detected if None)
        """
        device = self.device
        dtype = torch.float32

        if start_point.dim() != 1 or end_point.dim() != 1:
            raise ValueError("start_point and end_point must be 1D tensors of shape (D,).")
        if start_point.shape != end_point.shape:
            raise ValueError("start_point and end_point must have the same shape.")
        D = start_point.numel()
        if D != self.n_dims:
            raise ValueError(f"Point dimension ({D}) does not match manifold dimension ({self.n_dims}).")

        x0 = start_point.to(device=device, dtype=dtype).clone()
        x1 = end_point.to(device=device, dtype=dtype).clone()

        if torch.allclose(x0, x1, atol=1e-12, rtol=0.0):
            path = x0.unsqueeze(0).expand(num_points, D).clone()
            return path, torch.tensor(0.0, device=device, dtype=dtype)

        T_opt = max(100, num_points * 3)
        
        # Initialize with straight line
        t_vals = torch.linspace(0.0, 1.0, T_opt, device=device, dtype=dtype)
        X_init = torch.stack([(1.0 - t) * x0 + t * x1 for t in t_vals], dim=0)
        
        # Auto-detect metric threshold if avoiding obstacles
        if avoid_high_metric and metric_threshold is None:
            with torch.no_grad():
                # Sample grid to find typical metric values
                sample_pts = X_init[::max(1, T_opt//20)]
                try:
                    Gs_sample = self.metric_tensor(sample_pts, force=True)
                    metric_norms = torch.norm(Gs_sample.view(len(sample_pts), -1), dim=1)
                    # Set threshold at 75th percentile
                    metric_threshold = torch.quantile(metric_norms, 0.75).item()
                except Exception:
                    metric_threshold = float('inf')
                    avoid_high_metric = False
        
        def compute_path_cost(interior: torch.Tensor):
            """
            Computes: Riemannian length + obstacle penalty + spacing regularization
            """
            X = torch.cat([x0.unsqueeze(0), interior, x1.unsqueeze(0)], dim=0)
            diffs = X[1:] - X[:-1]
            
            # Standard Riemannian length
            mids = 0.5 * (X[1:] + X[:-1])
            Gs = self.metric_tensor(mids, force=True)
            seg_sq = torch.einsum('si,sij,sj->s', diffs, Gs, diffs).clamp(min=0.0)
            seg_lengths = torch.sqrt(seg_sq + 1e-10)
            total_length = seg_lengths.sum()
            
            # Obstacle avoidance: penalize high metric regions
            obstacle_penalty = torch.tensor(0.0, device=device, dtype=dtype)
            if avoid_high_metric:
                # Evaluate metric magnitude at each interior point
                Gs_interior = self.metric_tensor(interior, force=True)
                metric_norms = torch.norm(Gs_interior.view(len(interior), -1), dim=1)
                
                # Soft penalty that grows for metric values above threshold
                excess = torch.relu(metric_norms - metric_threshold)
                # Scale penalty by path length to make it relative
                obstacle_penalty = 2.0 * total_length.detach() * excess.sum() / len(interior)
            
            # Equal spacing regularization
            mean_seg_length = seg_lengths.mean()
            spacing_penalty = 0.1 * torch.sum((seg_lengths - mean_seg_length) ** 2)
            
            return total_length + obstacle_penalty + spacing_penalty
        
        # Try multiple initializations
        best_interior = None
        best_cost = float('inf')
        
        for restart in range(10):
            if restart == 0:
                # Start with straight line
                interior = X_init[1:-1].clone()
            elif restart <= 2:
                # Try detouring above/below in first dimension
                interior = X_init[1:-1].clone()
                detour_amount = 0.3 * torch.norm(x1 - x0).item() * (1 if restart == 1 else -1)
                # Apply sinusoidal detour
                t_interior = torch.linspace(0, 1, T_opt-2, device=device)
                detour = detour_amount * torch.sin(np.pi * t_interior)
                interior[:, 0] += detour
            else:
                # Random perturbations
                noise_scale = 0.1 * torch.norm(x1 - x0).item()
                interior = X_init[1:-1].clone() + torch.randn(T_opt-2, D, device=device) * noise_scale
            
            interior = torch.stack([self._clamp_point_to_bounds(interior[i]) for i in range(T_opt-2)], dim=0)
            interior = interior.requires_grad_(True)
            
            optimizer = torch.optim.Adam([interior], lr=0.02)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=25, factor=0.5)
            
            prev_loss = float('inf')
            patience_counter = 0
            
            for iteration in range(600):
                optimizer.zero_grad()
                loss = compute_path_cost(interior)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_([interior], max_norm=1.0)
                optimizer.step()
                scheduler.step(loss)
                
                with torch.no_grad():
                    interior.data = torch.stack([self._clamp_point_to_bounds(interior[i]) 
                                                for i in range(T_opt-2)], dim=0)
                
                current_loss = loss.item()
                if abs(prev_loss - current_loss) < 1e-7:
                    patience_counter += 1
                    if patience_counter > 40:
                        break
                else:
                    patience_counter = 0
                prev_loss = current_loss
            
            with torch.no_grad():
                final_cost = compute_path_cost(interior).item()
                if final_cost < best_cost:
                    best_cost = final_cost
                    best_interior = interior.detach().clone()
        
        if best_interior is None:
            path = torch.stack([self._clamp_point_to_bounds(X_init[int(i * (T_opt-1) / (num_points-1))]) 
                            for i in range(num_points)], dim=0)
            lin_dist = self.linear_interpolation_distance(x0, x1, num_points=num_points)
            return path, lin_dist
        
        with torch.no_grad():
            X_opt = torch.cat([x0.unsqueeze(0), best_interior, x1.unsqueeze(0)], dim=0)
            
            # Resample
            if T_opt != num_points:
                indices = torch.linspace(0, T_opt - 1, num_points, device=device)
                indices_floor = indices.long()
                indices_ceil = (indices_floor + 1).clamp(max=T_opt - 1)
                alpha = (indices - indices_floor.float()).unsqueeze(1)
                path = (1 - alpha) * X_opt[indices_floor] + alpha * X_opt[indices_ceil]
            else:
                path = X_opt
            
            # Compute true Riemannian length (without penalties)
            diffs = path[1:] - path[:-1]
            mids = 0.5 * (path[1:] + path[:-1])
            Gs = self.metric_tensor(mids, force=True)
            seg_sq = torch.einsum('si,sij,sj->s', diffs, Gs, diffs).clamp(min=0.0)
            total_length = torch.sqrt(seg_sq + 1e-10).sum()
        
        return path, total_length

    def exact_geodesic_distance(self, p1: torch.Tensor, p2: torch.Tensor, num_points: int = 50) -> torch.Tensor:
        """
        Computes the exact geodesic distance between points by solving the
        augmented geodesic equation.
        
        Args:
            p1: Starting point(s). Shape (D,) for single point or (B, D) for batch.
            p2: Ending point(s). Shape (D,) for single point or (B, D) for batch.
            num_points: Number of discretization points for the solver.
            
        Returns:
            torch.Tensor: Scalar distance if inputs are 1D, or shape (B,) if inputs are 2D.
        """
        # Ensure inputs are on the same device/dtype
        if p1.device != p2.device:
            p2 = p2.to(p1.device)
            
        # Case 1: Single points (1D tensors)
        if p1.dim() == 1 and p2.dim() == 1:
            _, distance = self.compute_geodesic(p1, p2, num_points=num_points)
            return distance

        # Case 2: Batch of points (2D tensors)
        elif p1.dim() == 2 and p2.dim() == 2:
            if p1.shape != p2.shape:
                raise ValueError(f"Batch shapes must match. Got {p1.shape} and {p2.shape}")
            
            batch_size = p1.shape[0]
            distances = []
            
            # Loop through the batch
            # Note: Since the solver uses L-BFGS, a loop is often safer/more stable 
            # than trying to vectorize the optimizer steps.
            for i in range(batch_size):
                _, dist = self.compute_geodesic(p1[i], p2[i], num_points=num_points)
                distances.append(dist)
            
            return torch.stack(distances)

        else:
            raise ValueError(f"Inputs must be both 1D (single) or both 2D (batch). Got dims {p1.dim()} and {p2.dim()}")
    
        