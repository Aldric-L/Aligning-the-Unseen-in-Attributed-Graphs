import torch
from typing import List, Dict, Any, Tuple, Optional, Union, Callable


def compute_heat_kernel_from_laplacian(laplacian: torch.Tensor, t: Union[float, List[float]], eigenvals=None, eigenvecs=None, num_eigenvalues=None) -> torch.Tensor:
        """
        Compute heat kernel using laplacian: K(t) = exp(-t*L)
        """
        try:
            if eigenvals is True or (isinstance(eigenvals, torch.Tensor) and isinstance(eigenvecs, torch.Tensor)):
                if eigenvals is None or eigenvecs is None or eigenvals is True:
                    # Eigendecomposition of Laplacian
                    eigenvals, eigenvecs = torch.linalg.eigh(laplacian)
                    
                    # Clamp eigenvalues to avoid numerical issues
                    eigenvals = torch.clamp(eigenvals, min=0.0)
            
                # Take only the first num_eigenvalues for efficiency
                if num_eigenvalues is not None and num_eigenvalues < eigenvals.size(0):
                    eigenvals = eigenvals[:num_eigenvalues]
                    eigenvecs = eigenvecs[:, :num_eigenvalues]
            
                # Compute heat kernel: K(t) = V * exp(-t*Λ) * V^T
                if isinstance(t, (float, int)):
                    exp_eigenvals = torch.exp(-t * eigenvals)
                    return eigenvecs @ torch.diag(exp_eigenvals) @ eigenvecs.t()
                else:
                    heat_kernels = []
                    for t_i in t:
                        exp_eigenvals = torch.exp(-t_i * eigenvals)
                        K_t = eigenvecs @ torch.diag(exp_eigenvals) @ eigenvecs.t()
                        heat_kernels.append(K_t)
                    return heat_kernels
                
            # Compute heat kernel: K(t) = V * exp(-t*Λ) * V^T
            if isinstance(t, (float, int)):
                return torch.matrix_exp(-t * laplacian)
            else:
                heat_kernels = []
                for t_i in t:
                    K_t = torch.matrix_exp(-t_i * laplacian)
                    heat_kernels.append(K_t)
                return heat_kernels
                        
        except Exception as e:
            print(f"Warning: heat kernel computation failed: {e}")
            # Fallback to identity
            return torch.eye(laplacian.size(0), device=laplacian.device, dtype=laplacian.dtype)
        
def compute_heat_kernel_divergence(
    K_manifold: Union[torch.Tensor, List[torch.Tensor]],
    K_graph:    Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
    """
    Compute divergence between manifold heat kernel and graph heat kernel
    Using Frobenius norm of the difference
    """
    # If we got lists of kernels, compute per‐t and average:
    if isinstance(K_manifold, (list, tuple)):
        divergences = [
            compute_heat_kernel_divergence(Km, Kg)
            for Km, Kg in zip(K_manifold, K_graph)
        ]
        # average (you can also sum or weight here)
        return torch.stack(divergences).sum()

    # Normalize both kernels to have same trace for fair comparison
    trace_manifold = torch.trace(K_manifold)
    trace_graph    = torch.trace(K_graph)
    
    if trace_manifold > 1e-8:
        K_manifold_norm = K_manifold * (trace_graph / trace_manifold)
    else:
        K_manifold_norm = K_manifold
    
    # Compute Frobenius norm of difference
    diff = K_manifold_norm - K_graph
    #return torch.norm(diff, p='fro')
    return (diff**2).sum()

def compute_graph_laplacian(adjacency_matrix: torch.tensor, normalize: bool = True, laplacian_regularization : float = 0.0) -> torch.Tensor:
    # Compute degree matrix
    num_nodes = adjacency_matrix.size(0)
    degree = torch.sum(adjacency_matrix, dim=1)
    D = torch.diag(degree)
    
    # Laplacian: L = D - A
    if normalize:
        d_inv_sqrt = torch.where(degree > 0, torch.pow(degree + 1e-8, -0.5), torch.zeros_like(degree))
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        laplacian = torch.eye(adjacency_matrix.size(0)) - D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt
        # print("Any NaN in L_sym?", torch.isnan(laplacian).any().item())
        # print("Any Inf in L_sym?", torch.isinf(laplacian).any().item())
    else:
        laplacian = D - adjacency_matrix
    
    # Add small regularization
    if laplacian_regularization > 0:
        laplacian += laplacian_regularization * torch.eye(num_nodes, device=adjacency_matrix.device)
    
    return laplacian

def compute_graph_laplacian_from_targets(targets: Dict[str, torch.Tensor], normalize: bool = True, laplacian_regularization : float = 0, threshold_eps: float = 1e-6, debug: bool = False,) -> torch.Tensor:
    """
    Compute the graph Laplacian from edge information
    """
    if "edge_index" in targets:
        edge_index = targets["edge_index"]
        num_nodes = targets.get("num_nodes", torch.max(edge_index) + 1)
        edge_weights = targets.get("edge_labels", torch.ones(edge_index.size(1)))
        
        # Create adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device, dtype=torch.float32)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            weight = edge_weights[i].item() if edge_weights is not None else 1.0
            adj[src, dst] = weight
            adj[dst, src] = weight  # Symmetric
        
    elif "adj_matrix" in targets:
        adj = targets["adj_matrix"].float()
        num_nodes = adj.size(0)
    else:
        raise ValueError("Targets must contain either 'edge_index' or 'adj_matrix'")
    
    sigma = compute_sigma_with_knn(adj)
    weights = torch.exp(-adj**2 / (2 * sigma**2))

    # --- Zero out diagonal safely (autograd-friendly) ---
    eye = torch.eye(weights.size(0), device=weights.device)
    weights = weights * (1 - eye)

    # --- Compute cutoff distance based on sigma and epsilon ---
    d_cut = sigma * torch.sqrt(-2 * torch.log(torch.tensor(threshold_eps, device=weights.device)))
    w_cut = torch.exp(-(1.2 * d_cut)**2 / (2 * sigma**2))  # == threshold_eps numerically

    # --- Threshold small weights (sparsify, but remain differentiable) ---
    weights = torch.where(weights > w_cut, weights, torch.zeros_like(weights))

    # --- Debug information ---
    if debug:
        nonzero_frac = (weights > 0).float().mean().item()
        energy_retained = weights.sum().item() / (torch.exp(-adj**2 / (2 * sigma**2)).sum().item() + 1e-12)
        print(
            f"[DEBUG] (From Existing Graph) RBF sparsity: {(1 - nonzero_frac) * 100:.2f}% zeros | "
            f"Energy retained: {energy_retained * 100:.3f}% | "
            f"Cutoff weight={w_cut.item():.2e}"
        )
    
    return compute_graph_laplacian(adjacency_matrix=weights, normalize=normalize, laplacian_regularization=laplacian_regularization)
        

def compute_sigma_with_knn(distances: torch.Tensor, knn_for_sigma: int =7):
    num_nodes = distances.size(0)
    if knn_for_sigma >= num_nodes**0.5 or knn_for_sigma is None:
        knn_for_sigma = max(1, num_nodes**0.5 - 1)

    # sort row-wise (excluding diagonal)
    #sorted_dists, _ = torch.sort(distances + torch.eye(num_nodes, device=device) * 1e9, dim=1)
    # k-th nearest: index knn_for_sigma (0-based if we excluded diag by big number)
    #sigma_i = sorted_dists[:, knn_for_sigma].clamp(min=1e-8)  # shape (N,)
    #sigma_i = sigma_i.clamp(min=1e-4).view(num_nodes, 1)
    # build symmetric, locally-scaled Gaussian kernel
    #sigma_matrix = sigma_i * sigma_i.t()  # σ_i * σ_j
    #print("Min/Max sigma_matrix:", torch.min(sigma_matrix), torch.max(sigma_matrix))
    #alpha = self.lag_factor  # small update
    # if self.sigma_ema is None:
    #      self.sigma_ema = sigma_matrix
    # else:
    #     sigma_matrix = (1 - alpha) * self.sigma_ema + alpha * sigma_matrix
    #     self.sigma_ema = sigma_matrix
    # print("Min/Max sigma_matrix (corrected):", torch.min(sigma_matrix), torch.max(sigma_matrix))
    sigma = torch.median(distances)
    return sigma

# def compute_manifold_laplacian(distances: torch.Tensor, sigma: float, normalize:bool=True, laplacian_regularization : float = 0) -> torch.Tensor:
#     """
#     Compute the Laplace-Beltrami operator on the manifold using proper Riemannian distances
#     Fully differentiable using Gaussian RBF weights
#     """
#     weights = torch.exp(-distances**2 / (2 * sigma**2))

#     # zero out diagonal (no self-weight)
#     weights = weights * (1 - torch.eye(weights.size(0), device=weights.device))
#     # Note: you would very much like to use weights.fill_diagonal_(0.0) but it breaks autograd...
    
#     # Optional: Threshold very small weights to maintain some sparsity
#     threshold = 1e-4
#     weights = torch.where(weights > threshold, weights, torch.zeros_like(weights))
    
#     return compute_graph_laplacian(adjacency_matrix=weights, normalize=normalize, laplacian_regularization=laplacian_regularization)

def compute_manifold_laplacian(
    distances: torch.Tensor,
    sigma: float,
    normalize: bool = True,
    laplacian_regularization: float = 0.0,
    threshold_eps: float = 1e-6,
    debug: bool = False,
) -> torch.Tensor:
    """
    Compute the Laplace–Beltrami operator (manifold Laplacian) using Gaussian RBF weights.
    Adds an optional sparsity threshold based on sigma.
    Fully differentiable and compatible with backprop.

    Args:
        distances: Pairwise Riemannian distances (N x N tensor).
        sigma: RBF bandwidth.
        normalize: Whether to compute normalized Laplacian.
        laplacian_regularization: Optional regularization term added to diagonal.
        threshold_eps: Relative cutoff epsilon. Default 1e-6 keeps 99.9999% of RBF energy.
        debug: If True, prints diagnostic info on sparsity and energy retention.

    Returns:
        torch.Tensor: The (optionally normalized) Laplacian matrix.
    """
    # --- Compute RBF weights ---
    weights = torch.exp(-distances**2 / (2 * sigma**2))

    # --- Zero out diagonal safely (autograd-friendly) ---
    eye = torch.eye(weights.size(0), device=weights.device)
    weights = weights * (1 - eye)

    # --- Compute cutoff distance based on sigma and epsilon ---
    d_cut = sigma * torch.sqrt(-2 * torch.log(torch.tensor(threshold_eps, device=weights.device)))
    w_cut = torch.exp(-(1.2 * d_cut)**2 / (2 * sigma**2))  # == threshold_eps numerically

    # --- Threshold small weights (sparsify, but remain differentiable) ---
    weights = torch.where(weights > w_cut, weights, torch.zeros_like(weights))

    # --- Debug information ---
    if debug:
        nonzero_frac = (weights > 0).float().mean().item()
        energy_retained = weights.sum().item() / (torch.exp(-distances**2 / (2 * sigma**2)).sum().item() + 1e-12)
        print(
            f"[DEBUG] (From Laplacian) RBF sparsity: {(1 - nonzero_frac) * 100:.2f}% zeros | "
            f"Energy retained: {energy_retained * 100:.3f}% | "
            f"Cutoff weight={w_cut.item():.2e}"
        )

    # --- Compute Laplacian ---
    L = compute_graph_laplacian(
        adjacency_matrix=weights,
        normalize=normalize,
        laplacian_regularization=laplacian_regularization,
    )

    return L


def approx_spectral_norm(K: torch.Tensor, iters: int = 3, device=None) -> float:
    """
    Cheap power-iteration for spectral norm approximation.
    Only uses mat-vec ops. If iters==0 returns Frobenius norm as fallback proxy.
    """
    if iters <= 0:
        return float(torch.norm(K, p='fro').item())
    n = K.shape[0]
    # random init (unit)
    v = torch.randn(n, device=K.device, dtype=K.dtype)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        w = K.mv(v)            # K @ v  (fast)
        normw = w.norm()
        if normw.item() == 0:
            return 0.0
        v = w / (normw + 1e-12)
    # Rayleigh quotient ~ spectral norm
    sigma = float(v.dot(K.mv(v)).item())
    # ensure non-negative
    return max(sigma, 0.0)

def check_heat_kernels_informativeness_fast(
    K_list: List[torch.Tensor],
    *,
    trace_low_frac: float = 0.01,     # keep if trace/N > trace_low_frac
    trace_high_frac: float = 0.99,    # drop if trace/N > trace_high_frac (too identity-like)
    var_eps: float = 1e-6,            # minimal entry variance
    diag_offdiag_ratio_min: float = 0.1,   # minimal diag/offdiag ratio to avoid pure diag
    diag_offdiag_ratio_max: float = 10.0,  # maximal ratio to avoid pure diag or pure constant
    rowstd_eps: float = 1e-4,         # minimal std of row sums (normalized)
    use_power_iter: bool = False,     # optional small-power-iteration for spectral norm proxy
    power_iters: int = 3,
    verbose: bool = True
) -> Tuple[List[Dict], torch.Tensor]:
    """
    Fast, eig-free check for informativeness of each kernel in K_list.

    Args:
        K_list: list of (N,N) symmetric torch.Tensor heat kernels
        thresholds: tunable heuristics
        use_power_iter: whether to approximate spectral norm (slower but still cheap)
    Returns:
        diagnostics: list of dicts with stats for each K
        kept_mask: boolean tensor with True for informative kernels
    """
    diagnostics = []
    kept = []

    if len(K_list) == 0:
        return diagnostics, torch.tensor([], dtype=torch.bool)

    N = K_list[0].shape[0]
    # Precompute normalization denom used across checks
    eps = 1e-12

    for i, K in enumerate(K_list):
        # ensure float and on same device
        K = K.to(dtype=torch.get_default_dtype())

        # Symmetrize (cheap) to avoid numerical asymmetry
        K = 0.5 * (K + K.T)

        # Basic stats (cheap)
        trace = float(torch.trace(K).item())
        trace_norm = trace / max(N, 1)

        mean_all = float(K.mean().item())
        var_all = float(K.var(unbiased=False).item())  # population var

        # diag vs off-diagonal means
        diag_mean = float(torch.diagonal(K).mean().item())
        # sum of all entries minus diagonal sum -> off-diag sum
        offdiag_sum = float(K.sum().item() - torch.diagonal(K).sum().item())
        offdiag_count = max(N * (N - 1), 1)
        offdiag_mean = offdiag_sum / offdiag_count

        # diag/offdiag ratio (avoid division by zero)
        if abs(offdiag_mean) < eps:
            diag_offdiag_ratio = float('inf') if diag_mean > eps else 1.0
        else:
            diag_offdiag_ratio = float((diag_mean / (offdiag_mean + 1e-20)))

        # row sums statistics: if rows are identical -> constant kernel
        row_sums = K.sum(dim=1)
        row_sums_norm = row_sums / (row_sums.mean().abs() + 1e-12)
        row_std = float(row_sums_norm.std(unbiased=False).item())

        # magnitude diagnostics
        mean_abs = float(K.abs().mean().item())
        max_abs = float(K.abs().max().item())

        # optional cheap spectral proxy
        spectral_proxy = None
        if use_power_iter:
            spectral_proxy = approx_spectral_norm(K, iters=power_iters)

        # Decision rules (heuristic, tunable)
        # Condition A: trace should not be extremely close to N (identity) nor extremely close to 1 (constant)
        cond_trace = (trace_norm > trace_low_frac) and (trace_norm < trace_high_frac)

        # Condition B: variance of entries should be non-trivial
        cond_var = var_all > var_eps

        # Condition C: diag/offdiag ratio should be within a reasonable band
        cond_diag = (diag_offdiag_ratio > diag_offdiag_ratio_min) and (diag_offdiag_ratio < diag_offdiag_ratio_max)

        # Condition D: rows should not be identical
        cond_row = row_std > rowstd_eps

        informative = bool(cond_trace and cond_var and cond_diag and cond_row)

        diagnostics.append({
            "idx": i,
            "trace": trace,
            "trace_norm": trace_norm,
            "mean": mean_all,
            "var": var_all,
            "diag_mean": diag_mean,
            "offdiag_mean": offdiag_mean,
            "diag_offdiag_ratio": diag_offdiag_ratio,
            "row_std": row_std,
            "mean_abs": mean_abs,
            "max_abs": max_abs,
            "spectral_proxy": spectral_proxy,
            "informative": informative,
            "conditions": {
                "trace_ok": cond_trace,
                "var_ok": cond_var,
                "diag_ok": cond_diag,
                "row_ok": cond_row
            }
        })
        kept.append(informative)

        if verbose:
            print(f"[t_{i:02d}] tr/N={trace_norm:.4f}, var={var_all:.2e}, diag/off={diag_offdiag_ratio:.2f}, "
                  f"row_std={row_std:.2e}, keep={informative}")

    kept_mask = torch.tensor(kept, dtype=torch.bool)
    return diagnostics, kept_mask


def compute_heat_time_scale_from_laplacian(
    L: torch.Tensor,
    *,
    num_times: int = 10,
    retain_high_freq_threshold: float = 0.95,   # a in notes: e^{-t_min * lambda_max} >= a
    suppress_low_freq_threshold: float = 1e-3,  # b in notes: e^{-t_max * lambda_2} <= b
    trace_eps: float = 1e-4,                    # remove times where K trace is ~0
    clamp_min_log: float = -12.0,               # avoid extremely small times in log-space
    clamp_max_log: float = 12.0,                # avoid extremely large times in log-space
    eigen_compute_method: str = "full",         # "full" or "approx" (approx not implemented here)
    return_diagnostics: bool = True,
) -> Tuple[torch.Tensor, Optional[Dict]]:
    """
    Compute a meaningful set of heat times for a given symmetric Laplacian L.
    Returns (heat_times, diagnostics) where heat_times is a 1D torch tensor of shape (T,)
    and diagnostics contains ('lambda_max', 'lambda_pos_min', 't_min', 't_max', 'traces', 'kept_mask').
    ---
    Notes:
    - L must be symmetric (torch.float32/64), shape (N, N).
    - For very large N, computing full eigendecomposition is O(N^3). Consider using
        sparse/eigsh (scipy) externally to get the few required extremal eigenvalues.
    """

    if L.dim() != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square 2D tensor")

    # ensure symmetric
    if not torch.allclose(L, L.T, atol=1e-6, rtol=1e-5):
        # symmetrize for numerical safety
        L = 0.5 * (L + L.T)

    N = L.shape[0]

    # --- compute eigenvalues (full) ---
    # For large N, user should replace this step with a sparse eigen-solver to get
    # lambda_max and the smallest positive eigenvalue (lambda_2).
    if eigen_compute_method == "full":
        # eigenvalues in ascending order
        ev = torch.linalg.eigvalsh(L)  # real, sorted
        # cast to real float (torch returns real tensor)
        ev = ev.clamp_min(0.0)  # numerical negatives -> 0
        lambda_max = float(ev[-1].item())
        # find smallest positive eigenvalue (strictly > small_eps)
        small_eps = 1e-12
        pos_eigs = ev[ev > small_eps]
        if pos_eigs.numel() == 0:
            # all zeros: trivial Laplacian (disconnected with all-zero spectrum)
            lambda_pos_min = 0.0
        else:
            lambda_pos_min = float(pos_eigs[0].item())
    else:
        raise NotImplementedError("Only full eigen computation implemented. "
                                "For large graphs, compute lambda_max and lambda_2 externally (e.g. scipy.sparse.linalg.eigsh).")

    # If lambda_max is zero (degenerate), choose a fallback to avoid division by zero:
    if lambda_max <= 0.0:
        lambda_max = 1.0

    # If lambda_pos_min is zero (multiple connected components or degenerate), try fallback:
    if lambda_pos_min <= 0.0:
        # fallback: use a small positive fraction of lambda_max
        lambda_pos_min = lambda_max * 1e-6

    # --- derive t_min and t_max from thresholds ---
    # t_min: keep high-frequency content up to lambda_max with factor `retain_high_freq_threshold`
    # t_min <= -ln(a) / lambda_max
    t_min = -torch.log(torch.tensor(retain_high_freq_threshold, dtype=torch.float64)) / float(lambda_max)
    # t_max: suppress non-zero modes up to lambda_pos_min with factor `suppress_low_freq_threshold`
    t_max = -torch.log(torch.tensor(suppress_low_freq_threshold, dtype=torch.float64)) / float(lambda_pos_min)

    # numeric clamps on log scale to avoid extremes
    # compute logs then clamp
    min_log = float(torch.log10(torch.tensor(t_min)).item()) if t_min > 0.0 else clamp_min_log
    max_log = float(torch.log10(torch.tensor(t_max)).item()) if t_max > 0.0 else clamp_max_log

    # clamp
    min_log = max(min_log, clamp_min_log)
    max_log = min(max_log, clamp_max_log)
    if min_log >= max_log:
        # degenerate: choose a small window around min_log
        max_log = min_log + 1.0

    # sample logspace
    heat_times = torch.logspace(min_log, max_log, steps=num_times, base=10.0, dtype=torch.float64)

    # --- filter out times where heat kernel is essentially null by checking trace(K) ~ 0 ---
    # trace(K(t)) = sum_i e^{-t * lambda_i} = sum of exponentials of spectrum.
    # computing full kernel is O(N^3) but trace can be computed from eigenvalues easily
    # as sum(exp(-t * ev)).
    ev_np = ev.cpu() if isinstance(ev, torch.Tensor) else torch.tensor([ev])
    traces = torch.stack([torch.exp(-t * ev_np).sum() for t in heat_times])  # shape (T,)

    # If trace is extremely small relative to trace at t=0 (=N), drop those times
    trace0 = float(N)
    kept_mask = traces > (trace0 * trace_eps)

    # Optionally remove times that are too close to trace0 (no smoothing)
    # e.g., if desired to avoid times that do not smooth at all:
    # keep only times where some smoothing happens: traces < trace0 * (1 - tiny)
    # tiny = 1e-6
    # kept_mask &= (traces < trace0 * (1.0 - tiny))

    if kept_mask.sum().item() == 0:
        # keep at least one time (the smallest)
        kept_mask[0] = True

    heat_times_kept = heat_times[kept_mask]

    diagnostics = {
        "lambda_max": lambda_max,
        "lambda_pos_min": lambda_pos_min,
        "t_min": float(t_min),
        "t_max": float(t_max),
        "min_log": float(min_log),
        "max_log": float(max_log),
        "candidate_times": heat_times.cpu().numpy(),
        "traces": traces.cpu().numpy(),
        "kept_mask": kept_mask.cpu().numpy(),
        "kept_times": heat_times_kept.cpu().numpy()
    }

    if return_diagnostics:
        return heat_times_kept.type(torch.get_default_dtype()), diagnostics
    else:
        return heat_times_kept.type(torch.get_default_dtype()), None
