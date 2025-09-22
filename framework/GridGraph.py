
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from typing import Optional, Tuple, List, Union, Dict, Any
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import heapq
import math
import traceback
import sys

import heapq
def dijkstra_worker(adjacency, source, targets):
    """
    Top-level Dijkstra worker suitable for ProcessPoolExecutor (picklable).
    adjacency: dict[node] -> list[(neighbor, weight)]
    source: int
    targets: iterable of ints
    Returns: dict target_node -> path (list of nodes) or None
    """
    dist = {}
    prev = {}
    visited = set()
    heap = [(0.0, source)]
    dist[source] = 0.0
    targets_remaining = set(targets)
    results = {}

    while heap and targets_remaining:
        d_u, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if d_u > dist.get(u, float('inf')):
            continue

        if u in targets_remaining:
            # reconstruct path
            path = []
            cur = u
            while True:
                path.append(cur)
                if cur == source:
                    break
                cur = prev[cur]
            path.reverse()
            results[u] = path
            targets_remaining.remove(u)
            if not targets_remaining:
                break

        for v, w in adjacency.get(u, []):
            nd = d_u + float(w)
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    # ensure all targets have an entry
    for t in targets:
        if t not in results:
            results[t] = None
    return results


class GridGraph:
    """
    GridGraph with:
      - tqdm for building adjacency, query->grid, query->query, and pathfinding stages
      - precomputation parallelized: shortest-path (Dijkstra) stage runs in multiple processes
      - gradient-aware recomputation runs in-process (to preserve autograd)
      - options to control number of workers and whether to use processes for pathfinding
    """

    def __init__(self, 
                 bounded_manifold,
                 connectivity: str = "nearest_neighbors",
                 include_diagonal: bool = True,
                 max_distance: float = None):
        self.manifold = bounded_manifold
        self.connectivity = connectivity
        self.include_diagonal = include_diagonal
        self.max_distance = max_distance

        self.n_dims = bounded_manifold.n_dims
        self.grid_size = bounded_manifold.grid_size
        self.bounds = bounded_manifold.bounds
        self.device = bounded_manifold.device

        self.node_positions = None
        self.node_indices = None
        self.edge_index = None
        self.edge_attr = None
        self.num_nodes = None

        self._build_graph()

    def _build_graph(self):
        self._create_nodes()
        self._create_edges()

    def _create_nodes(self):
        ranges = [torch.arange(self.grid_size + 1, device=self.device) for _ in range(self.n_dims)]
        grid_indices = torch.meshgrid(*ranges, indexing='ij')
        self.node_indices = torch.stack([g.flatten() for g in grid_indices], dim=1)  # (N, D)
        self.num_nodes = int(self.node_indices.shape[0])
        self.node_positions = self._grid_to_physical(self.node_indices)

    def _grid_to_physical(self, grid_indices: torch.Tensor) -> torch.Tensor:
        steps = (self.bounds[:, 1] - self.bounds[:, 0]) / self.grid_size
        return self.bounds[:, 0] + grid_indices.float() * steps

    def _create_edges(self):
        if self.connectivity == "nearest_neighbors":
            self._create_nearest_neighbor_edges()
        elif self.connectivity == "radius":
            self._create_radius_edges()
        elif self.connectivity == "full":
            self._create_full_edges()
        else:
            raise ValueError(f"Unknown connectivity type: {self.connectivity}")

    def _create_nearest_neighbor_edges(self):
        device = self.device
        N = self.num_nodes
        D = self.n_dims

        # Offsets
        if self.include_diagonal:
            choices = [torch.tensor([-1, 0, 1], dtype=torch.long, device=device) for _ in range(D)]
            offsets = torch.cartesian_prod(*choices) if hasattr(torch, "cartesian_prod") else torch.cartesian_prod(*choices)
            offsets = offsets[~(offsets == 0).all(dim=1)]
        else:
            offsets = torch.zeros(2 * D, D, dtype=torch.long, device=device)
            for i in range(D):
                offsets[2*i, i] = -1
                offsets[2*i+1, i] = 1

        # Vectorized neighbor indices
        neighbors = self.node_indices.unsqueeze(1) + offsets.unsqueeze(0)  # (N, K, D)
        valid_mask = (neighbors >= 0) & (neighbors <= self.grid_size)
        valid_mask = valid_mask.all(dim=2)  # (N, K)
        if valid_mask.sum() == 0:
            self.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            self.edge_attr = torch.empty((0,), device=device)
            return

        # strides for flattening
        strides = self.manifold._strides.to(device)  # (D,)
        flat_neighbors = (neighbors * strides.view(1, 1, -1)).sum(dim=2)  # (N, K)

        src_nodes = torch.arange(N, device=device).unsqueeze(1).expand(-1, flat_neighbors.shape[1])  # (N, K)
        src_flat = src_nodes[valid_mask]  # (M,)
        tgt_flat = flat_neighbors[valid_mask]  # (M,)

        a = torch.minimum(src_flat, tgt_flat)
        b = torch.maximum(src_flat, tgt_flat)
        pairs = torch.stack([a, b], dim=0)
        pairs_unique = torch.unique(pairs.t(), dim=0).t().contiguous()
        self.edge_index = pairs_unique.to(torch.long)
        self._compute_edge_attributes()

    def _create_radius_edges(self):
        if self.max_distance is None:
            raise ValueError("max_distance must be specified for radius connectivity")
        distances = torch.cdist(self.node_positions, self.node_positions)
        adj_matrix = (distances <= self.max_distance) & (distances > 0)
        self.edge_index, _ = dense_to_sparse(adj_matrix.float())
        self._compute_edge_attributes()

    def _create_full_edges(self):
        device = self.device
        N = self.num_nodes
        src = torch.arange(N, device=device).repeat_interleave(N - 1)
        tgt_list = []
        for i in range(N):
            if i > 0:
                tgt_list.append(torch.arange(0, i, device=device))
            if i < N - 1:
                tgt_list.append(torch.arange(i+1, N, device=device))
        tgt = torch.cat(tgt_list)
        self.edge_index = torch.stack([src, tgt], dim=0)
        self._compute_edge_attributes()

    def _compute_edge_attributes(self):
        if self.edge_index is None or self.edge_index.shape[1] == 0:
            self.edge_attr = torch.empty((0,), device=self.device)
            return
        src = self.edge_index[0]
        tgt = self.edge_index[1]
        source_pos = self.node_positions[src]
        target_pos = self.node_positions[tgt]
        dx = target_pos - source_pos  # (E, D)
        try:
            g_source = self.manifold.metric_tensor(source_pos)  # (E, D, D)
            g_target = self.manifold.metric_tensor(target_pos)
            # g_avg = (g_source + g_target) * 0.5
            # q = torch.bmm(torch.bmm(dx.unsqueeze(1), g_avg), dx.unsqueeze(2)).squeeze()
            # q = torch.clamp(q, min=1e-12)
            # weights = torch.sqrt(q)

            tmp0 = torch.bmm(g_source, dx.unsqueeze(-1)).squeeze(-1)  # (E, D)
            q0 = (dx * tmp0).sum(dim=1)                              # (E,)

            tmp1 = torch.bmm(g_target, dx.unsqueeze(-1)).squeeze(-1)
            q1 = (dx * tmp1).sum(dim=1)

            eps = 1e-12
            f0 = torch.sqrt(torch.clamp(q0, min=eps))
            f1 = torch.sqrt(torch.clamp(q1, min=eps))

            # trapezoid rule on the integrand (uses only endpoint metrics)
            weights = 0.5 * (f0 + f1)
            self.edge_attr = weights
        except Exception:
            distances = torch.norm(dx, dim=1)
            self.edge_attr = distances

    def get_node_features(self, feature_type: str = "position") -> torch.Tensor:
        if feature_type == "position":
            return self.node_positions
        elif feature_type == "grid_index":
            return self.node_indices.float()
        elif feature_type == "metric":
            return self._get_metric_features()
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    def _get_metric_features(self) -> torch.Tensor:
        metrics = self.manifold.metric_tensor(self.node_positions)
        if metrics.dim() == 3:
            return metrics.view(metrics.shape[0], -1)
        else:
            return metrics

    def to_pyg_data(self, node_features: str = "position", include_edge_attr: bool = True) -> Data:
        x = self.get_node_features(node_features)
        edge_attr = self.edge_attr if include_edge_attr else None
        data = Data(x=x, edge_index=self.edge_index, edge_attr=edge_attr, pos=self.node_positions)
        return data

    def get_subgraph(self, node_mask: torch.Tensor) -> 'GridGraph':
        raise NotImplementedError("Subgraph extraction not yet implemented")

    def visualize_2d(self, node_colors: Optional[torch.Tensor] = None, save_path: Optional[str] = None):
        if self.n_dims != 2:
            raise ValueError("Visualization only supported for 2D grids")
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            G = nx.Graph()
            G.add_nodes_from(range(self.num_nodes))
            if self.edge_index.shape[1] > 0:
                edges = self.edge_index.t().cpu().numpy()
                G.add_edges_from(edges)
            pos = {i: self.node_positions[i].cpu().numpy() for i in range(self.num_nodes)}
            plt.figure(figsize=(10, 10))
            nx.draw(G, pos, 
                   node_color=node_colors.cpu() if node_colors is not None else 'lightblue',
                   node_size=50,
                   alpha=0.7)
            if save_path:
                plt.savefig(save_path)
            plt.show()
        except ImportError:
            print("matplotlib and networkx required for visualization")

    def get_edge_weights(self, weight_type: str = "geodesic") -> torch.Tensor:
        if self.edge_index.shape[1] == 0:
            return torch.empty((0,), device=self.device)
        if weight_type == "geodesic":
            return self.edge_attr
        elif weight_type == "euclidean":
            source_pos = self.node_positions[self.edge_index[0]]
            target_pos = self.node_positions[self.edge_index[1]]
            return torch.norm(target_pos - source_pos, dim=1)
        elif weight_type == "unit":
            return torch.ones(self.edge_index.shape[1], device=self.device)
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")

    ####################################################################
    # compute_shortest_paths now uses explicit staged tqdm and parallel dijkstra
    ####################################################################
    def compute_shortest_paths(self, 
                             query_points: torch.Tensor,
                             weight_type: str = "geodesic",
                             connection_radius: float = None,
                             max_grid_neighbors: int = 8,
                             num_threads: int = 8,
                             pathfinding_workers: int = None,
                             use_process_for_pathfinding: bool = True) -> torch.Tensor:
        """
        Added parameters:
          - pathfinding_workers: number of processes to use for the precomputation (defaults to num_threads)
          - use_process_for_pathfinding: if True, use ProcessPoolExecutor for the Dijkstra stage (bypass GIL)
        """
        query_points = query_points.to(self.device).float()
        n_query = query_points.shape[0]

        if n_query == 0:
            return torch.empty((0, 0), device=self.device)
        if n_query == 1:
            return torch.zeros((1, 1), device=self.device)
        if not self.manifold._is_within_bounds(query_points).all():
            outside_indices = ~self.manifold._is_within_bounds(query_points)
            out_of_bounds_points = query_points[outside_indices]
            
            # Print the points that are outside
            print("The following query points are outside the manifold bounds:")
            print(out_of_bounds_points)
            raise ValueError("Some query points are outside the manifold bounds")

        # Stage 1: build extended graph (no-grad)  -> show progress inside
        with torch.no_grad():
            extended_graph_info = self._build_extended_graph_structure(
                query_points, connection_radius, max_grid_neighbors, show_progress=True, weight_type=weight_type
            )

        # Stage 2: find shortest paths between query nodes -> use parallel Dijkstra (process pool) with tqdm
        with torch.no_grad():
            paths = self._find_shortest_paths_parallel(extended_graph_info, n_query,
                                                       workers=pathfinding_workers or num_threads,
                                                       use_processes=use_process_for_pathfinding)

        # Stage 3: recompute distances with gradients (must be in-process)
        distance_matrix = self._recompute_path_distances_multithreaded(
            paths, extended_graph_info, query_points, weight_type, n_query, num_threads
        )

        return distance_matrix

    ####################################################################
    # Build extended graph structure but instrument with tqdm and vectorized weight computations
    ####################################################################
    def _build_extended_graph_structure(self,
                                        query_points: torch.Tensor,
                                        connection_radius: float,
                                        max_grid_neighbors: int,
                                        show_progress: bool = False,
                                        weight_type: str = "geodesic") -> dict:
        """
        Builds adjacency dict where edge weights are floats.
        If weight_type == "geodesic", uses metric tensor-based edge lengths.
        This function returns a picklable adjacency dict suitable for dijkstra_worker.
        """

        n_query = query_points.shape[0]
        if connection_radius is None:
            grid_spacing = torch.mean(self.manifold.steps)
            connection_radius = 2.0 * grid_spacing

        all_positions = torch.cat([self.node_positions, query_points], dim=0)
        adjacency: Dict[int, List[Tuple[int, float]]] = {}
        device = self.device
        n_grid = self.num_nodes

        # Bring positions to cpu numpy for adjacency dict (and to keep adjacency picklable)
        pos_cpu = all_positions.detach().cpu().numpy()  # shape (n_grid + n_query, D)

        # Precompute metric tensors on CPU for grid nodes (and queries if geodesic)
        if weight_type == "geodesic":
            # metric_tensor expects torch tensors: compute on CPU to get numpy arrays
            with torch.no_grad():
                grid_pos_t = self.node_positions.cpu()  # (n_grid, D)

                # bounds on CPU
                lower = self.bounds[:, 0].cpu().unsqueeze(0)   # (1, D)
                upper = self.bounds[:, 1].cpu().unsqueeze(0)   # (1, D)

                # tiny epsilon relative to span (avoid zero eps)
                span = (upper - lower).clamp(min=1e-12)
                eps = (span / float(self.grid_size + 1)) * 1e-3   # small fraction of grid step

                # clamp positions strictly inside (lower + eps, upper - eps)
                grid_pos_safe = torch.clamp(grid_pos_t, lower + eps, upper - eps)

                grid_metrics = self.manifold.metric_tensor(grid_pos_safe)  # (n_grid, D, D)
                grid_metrics = grid_metrics.cpu().numpy()

                if n_query > 0:
                    query_pos_t = query_points.cpu()
                    query_pos_safe = torch.clamp(query_pos_t, lower + eps, upper - eps)
                    query_metrics = self.manifold.metric_tensor(query_pos_safe)
                    query_metrics = query_metrics.cpu().numpy()
                else:
                    query_metrics = None
        else:
            grid_metrics = None
            query_metrics = None

        # --- Original graph edges ---
        orig_edges = self.edge_index
        if orig_edges is not None and orig_edges.shape[1] > 0:
            u = orig_edges[0].cpu().numpy().astype(int)
            v = orig_edges[1].cpu().numpy().astype(int)

            if weight_type == "geodesic" and getattr(self, "edge_attr", None) is not None and len(self.edge_attr) == len(u):
                # simplest and fastest: use precomputed self.edge_attr (metric-aware) if available
                weights = self.edge_attr.cpu().numpy().astype(float)
                iterator = range(len(u))
                if show_progress:
                    iterator = tqdm(iterator, desc="Building adjacency (original edges)", dynamic_ncols=True)
                for idx in iterator:
                    uu = int(u[idx]); vv = int(v[idx]); w = float(weights[idx])
                    adjacency.setdefault(uu, []).append((vv, w))
                    adjacency.setdefault(vv, []).append((uu, w))
            elif weight_type == "geodesic":
                # fallback: compute geodesic for each original edge using cached grid_metrics
                iterator = range(len(u))
                if show_progress:
                    iterator = tqdm(iterator, desc="Building adjacency (orig edges geodesic)", dynamic_ncols=True)
                for idx in iterator:
                    uu = int(u[idx]); vv = int(v[idx])
                    pos_u = pos_cpu[uu]
                    pos_v = pos_cpu[vv]
                    dx = pos_v - pos_u  # numpy (D,)
                    g_u = grid_metrics[uu]
                    g_v = grid_metrics[vv]
                    g_avg = 0.5 * (g_u + g_v)
                    q = float(dx.dot(g_avg).dot(dx))
                    q = max(q, 1e-12)
                    w = float(math.sqrt(q))
                    adjacency.setdefault(uu, []).append((vv, w))
                    adjacency.setdefault(vv, []).append((uu, w))
            else:
                # Euclidean fallback (original behavior)
                diffs = pos_cpu[v] - pos_cpu[u]
                weights = np.linalg.norm(diffs, axis=1).astype(float)
                iterator = range(len(u))
                if show_progress:
                    iterator = tqdm(iterator, desc="Building adjacency (original edges euclidean)", dynamic_ncols=True)
                for idx in iterator:
                    uu = int(u[idx]); vv = int(v[idx]); w = float(weights[idx])
                    adjacency.setdefault(uu, []).append((vv, w))
                    adjacency.setdefault(vv, []).append((uu, w))
        else:
            if show_progress:
                tqdm([], desc="Building adjacency (no edges)", dynamic_ncols=True)

        # --- Query -> Grid connections ---
        if show_progress:
            qbar = tqdm(range(n_query), desc="Query->Grid connections", dynamic_ncols=True)
        else:
            qbar = range(n_query)

        # distances (nq, n_grid) on cpu (euclidean) used only to choose neighbors, not necessarily as weights
        dists = torch.cdist(query_points, self.node_positions).cpu()  # (nq, n_grid)
        for qi in qbar:
            row = dists[qi]
            valid_mask = row <= connection_radius
            if valid_mask.any():
                valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
                if valid_idx.numel() > max_grid_neighbors:
                    vd = row[valid_idx]
                    topk_idx = torch.topk(vd, max_grid_neighbors, largest=False)[1]
                    chosen = valid_idx[topk_idx].cpu().numpy().astype(int)
                else:
                    chosen = valid_idx.cpu().numpy().astype(int)
            else:
                chosen = torch.topk(row, min(max_grid_neighbors, row.shape[0]), largest=False)[1].cpu().numpy().astype(int)

            q_node = n_grid + qi
            adjacency.setdefault(q_node, [])
            for g in chosen:
                # compute weight according to weight_type
                if weight_type == "geodesic":
                    # use grid_metrics[g] and query_metrics[qi]
                    pos_u = pos_cpu[g]
                    pos_v = pos_cpu[q_node]
                    dx = pos_v - pos_u
                    g_grid = grid_metrics[g]
                    g_q = query_metrics[qi]
                    # g_avg = 0.5 * (g_grid + g_q)
                    # qval = float(dx.dot(g_avg).dot(dx))
                    # qval = max(qval, 1e-12)
                    # w = float(math.sqrt(qval))
                    # quadratic forms at endpoints
                    q0 = float(dx.dot(g_grid).dot(dx))
                    q1 = float(dx.dot(g_q).dot(dx))

                    eps = 1e-12
                    f0 = math.sqrt(max(q0, eps))
                    f1 = math.sqrt(max(q1, eps))

                    # trapezoid-on-integrand (recommended)
                    w = float(0.5 * (f0 + f1))
                else:
                    # euclidean distance
                    w = float(row[g].item())
                adjacency.setdefault(g, []).append((q_node, w))
                adjacency[q_node].append((g, w))

        # --- Query -> Query connections ---
        if n_query > 1:
            dqq = torch.cdist(query_points, query_points).cpu().numpy()
            grid_spacing = float(torch.mean(self.manifold.steps).cpu().numpy())
            thresh = 1.5 * grid_spacing
            iu, ju = np.triu_indices(n_query, k=1)
            close_pairs = [(int(i), int(j)) for (i, j) in zip(iu, ju) if dqq[i, j] <= thresh]
            if show_progress:
                close_iter = tqdm(close_pairs, desc="Query->Query connections", dynamic_ncols=True)
            else:
                close_iter = close_pairs
            for i, j in close_iter:
                qi = n_grid + i
                qj = n_grid + j
                if weight_type == "geodesic":
                    pos_i = pos_cpu[qi]
                    pos_j = pos_cpu[qj]
                    dx = pos_j - pos_i
                    g_i = query_metrics[i]
                    g_j = query_metrics[j]
                    # g_avg = 0.5 * (g_i + g_j)
                    # qval = float(dx.dot(g_avg).dot(dx))
                    # qval = max(qval, 1e-12)
                    # w = float(math.sqrt(qval))

                    # quadratic forms at endpoints
                    q0 = float(dx.dot(g_i).dot(dx))
                    q1 = float(dx.dot(g_j).dot(dx))

                    eps = 1e-12
                    f0 = math.sqrt(max(q0, eps))
                    f1 = math.sqrt(max(q1, eps))

                    # trapezoid-on-integrand (recommended)
                    w = float(0.5 * (f0 + f1))
                else:
                    w = float(dqq[i, j])
                adjacency.setdefault(qi, []).append((qj, w))
                adjacency.setdefault(qj, []).append((qi, w))
        else:
            if show_progress:
                tqdm([], desc="Query->Query connections", dynamic_ncols=True)

        return {
            'adjacency': adjacency,
            'all_positions': all_positions,
            'n_total_nodes': n_grid + n_query,
            'query_node_start': n_grid
        }

    ####################################################################
    # Parallel Dijkstra-based pathfinding (runs in processes by default)
    ####################################################################
    def _dijkstra_single_source(self, adjacency: Dict[int, List[Tuple[int, float]]], source: int, targets_set: set) -> Dict[int, List[int]]:
        """
        Pure-Python Dijkstra that returns paths from source to any node in targets_set.
        Returns dict: target_node -> path_list (node indices)
        This function is suitable for running in a separate process.
        """
        # adjacency structure: {node: [(neighbor, weight), ...], ...}
        dist = {}
        prev = {}
        visited = set()
        heap = [(0.0, source)]
        dist[source] = 0.0

        targets_remaining = set(targets_set)
        results = {}

        while heap and targets_remaining:
            d_u, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if d_u > dist.get(u, float('inf')):
                continue

            if u in targets_remaining:
                # reconstruct path u <- ... <- source (reverse)
                path = []
                cur = u
                while True:
                    path.append(cur)
                    if cur == source:
                        break
                    cur = prev[cur]
                path.reverse()
                results[u] = path
                targets_remaining.remove(u)
                if not targets_remaining:
                    break

            for v, w in adjacency.get(u, []):
                nd = d_u + float(w)
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        # For any target not reached, store None
        for t in targets_set:
            if t not in results:
                results[t] = None
        return results

    def _find_shortest_paths_parallel(self, graph_info: Dict[str, Any], n_query: int,
                                  workers: int = 4, use_processes: bool = True) -> Dict[tuple, List[int]]:
        """
        Robust parallel single-source Dijkstra. Tries to use ProcessPoolExecutor (no GIL),
        but falls back to ThreadPoolExecutor on environments where process workers cannot
        import the worker function (e.g. notebooks / __main__ pickling issue), or if any
        process-level errors occur.

        Returns: paths dict mapping (i, j) -> path (list) or None.
        """
        adjacency = graph_info['adjacency']
        qstart = graph_info['query_node_start']
        query_nodes = [qstart + i for i in range(n_query)]
        target_sets = [set(query_nodes) - {qn} for qn in query_nodes]

        paths: Dict[tuple, List[int]] = {}

        # Detect interactive / notebook environments where ProcessPool often fails to import __main__:
        running_in_ipython = False
        try:
            # get_ipython exists in IPython/Jupyter kernels
            running_in_ipython = 'get_ipython' in globals() or ('IPython' in sys.modules and hasattr(sys.modules['IPython'], 'get_ipython') and sys.modules['IPython'].get_ipython() is not None)
        except Exception:
            running_in_ipython = False

        prefer_processes = bool(use_processes) and (not running_in_ipython)

        max_workers = min(workers, max(1, len(query_nodes)))

        # Helper to run via executor and collect results with tqdm
        def run_with_executor(executor, worker_fn, exec_name="executor"):
            futures_map = {}
            for i, (src, tgt_set) in enumerate(zip(query_nodes, target_sets)):
                # submit (worker_fn, adjacency, src, tgt_set)
                futures_map[executor.submit(worker_fn, adjacency, src, tgt_set)] = (i, src)
            with tqdm(total=len(futures_map), desc=f"Pathfinding ({exec_name})", dynamic_ncols=True) as pbar:
                for fut in as_completed(list(futures_map.keys())):
                    idx, src = futures_map[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:
                        # catch worker failure (including pickling / process death)
                        print(f"Pathfinding worker failed for source {src}: {exc}")
                        # show a short traceback for debugging (but don't crash)
                        tb = traceback.format_exc()
                        print(tb)
                        res = {t: None for t in target_sets[idx]}
                    i_src = src - qstart
                    for target_node, path in res.items():
                        j_tgt = target_node - qstart
                        if path is None:
                            paths[(i_src, j_tgt)] = None
                        else:
                            paths[(i_src, j_tgt)] = path
                    pbar.update(1)

        # 1) Try ProcessPoolExecutor if it's reasonable
        if prefer_processes:
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    # Use top-level function dijkstra_worker (must be importable)
                    run_with_executor(ex, dijkstra_worker, exec_name="parallel Dijkstra (processes)")
                # success; ensure diagonal entries
                for i in range(n_query):
                    paths[(i, i)] = [qstart + i]
                return paths

            except Exception as e:
                # If process spawning / pickling fails -> fall back to threads
                print("Process-based pathfinding failed or raised during startup:")
                print(f"  {type(e).__name__}: {e}")
                # If the specific error mentions missing attribute in __main__, give a hint
                msg = str(e)
                if "Can't get attribute 'dijkstra_worker'" in msg or "can't get attribute" in msg or running_in_ipython:
                    print("This environment cannot reliably spawn worker processes for top-level functions "
                        "(common in notebooks or interactive shells). Falling back to threaded execution.")
                else:
                    print("Falling back to threaded execution. If you run this script from a file (python script.py), "
                        "process-based execution may be faster.")
                # fall-through to threaded executor below

        # 2) Threaded fallback (safe in all environments, but subject to GIL)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            run_with_executor(ex, self._dijkstra_single_source, exec_name="threaded Dijkstra")
        # ensure diagonal entries
        for i in range(n_query):
            paths[(i, i)] = [qstart + i]

        return paths


    ####################################################################
    # recompute distances (must preserve gradients) - improved chunking and tqdm
    ####################################################################
    def _recompute_path_distances_multithreaded(self,
                                           paths: dict,
                                           graph_info: dict,
                                           query_points: torch.Tensor,
                                           weight_type: str,
                                           n_query: int,
                                           num_threads: int) -> torch.Tensor:
        """
        Memory-efficient, batched recomputation of path distances with autograd support.

        Replaces previous thread/future approach. Processes (i,j) pairs in chunks,
        builds all edges for a chunk, computes all their weights in a batch (vectorized),
        then scatter-adds per-path sums back into the distance matrix.

        - paths: dict[(i,j)] -> list of node indices (global indices, includes grid + query nodes) or None
        - graph_info['query_node_start'] gives offset where query nodes begin
        - num_threads argument retained for API but not used for concurrency here (we run in-process).
        """
        device = self.device
        query_start = graph_info['query_node_start']

        # allocate distance matrix (float, on device)
        distance_matrix = torch.full((n_query, n_query), float('inf'), device=device)

        # Build list of unique upper-triangle pairs (i<=j) to exploit symmetry and avoid double work
        pairs = [(i, j) for i in range(n_query) for j in range(i, n_query)]
        if len(pairs) == 0:
            return distance_matrix

        # If extremely many pairs, choose a conservative chunk size; you can tune this.
        # This parameter bounds how many (i,j) pairs we accumulate before vectorizing their edges.
        max_pairs_per_chunk = 64  # default, lower => lower mem usage; raise for speed if you have RAM/GPU mem.

        # You can make chunk size adapt to graph size: smaller when many pairs
        # For bigger problems reduce this number.
        n_pairs = len(pairs)
        n_chunks = (n_pairs + max_pairs_per_chunk - 1) // max_pairs_per_chunk

        # tqdm over chunks (this produces a visible progress bar)
        chunk_iter = (pairs[k*max_pairs_per_chunk:(k+1)*max_pairs_per_chunk] for k in range(n_chunks))
        with tqdm(total=n_pairs, desc="Recomputing distances (batched)", dynamic_ncols=True) as pbar:
            for chunk_pairs in chunk_iter:
                # For this chunk we will collect all edges for all paths in chunk
                edges_u = []
                edges_v = []
                edge_pair_idx = []  # for each edge, which pair index in chunk it belongs to
                pair_indices = []   # map local pair index -> global (i,j)

                for local_idx, (i, j) in enumerate(chunk_pairs):
                    pair_indices.append((i, j))
                    path = paths.get((i, j), None)
                    if path is None:
                        # mark as infinite (we'll skip edges assembly)
                        continue
                    # If self-path, zero and continue
                    if len(path) <= 1:
                        continue
                    # collect edges along path in order
                    for k in range(len(path) - 1):
                        u = int(path[k])
                        v = int(path[k+1])
                        edges_u.append(u)
                        edges_v.append(v)
                        edge_pair_idx.append(local_idx)

                if len(edges_u) == 0:
                    # nothing computed in this chunk; set diag zeros and update progress
                    for local_idx, (i, j) in enumerate(chunk_pairs):
                        if i == j:
                            distance_matrix[i, j] = 0.0
                    pbar.update(len(chunk_pairs))
                    continue

                # convert to tensors on device
                edges_u = torch.tensor(edges_u, dtype=torch.long, device=device)
                edges_v = torch.tensor(edges_v, dtype=torch.long, device=device)
                edge_pair_idx = torch.tensor(edge_pair_idx, dtype=torch.long, device=device)  # shape (E_chunk,)

                # Gather positions for each edge endpoint (batched). Preserve grad for query points.
                # If node index >= query_start -> from query_points (which may require grad)
                # else from self.node_positions (constants)
                def gather_pos(idx_tensor):
                    # idx_tensor is (E_chunk,) long, values in 0..(num_nodes + n_query - 1)
                    # we pick positions elementwise
                    # boolean mask for query nodes
                    is_query = idx_tensor >= query_start
                    pos_list = []
                    if is_query.any():
                        # query indices (relative)
                        q_idx = (idx_tensor[is_query] - query_start).long()
                        pos_q = query_points[q_idx]  # (n_q_selected, D) - keeps grad if query_points require grad
                    else:
                        pos_q = None
                    if (~is_query).any():
                        g_idx = idx_tensor[~is_query].long()
                        pos_g = self.node_positions[g_idx]  # constants
                    else:
                        pos_g = None

                    # Now we need to reconstruct positions in the original order.
                    # Create an empty tensor and scatter into it.
                    D = self.node_positions.shape[1]
                    pos = torch.empty((idx_tensor.shape[0], D), device=device, dtype=self.node_positions.dtype)
                    if pos_q is not None:
                        pos[is_query] = pos_q
                    if pos_g is not None:
                        pos[~is_query] = pos_g
                    return pos

                pos_u = gather_pos(edges_u)  # (E_chunk, D)
                pos_v = gather_pos(edges_v)  # (E_chunk, D)
                dx = pos_v - pos_u  # (E_chunk, D)

                # Compute weights in batch
                if weight_type == "geodesic":
                    # metric_tensor should accept (E_chunk, D) -> (E_chunk, D, D)
                    g_u = self.manifold.metric_tensor(pos_u)
                    g_v = self.manifold.metric_tensor(pos_v)
                    g_avg = (g_u + g_v) * 0.5
                    # quadratic forms: (E,1,1) -> squeeze to (E,)
                    qf = torch.bmm(torch.bmm(dx.unsqueeze(1), g_avg), dx.unsqueeze(2)).squeeze()
                    qf = torch.clamp(qf, min=1e-12)
                    weights = torch.sqrt(qf)
                elif weight_type == "euclidean":
                    weights = torch.norm(dx, dim=1)
                else:  # unit
                    weights = torch.ones((dx.shape[0],), device=device, dtype=dx.dtype)

                # Sum edge weights per pair using scatter_add
                n_local_pairs = len(chunk_pairs)
                totals = torch.zeros((n_local_pairs,), device=device, dtype=weights.dtype)
                totals = totals.scatter_add(0, edge_pair_idx, weights)

                # Assign totals to distance matrix for each pair (and symmetric counterpart)
                for local_idx, (i, j) in enumerate(chunk_pairs):
                    # if path is None -> infinite; if i==j -> zero
                    path = paths.get((i, j), None)
                    if path is None:
                        distance_matrix[i, j] = float('inf')
                        distance_matrix[j, i] = float('inf')
                    elif i == j:
                        distance_matrix[i, j] = 0.0
                    else:
                        # totals[local_idx] contains sum for this pair (could be zero if path length 0)
                        val = totals[local_idx]
                        distance_matrix[i, j] = val
                        distance_matrix[j, i] = val

                # free big temporaries as soon as possible
                del edges_u, edges_v, edge_pair_idx, pos_u, pos_v, dx, weights, totals, g_u, g_v, g_avg
                if torch.cuda.is_available():
                    # free any cached memory (helpful for long runs)
                    torch.cuda.empty_cache()

                pbar.update(len(chunk_pairs))

        return distance_matrix

    def _recompute_path_distances_with_gradients(self,
                                               paths: dict,
                                               graph_info: dict,
                                               query_points: torch.Tensor,
                                               weight_type: str,
                                               n_query: int) -> torch.Tensor:
        # fallback single-threaded (unchanged behaviour)
        distance_matrix = torch.zeros((n_query, n_query), device=self.device)
        query_start = graph_info['query_node_start']
        for i in range(n_query):
            for j in range(n_query):
                if i == j:
                    distance_matrix[i, j] = 0.0
                    continue
                path = paths[(i, j)]
                if path is None:
                    distance_matrix[i, j] = float('inf')
                    continue
                total_distance = torch.tensor(0.0, device=self.device)
                for k in range(len(path) - 1):
                    node_u = path[k]; node_v = path[k+1]
                    if node_u >= query_start:
                        pos_u = query_points[node_u - query_start]
                    else:
                        pos_u = self.node_positions[node_u]
                    if node_v >= query_start:
                        pos_v = query_points[node_v - query_start]
                    else:
                        pos_v = self.node_positions[node_v]
                    if weight_type == "geodesic":
                        edge_weight = self._compute_geodesic_weight_with_gradients(pos_u, pos_v)
                    elif weight_type == "euclidean":
                        edge_weight = torch.norm(pos_v - pos_u)
                    else:
                        edge_weight = torch.tensor(1.0, device=self.device)
                    total_distance = total_distance + edge_weight
                distance_matrix[i, j] = total_distance
        return distance_matrix

    def _compute_geodesic_weight_with_gradients(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        try:
            if pos1.dim() == 1:
                pos1 = pos1.unsqueeze(0)
            if pos2.dim() == 1:
                pos2 = pos2.unsqueeze(0)
            g1 = self.manifold.metric_tensor(pos1)
            g2 = self.manifold.metric_tensor(pos2)
            g_avg = (g1 + g2) * 0.5
            dx = pos2 - pos1
            q = torch.bmm(torch.bmm(dx.unsqueeze(-2), g_avg), dx.unsqueeze(-1)).squeeze()
            q = torch.clamp(q, min=1e-12)
            return torch.sqrt(q)
        except Exception:
            return torch.norm(pos2.squeeze() - pos1.squeeze())

    def _compute_geodesic_weight(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        try:
            g1 = self.manifold.metric_tensor(pos1)
            g2 = self.manifold.metric_tensor(pos2)
            g_avg = (g1 + g2) * 0.5
            dx = pos2 - pos1
            w = torch.sqrt(torch.clamp(torch.bmm(torch.bmm(dx.unsqueeze(-2), g_avg), dx.unsqueeze(-1)).squeeze(), min=1e-12))
            return w
        except Exception:
            return torch.norm(pos2 - pos1, dim=-1)

    def _create_extended_graph(self,
                             query_to_grid_edges: torch.Tensor,
                             query_to_grid_weights: torch.Tensor,
                             query_to_query_edges: torch.Tensor,
                             query_to_query_weights: torch.Tensor,
                             n_query: int,
                             weight_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        original_edges = self.edge_index
        if weight_type == "geodesic":
            original_weights = self.edge_attr
        else:
            original_weights = self.get_edge_weights(weight_type)
        all_edges = [original_edges]
        all_weights = [original_weights]
        if query_to_grid_edges.shape[1] > 0:
            all_edges.append(query_to_grid_edges)
            all_weights.append(query_to_grid_weights)
        if query_to_query_edges.shape[1] > 0:
            all_edges.append(query_to_query_edges)
            all_weights.append(query_to_query_weights)
        extended_edge_index = torch.cat(all_edges, dim=1)
        extended_edge_weights = torch.cat(all_weights, dim=0)
        return extended_edge_index, extended_edge_weights

    def _compute_distance_matrix(self,
                               extended_edge_index: torch.Tensor,
                               extended_edge_weights: torch.Tensor,
                               n_query: int) -> torch.Tensor:
        try:
            from torch_geometric.utils import dijkstra
            query_indices = torch.arange(self.num_nodes, self.num_nodes + n_query, device=self.device)
            distance_matrix = torch.full((n_query, n_query), float('inf'), device=self.device)
            for i, source_idx in enumerate(query_indices):
                distances = dijkstra(extended_edge_index, extended_edge_weights, source_idx, num_nodes=self.num_nodes + n_query)
                for j, target_idx in enumerate(query_indices):
                    if i != j:
                        distance_matrix[i, j] = distances[target_idx]
                    else:
                        distance_matrix[i, j] = 0.0
            return distance_matrix
        except ImportError:
            return self._compute_distance_matrix_networkx(extended_edge_index, extended_edge_weights, n_query)
        except Exception as e:
            print(f"Warning: Could not compute shortest paths: {e}")
            return self._compute_direct_distances(n_query)

    def _compute_distance_matrix_networkx(self,
                                        extended_edge_index: torch.Tensor,
                                        extended_edge_weights: torch.Tensor,
                                        n_query: int) -> torch.Tensor:
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("Neither torch_geometric.utils.dijkstra nor networkx is available")
        edges_np = extended_edge_index.cpu().numpy()
        weights_np = extended_edge_weights.cpu().numpy()
        G = nx.Graph()
        for i in range(edges_np.shape[1]):
            u, v = int(edges_np[0, i]), int(edges_np[1, i])
            w = float(weights_np[i])
            G.add_edge(u, v, weight=w)
        query_indices = list(range(self.num_nodes, self.num_nodes + n_query))
        distance_matrix = torch.full((n_query, n_query), float('inf'), device=self.device)
        for i, s in enumerate(query_indices):
            try:
                lengths = nx.single_source_dijkstra_path_length(G, s, weight='weight')
                for j, t in enumerate(query_indices):
                    if t in lengths:
                        distance_matrix[i, j] = lengths[t]
                    elif i == j:
                        distance_matrix[i, j] = 0.0
            except nx.NetworkXNoPath:
                if i == j:
                    distance_matrix[i, j] = 0.0
        return distance_matrix

    def get_stats(self) -> dict:
        stats = {
            "num_nodes": self.num_nodes,
            "num_edges": int(self.edge_index.shape[1]) if (self.edge_index is not None) else 0,
            "grid_dimensions": self.n_dims,
            "grid_size": self.grid_size,
            "connectivity": self.connectivity,
            "average_degree": (2 * self.edge_index.shape[1] / self.num_nodes) if (self.edge_index is not None and self.num_nodes > 0) else 0
        }
        if self.edge_attr is not None and len(self.edge_attr) > 0:
            stats.update({
                "edge_weight_mean": float(self.edge_attr.mean()),
                "edge_weight_std": float(self.edge_attr.std()),
                "edge_weight_min": float(self.edge_attr.min()),
                "edge_weight_max": float(self.edge_attr.max())
            })
        return stats
