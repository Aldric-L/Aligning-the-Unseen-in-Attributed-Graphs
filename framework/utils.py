import torch
import numpy as np
from scipy.spatial.distance import cosine # For cosine similarity

from framework.GraphVAE import GraphVAE

def adj_matrix_to_edge_index(adj_matrix):
    """
    Convert an adjacency matrix to edge index format for PyTorch Geometric.
    
    Args:
        adj_matrix: A square adjacency matrix as torch.Tensor or numpy.ndarray
                   Can be binary or weighted
    
    Returns:
        edge_index: Tensor of shape [2, num_edges] containing the edge indices
        edge_attr: Tensor of shape [num_edges] containing the edge weights
                  (only if adj_matrix is weighted, otherwise None)
    """
    # Convert to torch tensor if numpy array
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = torch.from_numpy(adj_matrix).float()
    
    # Make sure it's a square matrix
    assert adj_matrix.shape[0] == adj_matrix.shape[1], "Adjacency matrix must be square"
    
    # Get non-zero entries
    edge_index = adj_matrix.nonzero(as_tuple=False).t()
    
    # Check if the matrix is weighted (non-binary)
    is_weighted = not torch.all(torch.eq(adj_matrix[adj_matrix > 0], 1.0))
    
    if is_weighted:
        # Extract edge weights
        edge_attr = adj_matrix[edge_index[0], edge_index[1]]
        return edge_index, edge_attr
    else:
        # Binary adjacency matrix, no edge attributes
        return edge_index, None
    
# Heat kernel (diffusion distance)
# def heat_kernel_distance(adj_matrix1, adj_matrix2, t=1.0):
#     """
#     Compute the heat kernel distance between two graphs.
#     t is the diffusion time parameter.
#     """
#     device = adj_matrix1.device
    
#     # Normalize adjacency matrices
#     def normalize_adj(adj):
#         # Add self-loops
#         adj_with_self_loops = adj + torch.eye(adj.size(0), device=device)
#         # Compute degree matrix
#         degree = torch.sum(adj_with_self_loops, dim=1)
#         # Compute D^(-1/2)
#         d_inv_sqrt = torch.pow(degree, -0.5)
#         d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
#         d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
#         # Compute normalized adjacency
#         return torch.mm(torch.mm(d_mat_inv_sqrt, adj_with_self_loops), d_mat_inv_sqrt)
    
#     # Normalized adjacency matrices
#     norm_adj1 = normalize_adj(adj_matrix1)
#     norm_adj2 = normalize_adj(adj_matrix2)
    
#     # Heat kernel: H(t) = exp(-t(I-A))
#     heat_kernel1 = torch.matrix_exp(-t * (torch.eye(adj_matrix1.size(0), device=device) - norm_adj1))
#     heat_kernel2 = torch.matrix_exp(-t * (torch.eye(adj_matrix2.size(0), device=device) - norm_adj2))
    
#     # Compute Frobenius norm between heat kernels
#     distance = torch.norm(heat_kernel1 - heat_kernel2, p='fro')
    
#     return distance

def get_adjacency_matrix_from_tensors(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor
) -> torch.Tensor:
    """
    Constructs a dense adjacency matrix from edge indices and edge weights.

    Assumes a 0-based indexing for nodes.
    Assumes the graph is undirected (populates both A[i, j] and A[j, i]).
    Handles potential duplicate edges by summing weights.

    Args:
        edge_index: A PyTorch tensor of shape (2, num_edges) representing the
                    source and target nodes of each edge.
        edge_weight: A PyTorch tensor of shape (num_edges,) or (1, num_edges)
                     representing the weight of each edge.

    Returns:
        A dense PyTorch tensor representing the adjacency matrix of shape
        (num_nodes, num_nodes).
    """
    # Squeeze edge_weight if it has a shape of (1, num_edges)
    if edge_weight.dim() == 2 and edge_weight.shape[0] == 1:
        edge_weight = edge_weight.squeeze(0)
    elif edge_weight.dim() != 1:
         raise ValueError("edge_weight must have shape (num_edges,) or (1, num_edges)")

    if edge_index.shape[1] != edge_weight.shape[0]:
         raise ValueError("Number of edges in edge_index and edge_weight must match")

    # Determine the number of nodes based on the maximum node index
    num_nodes = int(edge_index.max().item()) + 1

    # For an undirected graph, add both (u, v) and (v, u) indices
    # This also handles cases where edge_index might contain duplicates
    # or both directions explicitly by summing weights via sparse tensor.
    sparse_indices = torch.cat([edge_index, torch.flip(edge_index, dims=[0])], dim=1)
    sparse_values = torch.cat([edge_weight, edge_weight], dim=0)

    # Create a sparse COO tensor. This is efficient and handles summing
    # weights for duplicate (row, col) indices automatically.
    adj_matrix_sparse = torch.sparse_coo_tensor(
        sparse_indices,
        sparse_values,
        (num_nodes, num_nodes),
        dtype=edge_weight.dtype # Use the dtype of the weights
    )

    # Convert the sparse tensor to a dense tensor
    adj_matrix_dense = adj_matrix_sparse.to_dense()

    return adj_matrix_dense


def read_matrix_from_csv_loadtxt(filepath, delimiter=',', dtype=float):
  """
  Reads a NumPy matrix from a CSV file using np.loadtxt().

  Args:
    filepath (str): The path to the CSV file.
    delimiter (str): The character separating values in the CSV file (default is comma).

  Returns:
    numpy.ndarray: The matrix read from the CSV file.
  """
  try:
    matrix = np.loadtxt(filepath, delimiter=delimiter, dtype=dtype)
    print(f"Successfully loaded matrix from {filepath} using np.loadtxt().")
    return matrix
  except FileNotFoundError:
    print(f"Error: The file '{filepath}' was not found.")
    return None
  except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    return None
  

def compute_node_feature_assortativity(G, node_features):
    """
    Computes a node-level assortativity-like score based on d-dimensional features.
    For each node, it calculates the average similarity to its neighbors
    and compares it to the average similarity to all other nodes.

    Args:
        adj_matrix (np.ndarray): The adjacency matrix of the graph (N x N).
        node_features (np.ndarray): A (N x D) array where N is the number of nodes
                                     and D is the dimensionality of features.

    Returns:
        dict: A dictionary where keys are node IDs (integers) and values are
              their respective node-level assortativity scores.
              A higher positive score indicates more assortative mixing for that node.
              A negative score indicates disassortative mixing for that node.
    """
    num_nodes = G.number_of_nodes()
    if node_features.shape[0] != num_nodes:
        raise ValueError("Number of nodes in adjacency matrix and features must match.")

    # 1. Create a NetworkX graph from the adjacency matrix

    # Add features to nodes
    for i in range(num_nodes):
        G.nodes[i]['features'] = node_features[i]

    def vector_similarity(vec1, vec2):
        # Handle cases where vectors might be identical to avoid cosine(0,0) issues
        if np.array_equal(vec1, vec2):
            return 1.0
        return 1 - cosine(vec1, vec2)

    node_assortativity_scores = {}

    for node_id in G.nodes():
        node_feat = G.nodes[node_id]['features']
        neighbors = list(G.neighbors(node_id))

        if not neighbors:
            # A node with no neighbors can't have neighbor similarity
            node_assortativity_scores[node_id] = 0.0 # Or np.nan, depending on desired behavior
            continue

        # Calculate average similarity to neighbors
        neighbor_similarities = []
        for neighbor_id in neighbors:
            neighbor_feat = G.nodes[neighbor_id]['features']
            neighbor_similarities.append(vector_similarity(node_feat, neighbor_feat))
        avg_neighbor_similarity = np.mean(neighbor_similarities)

        # Calculate average similarity to all other nodes (baseline)
        all_other_node_similarities = []
        for other_node_id in G.nodes():
            if other_node_id != node_id: # Don't compare a node to itself
                other_node_feat = G.nodes[other_node_id]['features']
                all_other_node_similarities.append(vector_similarity(node_feat, other_node_feat))

        if not all_other_node_similarities: # Should only happen with single node graphs
            baseline_avg_similarity = avg_neighbor_similarity # Fallback if no other nodes
        else:
            baseline_avg_similarity = np.mean(all_other_node_similarities)

        # Compute the node's assortativity score
        score = avg_neighbor_similarity - baseline_avg_similarity
        node_assortativity_scores[node_id] = score

    return node_assortativity_scores

def compute_curvature_change(model1: GraphVAE, model2: GraphVAE):
    model1.get_latent_manifold().compute_full_grid_metric_tensor()
    model2.get_latent_manifold().compute_full_grid_metric_tensor()
    resolution = 30
    bounds_np = model1.get_latent_manifold().get_bounds().cpu().numpy()
    plot_z1 = np.linspace(bounds_np[0, 0], bounds_np[0, 1], resolution)
    plot_z2 = np.linspace(bounds_np[1, 0], bounds_np[1, 1], resolution)

    Z1_np, Z2_np = np.meshgrid(plot_z1, plot_z2)
    Z1, Z2 = torch.from_numpy(Z1_np), torch.from_numpy(Z2_np)
                
    curvature_phase1 = torch.zeros((resolution, resolution))
    curvature_phase2 = torch.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            z = torch.stack([Z1[i, j], Z2[i, j]])
            clamped_z = model1.get_latent_manifold()._clamp_point_to_bounds(z)
            try:
                curv_val_1 = model1.get_latent_manifold().compute_gaussian_curvature(model1.get_latent_manifold().metric_tensor(clamped_z))
                curvature_phase1[i, j] = curv_val_1
            except (ValueError, RuntimeError) as e:
                print(f"Error computing curvature at point {z}: {e}. Setting to NaN.")
                curvature_phase1[i, j] = torch.nan

            try:
                curv_val_2 = model2.get_latent_manifold().compute_gaussian_curvature(model2.get_latent_manifold().metric_tensor(clamped_z))
                curvature_phase2[i, j] = curv_val_2
            except (ValueError, RuntimeError) as e:
                print(f"Error computing curvature at point {z}: {e}. Setting to NaN.")
                curvature_phase2[i, j] = torch.nan
            
    curvature_diff = (curvature_phase2 - curvature_phase1)/(curvature_phase1 + 1e-8)
    curvature_diff = curvature_diff.detach().numpy()
    curvature_diff -= np.nanmean(curvature_diff)

    return curvature_diff, Z1_np, Z2_np