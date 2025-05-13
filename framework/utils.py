import torch
import numpy as np

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
def heat_kernel_distance(adj_matrix1, adj_matrix2, t=1.0):
    """
    Compute the heat kernel distance between two graphs.
    t is the diffusion time parameter.
    """
    device = adj_matrix1.device
    
    # Normalize adjacency matrices
    def normalize_adj(adj):
        # Add self-loops
        adj_with_self_loops = adj + torch.eye(adj.size(0), device=device)
        # Compute degree matrix
        degree = torch.sum(adj_with_self_loops, dim=1)
        # Compute D^(-1/2)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        # Compute normalized adjacency
        return torch.mm(torch.mm(d_mat_inv_sqrt, adj_with_self_loops), d_mat_inv_sqrt)
    
    # Normalized adjacency matrices
    norm_adj1 = normalize_adj(adj_matrix1)
    norm_adj2 = normalize_adj(adj_matrix2)
    
    # Heat kernel: H(t) = exp(-t(I-A))
    heat_kernel1 = torch.matrix_exp(-t * (torch.eye(adj_matrix1.size(0), device=device) - norm_adj1))
    heat_kernel2 = torch.matrix_exp(-t * (torch.eye(adj_matrix2.size(0), device=device) - norm_adj2))
    
    # Compute Frobenius norm between heat kernels
    distance = torch.norm(heat_kernel1 - heat_kernel2, p='fro')
    
    return distance

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
