import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union, Callable

from framework.utils import heat_kernel_distance, get_adjacency_matrix_from_tensors

class DecoderBase(nn.Module, ABC):
    """
    Abstract base class for decoders to ensure a common interface.
    """
    def __init__(self, latent_dim: int, name: str = "base_decoder"):
        super(DecoderBase, self).__init__()
        self.latent_dim = latent_dim
        self.name = name
    
    @abstractmethod
    def forward(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the decoder
        
        Args:
            z: Latent embedding [num_nodes, latent_dim]
            
        Returns:
            Dict containing outputs and any additional info
        """
        pass
    
    @abstractmethod
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute decoder-specific loss
        
        Args:
            outputs: Outputs from the forward pass
            targets: Target values to compare against
            
        Returns:
            Loss value
        """
        pass

class AdjacencyDecoder(DecoderBase):
    """
    Decoder for reconstructing the adjacency matrix.
    Uses neural networks instead of inner products.
    """
    def __init__(
        self, 
        latent_dim: int, 
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1, 
        activation=nn.ReLU(),
        final_activation: Optional[Callable] = None,
        name: str = "adj_decoder"
    ):
        super(AdjacencyDecoder, self).__init__(latent_dim, name)
        
        self.dropout = dropout
        self.activation = activation
        self.final_activation = final_activation
        
        # Build MLP layers
        layers = []
        in_features = latent_dim * 2  # Concatenated node pairs
        
        for dim in hidden_dims:
            layers.append(nn.Linear(in_features, dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            in_features = dim
        
        layers.append(nn.Linear(in_features, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, edge_index: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Reconstruct adjacency matrix from node embeddings
        
        Args:
            z: Node embeddings [num_nodes, latent_dim]
            edge_index: Optional edge index for sparse reconstruction [2, num_edges]
            
        Returns:
            Dict with adjacency predictions
        """
        batch_size = z.size(0)
        
        if edge_index is not None:
            # Sparse version - only predict for specific edges
            src, dst = edge_index
            z_src = z[src]
            z_dst = z[dst]
            edge_features = torch.cat([z_src, z_dst], dim=1)
            edge_logits = (self.mlp(edge_features)).squeeze(-1)
            
            # Apply final activation if specified
            if self.final_activation is not None:
                edge_logits = self.final_activation(edge_logits)
            
            return {
                "edge_logits": edge_logits,
                "edge_index": edge_index
            }
        else:
            # Dense version - predict full adjacency matrix
            # Get all pairs of nodes
            node_i = torch.arange(batch_size, device=z.device).repeat_interleave(batch_size)
            node_j = torch.arange(batch_size, device=z.device).repeat(batch_size)
            
            # Concatenate embeddings for each node pair
            z_i = z[node_i]
            z_j = z[node_j]
            pair_features = torch.cat([z_i, z_j], dim=1)
            
            # Predict all pairwise edge probabilities
            edge_logits = (self.mlp(pair_features)).squeeze(-1)
            adj_logits = edge_logits.view(batch_size, batch_size)
            
            # Apply final activation if specified
            if self.final_activation is not None:
                adj_logits = self.final_activation(adj_logits)
            
            return {
                "adj_logits": adj_logits
            }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], lambda_variance: float = 0.1) -> torch.Tensor:
        """
        Compute BCE loss for adjacency reconstruction
        
        Args:
            outputs: Output from forward pass
            targets: Target adjacency matrix or edge labels
            
        Returns:
            BCE loss
        """
        if "adj_logits" not in outputs:
            # Sparse version
            edge_logits = outputs["edge_logits"]
            edge_labels = targets["edge_labels"]

            loss = F.smooth_l1_loss(edge_logits, edge_labels) + 2*F.mse_loss(edge_logits, edge_labels) 
            + lambda_variance*(
                #F.mse_loss(torch.var(edge_logits), torch.var(edge_labels)) 
                               #+ 
                               0.01 * F.mse_loss(torch.sum(edge_logits) , torch.sum(edge_labels)))
            loss += heat_kernel_distance(get_adjacency_matrix_from_tensors(targets["edge_index"], edge_labels), get_adjacency_matrix_from_tensors(targets["edge_index"], edge_logits))
        else:
            # Dense version
            adj_logits = outputs["adj_logits"]
            adj_target = targets["adj_matrix"]
        
            # Mask diagonal if needed
            mask = ~torch.eye(adj_logits.shape[0], dtype=torch.bool, device=adj_logits.device)
            adj_logits_masked = adj_logits[mask]
            adj_target_masked = adj_target[mask]
            loss = F.smooth_l1_loss(adj_logits_masked, adj_target_masked) + 2* F.mse_loss(adj_logits_masked, adj_target_masked) 
            + lambda_variance*(
                #F.mse_loss(torch.var(torch.flatten(adj_logits_masked)), torch.var(torch.flatten(adj_target_masked)))
                               #+ 
                            0.01 * F.mse_loss(torch.sum(torch.flatten(adj_logits_masked)) , torch.sum(torch.flatten(adj_target_masked))))
            loss += heat_kernel_distance(adj_logits_masked, adj_target_masked)
        
        return loss

    def compute_jacobian(self, z: torch.Tensor, node_idx: int = None) -> torch.Tensor:
        """
        Compute the Jacobian of the decoder output with respect to the latent space
        
        Args:
            z: Latent variables [num_nodes, latent_dim]
            node_idx: Optional index of the specific node to compute Jacobian for
            
        Returns:
            Jacobian matrix
        """
        # Detach z and require gradients
        z_detached = z.detach().clone().requires_grad_(True)
        
        # If node_idx is provided, focus on edges connected to that node
        batch_size = z_detached.size(0)
        
        if node_idx is not None:
            # Handle edges connected to specific node
            other_nodes = torch.arange(batch_size, device=z.device)
            # Create edge pairs (node_idx, all_other_nodes)
            src = torch.full((batch_size,), node_idx, device=z.device)
            edge_index = torch.stack([src, other_nodes], dim=0)
            
            # Forward pass to get edge predictions
            outputs = self.forward(z_detached, edge_index=edge_index)
            edge_preds = outputs["edge_logits"]
            
            # Initialize Jacobian matrix
            jacobian = torch.zeros(edge_preds.size(0), z.size(1), device=z.device)
            
            # Compute gradient for each edge
            for i in range(edge_preds.size(0)):
                if i == node_idx:  # Skip self-loop if needed
                    continue
                    
                # Zero gradients
                if z_detached.grad is not None:
                    z_detached.grad.zero_()
                
                # Compute gradient of edge prediction w.r.t. latent variables
                edge_preds[i].backward(retain_graph=True)
                
                # Store gradient (which is the Jacobian row)
                jacobian[i] = z_detached.grad[node_idx].clone()
        
        else:
            # Handle all edges
            # Forward pass with dense adjacency matrix
            outputs = self.forward(z_detached)
            adj_logits = outputs["adj_logits"]
            
            # Initialize Jacobian tensor (nodes × nodes × latent_dim)
            jacobian = torch.zeros(batch_size, batch_size, z.size(1), device=z.device)
            
            # For each entry in the adjacency matrix
            for i in range(batch_size):
                for j in range(batch_size):
                    if i == j:  # Skip diagonal elements (self-loops)
                        continue
                        
                    # Zero gradients
                    if z_detached.grad is not None:
                        z_detached.grad.zero_()
                    
                    # Backward for this specific element
                    adj_logits[i, j].backward(retain_graph=True)
                    
                    # Store gradient for both nodes
                    jacobian[i, j] = z_detached.grad[i].clone() + z_detached.grad[j].clone()
        
        return jacobian


class NodeAttributeDecoder(DecoderBase):
    """
    Decoder for reconstructing node attributes.
    """
    def __init__(
        self, 
        latent_dim: int, 
        output_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1, 
        activation=nn.ReLU(),
        final_activation: Optional[Callable] = None,
        name: str = "node_attr_decoder"
    ):
        super(NodeAttributeDecoder, self).__init__(latent_dim, name)
        
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = activation
        self.final_activation = final_activation
        
        # Build MLP layers
        layers = []
        in_features = latent_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(in_features, dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            in_features = dim
        
        layers.append(nn.Linear(in_features, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Reconstruct node attributes from latent embeddings
        
        Args:
            z: Latent node embeddings [num_nodes, latent_dim]
            
        Returns:
            Dict with node feature predictions
        """
        node_features = self.mlp(z)
        
        # Apply final activation if specified
        if self.final_activation is not None:
            node_features = self.final_activation(node_features)
        
        return {
            "node_features": node_features
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], lambda_variance: float = 0.001, epsilon_variance: float = 1e-8) -> torch.Tensor:
        """
        Compute MSE loss for node attribute reconstruction
        
        Args:
            outputs: Output from forward pass
            targets: Target node features
            
        Returns:
            MSE loss
        """
        pred_features = outputs["node_features"]
        target_features = targets["node_features"]
        
        #loss = F.mse_loss(pred_features, target_features)
        # Ensure shapes match and are suitable (at least 2D: batch, features)
        if pred_features.shape != target_features.shape:
             raise ValueError("Predicted and target features must have the same shape")
        if pred_features.dim() < 2:
             raise ValueError("Features tensor must have at least two dimensions (batch, features)")

        # --- 1. Standard MSE Loss ---
        # F.mse_loss computes the mean over all elements (batch and features) by default
        loss_mse = F.mse_loss(pred_features, target_features)

        # --- 2. Variance Penalty ---
        # Calculate the mean for each vector in the batch across the feature dimension (dim=1)
        # keepdim=True is used to maintain the dimension for correct broadcasting
        mean_target = torch.mean(target_features, dim=1, keepdim=True)
        mean_pred = torch.mean(pred_features, dim=1, keepdim=True)

        # Calculate the Sum of Squared Deviations (SSD) from the mean for each vector
        # Sum across the feature dimension (dim=1). The result is (batch_size,)
        ssd_target = torch.sum((target_features - mean_target)**2, dim=1)
        ssd_pred = torch.sum((pred_features - mean_pred)**2, dim=1)

        # Calculate the variance penalty for each item in the batch
        # Add epsilon to the denominator for numerical stability
        variance_penalty_per_item = ssd_target / (ssd_pred + epsilon_variance)

        # Average the variance penalty over the batch to get a single scalar loss
        loss_variance_penalty = torch.mean(variance_penalty_per_item)

        # --- 3. Total Loss ---
        total_loss = loss_mse + lambda_variance * loss_variance_penalty

        return total_loss

    def compute_jacobian(self, z: torch.Tensor, node_idx: int = None) -> torch.Tensor:
        """
        Compute the Jacobian of the decoder output with respect to the latent space
        
        Args:
            z: Latent variables [num_nodes, latent_dim]
            node_idx: Optional index of the specific node to compute Jacobian for
            
        Returns:
            Jacobian matrix
        """
        # Detach z and require gradients
        z_detached = z.detach().clone().requires_grad_(True)
        
        # Forward pass to get node feature predictions
        outputs = self.forward(z_detached)
        node_features = outputs["node_features"]
        
        # Handle single node case
        if node_idx is not None:
            node_features = node_features[node_idx]
            
            # Initialize Jacobian matrix (output_dim × latent_dim)
            jacobian = torch.zeros(self.output_dim, z.size(1), device=z.device)
            
            # Compute Jacobian row by row
            for i in range(self.output_dim):
                # Zero gradients
                if z_detached.grad is not None:
                    z_detached.grad.zero_()
                
                # Backward for this feature dimension
                node_features[i].backward(retain_graph=True)
                
                # Store gradient (Jacobian row)
                jacobian[i] = z_detached.grad[node_idx].clone()
        
        else:
            # Full Jacobian for all nodes
            batch_size = z_detached.size(0)
            
            # Initialize Jacobian tensor (nodes × output_dim × latent_dim)
            jacobian = torch.zeros(batch_size, self.output_dim, z.size(1), device=z.device)
            
            # For each node and feature dimension
            for n in range(batch_size):
                for i in range(self.output_dim):
                    # Zero gradients
                    if z_detached.grad is not None:
                        z_detached.grad.zero_()
                    
                    # Backward for this specific element
                    node_features[n, i].backward(retain_graph=True)
                    
                    # Store gradient
                    jacobian[n, i] = z_detached.grad[n].clone()
        
        return jacobian