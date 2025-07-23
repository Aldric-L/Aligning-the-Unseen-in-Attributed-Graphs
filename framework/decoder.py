import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from tqdm import tqdm
from torch.autograd.functional import jacobian

from framework.utils import heat_kernel_distance, get_adjacency_matrix_from_tensors
from framework.preprocessing import PreprocessingLayer, DistancePreprocessingLayer, JacobianMetricPreprocessingLayer


class DecoderBase(nn.Module, ABC):
    """
    Abstract base class for decoders to ensure a common interface.
    """
    def __init__(self, latent_dim: int, name: str = "base_decoder"):
        super(DecoderBase, self).__init__()
        self.latent_dim = latent_dim
        self.name = name
        # Store custom loss functions that can be added/removed dynamically
        self.custom_losses = {}
    
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
    
    def add_custom_loss(self, name: str, loss_fn: Callable, weight: float = 1.0):
        """
        Add a custom loss function to this decoder
        
        Args:
            name: Name of the custom loss
            loss_fn: Function that takes (outputs, targets, z, **kwargs) and returns loss tensor
            weight: Weight for this loss component
        """
        self.custom_losses[name] = {
            'function': loss_fn,
            'weight': weight,
            'active': True
        }
    
    def remove_custom_loss(self, name: str) -> bool:
        """
        Remove a custom loss function
        
        Args:
            name: Name of the custom loss to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in self.custom_losses:
            del self.custom_losses[name]
            return True
        return False
    
    def set_custom_loss_active(self, name: str, active: bool = True):
        """
        Activate or deactivate a custom loss without removing it
        
        Args:
            name: Name of the custom loss
            active: Whether the loss should be active
        """
        if name in self.custom_losses:
            self.custom_losses[name]['active'] = active
    
    def set_custom_loss_weight(self, name: str, weight: float):
        """
        Update the weight of a custom loss
        
        Args:
            name: Name of the custom loss
            weight: New weight value
        """
        if name in self.custom_losses:
            self.custom_losses[name]['weight'] = weight
    
    def compute_total_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], 
                          z: torch.Tensor = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute total loss including custom losses
        
        Args:
            outputs: Outputs from forward pass
            targets: Target values
            z: Latent variables (needed for some custom losses)
            **kwargs: Additional arguments for custom losses
            
        Returns:
            Dict with total loss and component losses
        """
        # Base reconstruction loss
        base_loss = self.compute_loss(outputs, targets)
        loss_components = {'base_loss': base_loss}
        total_loss = base_loss
        
        # Add custom losses
        for loss_name, loss_info in self.custom_losses.items():
            if not loss_info['active']:
                continue
                
            try:
                custom_loss = loss_info['function'](outputs, targets, z, **kwargs)
                weighted_loss = loss_info['weight'] * custom_loss
                loss_components[f'custom_{loss_name}'] = custom_loss
                loss_components[f'weighted_{loss_name}'] = weighted_loss
                total_loss += weighted_loss
            except Exception as e:
                print(f"Warning: Custom loss '{loss_name}' failed with error: {e}")
                continue
        
        loss_components['total'] = total_loss
        return loss_components

class MLPDecoder(DecoderBase):
    def __init__(self, latent_dim, out_dim, hidden_dims, dropout=0.5,
                 activation=nn.ReLU, name="mlp_decoder"):
        super().__init__(latent_dim, name)
        layers, d = [], latent_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(d, h),
                activation(),
                nn.Dropout(dropout)
            ]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z, **kwargs):
        return {"recon": self.net(z)}

    def compute_loss(self, outputs, targets):
        return F.mse_loss(outputs["recon"], targets["x"])
    
    def compute_jacobian(self, z: torch.Tensor, node_idx: Optional[int] = None) -> torch.Tensor:
        """
        Compute the Jacobian of the decoder output with respect to the latent space.

        Args:
            z: Latent variables.
               - If [num_nodes, latent_dim]: Computes the Jacobian for all nodes.
               - If [latent_dim]: Computes the Jacobian for a single node.

        Returns:
            Jacobian matrix or tensor:
            - If input z was [latent_dim]: [output_dim, latent_dim]
            - Else (input z was [num_nodes, latent_dim]): [num_nodes, output_dim, latent_dim]
        """
        z_original_ndim = z.ndim
        z = z.detach()  # Ensure no unwanted gradients

        if z_original_ndim == 1: # Single node case: z is [latent_dim]
            # Compute Jacobian for a single node
            # The function 'f' should take a [latent_dim] tensor and return [output_dim]
            def f(z_single_node: torch.Tensor) -> torch.Tensor:
                # Assuming self.forward can handle a single [latent_dim] input
                # or needs it unsqueezed to [1, latent_dim] then squeezed back to [output_dim]
                # Adjust 'f' based on your actual decoder_model.forward's expected input for a single item.
                return self.forward(z_single_node)["recon"]

            z.requires_grad_(True)
            J = jacobian(f, z, vectorize=True) # J will be [output_dim, latent_dim]
            return J

        elif z_original_ndim == 2: # Batch case: z is [num_nodes, latent_dim]
            # Compute full Jacobian for all nodes (batch Jacobian)
            def f(z_all: torch.Tensor) -> torch.Tensor:
                # self.decoder_model.forward should take [num_nodes, latent_dim]
                # and return [num_nodes, output_dim]
                return self.forward(z_all)["recon"]

            z.requires_grad_(True)
            # This computes J with shape [num_nodes, output_dim, num_nodes, latent_dim]
            J = jacobian(f, z, vectorize=True)

            # Extract only the relevant Jacobians (diagonal blocks)
            # This assumes that the output for node 'n' only depends on input 'z[n]'.
            # If there are cross-node dependencies, this extraction is not sufficient.
            num_nodes, output_dim, _, latent_dim = J.shape
            return torch.stack([J[n, :, n, :] for n in range(num_nodes)], dim=0) # [num_nodes, output_dim, latent_dim]

        else:
            raise ValueError(
                f"Unsupported input z dimensions for Jacobian computation: {z_original_ndim}. "
                f"Expected 1 (for single node) or 2 (for batch of nodes)."
            )

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
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], 
                    lambda_variance: float = 0.001, epsilon_variance: float = 1e-8) -> torch.Tensor:
        """
        Compute MSE loss for node attribute reconstruction
        
        Args:
            outputs: Output from forward pass
            targets: Target node features
            lambda_variance: Weight for variance penalty
            epsilon_variance: Small value for numerical stability
            
        Returns:
            MSE loss with variance penalty
        """
        pred_features = outputs["node_features"]
        target_features = targets["node_features"]
        
        # Ensure shapes match and are suitable (at least 2D: batch, features)
        if pred_features.shape != target_features.shape:
             raise ValueError("Predicted and target features must have the same shape")
        if pred_features.dim() < 2:
             raise ValueError("Features tensor must have at least two dimensions (batch, features)")

        # --- 1. Standard MSE Loss ---
        loss_mse = F.mse_loss(pred_features, target_features)

        # --- 2. Variance Penalty ---
        mean_target = torch.mean(target_features, dim=1, keepdim=True)
        mean_pred = torch.mean(pred_features, dim=1, keepdim=True)

        ssd_target = torch.sum((target_features - mean_target)**2, dim=1)
        ssd_pred = torch.sum((pred_features - mean_pred)**2, dim=1)

        variance_penalty_per_item = ssd_target / (ssd_pred + epsilon_variance)
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
    
    def compute_jacobian(self, z: torch.Tensor, node_idx: int = None) -> torch.Tensor:
        """
        Analytic Jacobian for our MLP:
        J = W_L · diag(σ'(h_{L-1})) · W_{L-1} · … · diag(σ'(h_0)) · W_1
        Returns [output_dim, latent_dim] for single node, or stacked [N, out, D].
        """
        # make sure we have just one forward pass saved
        out = self.forward(z)  # populates self._preacts, self._acts
        preacts = self._preacts   # list of length L: each [N, layer_dim]
        acts     = self._acts     # list of length L+1: first is z, last is output

        # helper to get derivative diag for a given layer & node
        def get_diag(layer_idx, idx):
            h = preacts[layer_idx][idx]              # [layer_dim]
            return torch.diag(self.activation.backward(h) if hasattr(self.activation, 'backward') 
                            else torch.autograd.functional.jacobian(lambda x: self.activation(x), h))

        if node_idx is not None:
            # start from last linear weight
            J = self.linears[-1].weight              # [out_dim, H_{L-1}]
            # chain backwards
            for l in range(len(preacts)-1, -1, -1):
                D = get_diag(l, node_idx)            # [layer_dim, layer_dim]
                W  = self.linears[l].weight          # [layer_dim, prev_dim]
                J = J @ D @ W                        # reduce to [out_dim, prev_dim]
            return J

        else:
            # for all nodes, stack them
            Js = []
            N = z.size(0)
            for n in range(N):
                Js.append(self.compute_jacobian(z, n))
            return torch.stack(Js, dim=0)           # [N, out, D]
    

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
            loss +=lambda_variance*(
                 F.mse_loss(torch.var(edge_logits), torch.var(edge_labels)) 
                                + 
                                0.01 * F.mse_loss(torch.sum(edge_logits) , torch.sum(edge_labels)))
            loss += heat_kernel_distance(get_adjacency_matrix_from_tensors(targets["edge_index"], edge_labels), get_adjacency_matrix_from_tensors(targets["edge_index"], edge_logits))
            #loss = F.smooth_l1_loss(edge_logits, edge_labels) + 2*F.mse_loss(edge_logits, edge_labels) 
        else:
            # Dense version
            adj_logits = outputs["adj_logits"]
            adj_target = targets["adj_matrix"]
        
            # Mask diagonal if needed
            mask = ~torch.eye(adj_logits.shape[0], dtype=torch.bool, device=adj_logits.device)
            adj_logits_masked = adj_logits[mask]
            adj_target_masked = adj_target[mask]
            loss = F.smooth_l1_loss(adj_logits_masked, adj_target_masked) + 2* F.mse_loss(adj_logits_masked, adj_target_masked) 
            loss += lambda_variance*(
                 F.mse_loss(torch.var(torch.flatten(adj_logits_masked)), torch.var(torch.flatten(adj_target_masked)))
                                + 
                             0.01 * F.mse_loss(torch.sum(torch.flatten(adj_logits_masked)) , torch.sum(torch.flatten(adj_target_masked))))
            loss += heat_kernel_distance(adj_logits_masked, adj_target_masked)
            #loss = F.smooth_l1_loss(adj_logits_masked, adj_target_masked) + 2* F.mse_loss(adj_logits_masked, adj_target_masked) 
            
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

    # def compute_jacobian(self, z: torch.Tensor, node_idx: int = None) -> torch.Tensor:
    #     """
    #     Compute the Jacobian of the decoder output with respect to the latent space
        
    #     Args:
    #         z: Latent variables [num_nodes, latent_dim]
    #         node_idx: Optional index of the specific node to compute Jacobian for
            
    #     Returns:
    #         Jacobian matrix
    #     """
    #     # Detach z and require gradients
    #     z_detached = z.detach().clone().requires_grad_(True)
        
    #     # Forward pass to get node feature predictions
    #     outputs = self.forward(z_detached)
    #     node_features = outputs["node_features"]
        
    #     # Handle single node case
    #     if node_idx is not None:
    #         node_features = node_features[node_idx]
            
    #         # Initialize Jacobian matrix (output_dim × latent_dim)
    #         jacobian = torch.zeros(self.output_dim, z.size(1), device=z.device)
            
    #         # Compute Jacobian row by row
    #         for i in range(self.output_dim):
    #             # Zero gradients
    #             if z_detached.grad is not None:
    #                 z_detached.grad.zero_()
                
    #             # Backward for this feature dimension
    #             node_features[i].backward(retain_graph=True)
                
    #             # Store gradient (Jacobian row)
    #             jacobian[i] = z_detached.grad[node_idx].clone()
        
    #     else:
    #         # Full Jacobian for all nodes
    #         batch_size = z_detached.size(0)
            
    #         # Initialize Jacobian tensor (nodes × output_dim × latent_dim)
    #         jacobian = torch.zeros(batch_size, self.output_dim, z.size(1), device=z.device)
            
    #         # For each node and feature dimension
    #         for n in range(batch_size):
    #             for i in range(self.output_dim):
    #                 # Zero gradients
    #                 if z_detached.grad is not None:
    #                     z_detached.grad.zero_()
                    
    #                 # Backward for this specific element
    #                 node_features[n, i].backward(retain_graph=True)
                    
    #                 # Store gradient
    #                 jacobian[n, i] = z_detached.grad[n].clone()
        
    #     return jacobian
    
    def compute_jacobian(self, z: torch.Tensor, node_idx: Optional[int] = None) -> torch.Tensor:
        """
        Compute the Jacobian of the decoder output with respect to the latent space.

        Args:
            z: Latent variables [num_nodes, latent_dim]
            node_idx: Optional index of a specific node to compute the Jacobian for

        Returns:
            Jacobian matrix or tensor:
            - If node_idx is given: [output_dim, latent_dim]
            - Else: [num_nodes, output_dim, latent_dim]
        """
        z = z.detach()  # Ensure no unwanted gradients

        if node_idx is not None or z.ndim == 1:
            # Compute Jacobian for a single node
            def f(z_node: torch.Tensor) -> torch.Tensor:
                z_full = z.clone()
                z_full[node_idx] = z_node
                return self.forward(z_full)["node_features"][node_idx]

            # Single-node input vector
            z_node = z[node_idx].detach().requires_grad_(True)
            J = jacobian(f, z_node, vectorize=True)  # [output_dim, latent_dim]
            return J if J.ndim == 2 else J.squeeze()  

        else:
            # Compute full Jacobian for all nodes
            def f(z_all: torch.Tensor) -> torch.Tensor:
                return self.forward(z_all)["node_features"]  # [num_nodes, output_dim]

            z.requires_grad_(True)
            J = jacobian(f, z, vectorize=True)  # [num_nodes, output_dim, num_nodes, latent_dim]

            # Extract only the relevant Jacobians (diagonal blocks)
            num_nodes, output_dim, _, latent_dim = J.shape
            return torch.stack([J[n, :, n, :] for n in range(num_nodes)], dim=0)  # [num_nodes, output_dim, latent_dim]

class PreprocessedDecoder(nn.Module):
    """
    Wrapper that combines a preprocessing layer with a decoder
    """
    def __init__(
        self,
        decoder: "DecoderBase",
        preprocessor: "PreprocessingLayer",
        name: Optional[str] = None
    ):
        super(PreprocessedDecoder, self).__init__()
        self.decoder = decoder
        self.preprocessor = preprocessor
        self.name = name if name is not None else f"preprocessed_{decoder.name}"
        
        # Delegate properties to the underlying decoder
        self.latent_dim = decoder.latent_dim
    
    def forward(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through preprocessor then decoder
        
        Args:
            z: Input latent code [num_nodes, latent_dim]
            **kwargs: Additional arguments for decoder
            
        Returns:
            Decoder outputs with preprocessing info
        """
        # Apply preprocessing
        z_preprocessed = self.preprocessor(z, **kwargs)
        print("Z-preprocessed", z_preprocessed)
        # Pass through decoder
        outputs = self.decoder(z_preprocessed, **kwargs)
        
        # Add preprocessing info to outputs
        outputs["preprocessed_z"] = z_preprocessed
        outputs["original_z"] = z
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Delegate loss computation to underlying decoder
        """
        return self.decoder.compute_loss(outputs, targets)
    
    def compute_jacobian(self, z: torch.Tensor, node_idx: int = None) -> torch.Tensor:
        """
        Compute Jacobian through the preprocessing layer
        """
        if hasattr(self.decoder, "compute_jacobian"):
            # This would require chain rule - simplified implementation
            z_preprocessed = self.preprocessor(z)
            return self.decoder.compute_jacobian(z_preprocessed, node_idx)
        else:
            raise NotImplementedError(f"Underlying decoder does not implement compute_jacobian")


# Extended GraphVAE class to support preprocessed decoders
def add_preprocessed_decoder_to_graphvae():
    """
    Extension methods for GraphVAE to support preprocessed decoders
    """
    def add_preprocessed_decoder(self, decoder: "DecoderBase", preprocessor: "PreprocessingLayer", name: Optional[str] = None):
        """
        Add a decoder with preprocessing layer
        
        Args:
            decoder: Base decoder to wrap
            preprocessor: Preprocessing layer to apply
            name: Optional name for the preprocessed decoder
        """
        preprocessed_decoder = PreprocessedDecoder(decoder, preprocessor, name)
        self.add_decoder(preprocessed_decoder)
        return preprocessed_decoder
    
    # This would be added to the GraphVAE class
    return add_preprocessed_decoder


# Example usage for your specific case
class DistanceBasedAdjacencyDecoder(DecoderBase):
    """
    Adjacency decoder that uses preprocessed distance features
    """
    def __init__(
        self,
        distance_feature_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
        activation=nn.ReLU(),
        name: str = "distance_adj_decoder"
    ):
        super(DistanceBasedAdjacencyDecoder, self).__init__(distance_feature_dim, name)
        
        self.dropout = dropout
        self.activation = activation
        
        # Build MLP that takes distance features
        layers = []
        in_features = distance_feature_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(in_features, dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            in_features = dim
        
        layers.append(nn.Linear(in_features, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, edge_index: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Predict adjacency from distance features
        
        Args:
            z: Distance features [num_nodes, distance_feature_dim]
            edge_index: Optional edge index for sparse prediction
            
        Returns:
            Adjacency predictions
        """
        if edge_index is not None:
            # For sparse case, we need to extract relevant distance features per edge
            # This is a simplified version - you might want to modify based on your needs
            src, dst = edge_index
            # Use average of source and destination distance features
            edge_features = (z[src] + z[dst]) / 2
            edge_logits = self.mlp(edge_features).squeeze(-1)
            
            return {
                "edge_logits": edge_logits,
                "edge_index": edge_index
            }
        else:
            # Dense version - predict full adjacency
            batch_size = z.size(0)
            adj_logits = []
            
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        # Combine features for this pair
                        pair_features = (z[i] + z[j]) / 2
                        logit = self.mlp(pair_features).squeeze()
                        adj_logits.append(logit)
            
            adj_logits = torch.stack(adj_logits).view(batch_size, batch_size)
            
            return {
                "adj_logits": adj_logits
            }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for distance-based adjacency prediction
        """
        if "adj_logits" not in outputs:
            # Sparse version
            edge_logits = outputs["edge_logits"]
            edge_labels = targets["edge_labels"]
            loss = F.smooth_l1_loss(edge_logits, edge_labels) + 2*F.mse_loss(edge_logits, edge_labels)
        else:
            # Dense version
            adj_logits = outputs["adj_logits"]
            adj_target = targets["adj_matrix"]
            
            # Mask diagonal
            mask = ~torch.eye(adj_logits.shape[0], dtype=torch.bool, device=adj_logits.device)
            adj_logits_masked = adj_logits[mask]
            adj_target_masked = adj_target[mask]
            loss = F.smooth_l1_loss(adj_logits_masked, adj_target_masked) + 2*F.mse_loss(adj_logits_masked, adj_target_masked)
            
        return loss


# Example of how to use this in practice:
def create_distance_based_adjacency_decoder(latent_dim: int, metric_function: Optional[Callable] = None):
    """
    Factory function to create a distance-based adjacency decoder with preprocessing
    
    Args:
        latent_dim: Dimension of the latent space
        metric_function: Optional custom metric function
        
    Returns:
        PreprocessedDecoder combining distance preprocessing and adjacency decoder
    """
    # Create distance preprocessing layer
    preprocessor = DistancePreprocessingLayer(
        latent_dim=latent_dim,
        metric_function=metric_function,
        num_integration_points=100,
        distance_mode="linear_interpolation",
        output_dim=64,  # Transform to 64-dim distance features
        name="distance_preprocessor"
    )
    
    # Create distance-based adjacency decoder
    decoder = DistanceBasedAdjacencyDecoder(
        distance_feature_dim=64,
        hidden_dims=[32, 16],
        name="distance_adj_decoder"
    )
    
    # Combine them
    return PreprocessedDecoder(decoder, preprocessor, "distance_based_adjacency")


class AdaptiveJacobianAdjacencyDecoder(DecoderBase):
    """
    Adjacency decoder that adapts to the geometry learned by another decoder
    through Jacobian-based metrics.
    """
    def __init__(
        self,
        distance_feature_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
        activation=nn.ReLU(),
        use_attention: bool = True,
        name: str = "adaptive_jacobian_adj_decoder"
    ):
        super(AdaptiveJacobianAdjacencyDecoder, self).__init__(distance_feature_dim, name)
        
        self.dropout = dropout
        self.activation = activation
        self.use_attention = use_attention
        
        # Main MLP for distance feature processing
        layers = []
        in_features = distance_feature_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(in_features, dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            in_features = dim
        
        self.feature_mlp = nn.Sequential(*layers)
        
        # Attention mechanism for edge prediction
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=in_features, 
                num_heads=4, 
                dropout=dropout,
                batch_first=True
            )
            self.edge_predictor = nn.Linear(in_features, 1)
        else:
            self.edge_predictor = nn.Linear(in_features, 1)
    
    def forward(self, z: torch.Tensor, edge_index: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Predict adjacency from Jacobian-derived distance features
        
        Args:
            z: Distance features [num_nodes, distance_feature_dim]
            edge_index: Optional edge index for sparse prediction
            
        Returns:
            Adjacency predictions
        """
        # Process features through MLP
        processed_features = self.feature_mlp(z)  # [num_nodes, hidden_dim]
        
        if edge_index is not None:
            # Sparse prediction
            src, dst = edge_index
            
            if self.use_attention:
                # Use attention to combine source and destination features
                src_features = processed_features[src].unsqueeze(1)  # [num_edges, 1, hidden_dim]
                dst_features = processed_features[dst].unsqueeze(1)  # [num_edges, 1, hidden_dim]
                
                # Concatenate for attention
                edge_features = torch.cat([src_features, dst_features], dim=1)  # [num_edges, 2, hidden_dim]
                
                # Apply attention
                attended_features, _ = self.attention(edge_features, edge_features, edge_features)
                # Take mean of attended features
                combined_features = torch.mean(attended_features, dim=1)  # [num_edges, hidden_dim]
            else:
                # Simple combination
                combined_features = (processed_features[src] + processed_features[dst]) / 2
            
            edge_logits = self.edge_predictor(combined_features).squeeze(-1)
            
            return {
                "edge_logits": edge_logits,
                "edge_index": edge_index,
                "processed_features": processed_features
            }
        else:
            # Dense prediction
            batch_size = z.size(0)
            adj_logits = torch.zeros(batch_size, batch_size, device=z.device)
            
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        if self.use_attention:
                            # Attention-based combination
                            pair_features = torch.stack([processed_features[i], processed_features[j]]).unsqueeze(0)
                            attended, _ = self.attention(pair_features, pair_features, pair_features)
                            combined = torch.mean(attended.squeeze(0), dim=0)
                        else:
                            combined = (processed_features[i] + processed_features[j]) / 2
                        
                        adj_logits[i, j] = self.edge_predictor(combined).squeeze()
            
            return {
                "adj_logits": adj_logits,
                "processed_features": processed_features
            }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss with additional regularization based on feature consistency
        """
        # Standard adjacency loss
        if "adj_logits" not in outputs:
            # Sparse version
            edge_logits = outputs["edge_logits"]
            edge_labels = targets["edge_labels"]
            adj_loss = F.smooth_l1_loss(edge_logits, edge_labels) + 2*F.mse_loss(edge_logits, edge_labels)
        else:
            # Dense version
            adj_logits = outputs["adj_logits"]
            print(targets.keys())
            adj_target = targets["adj_matrix"]
            
            # Mask diagonal
            mask = ~torch.eye(adj_logits.shape[0], dtype=torch.bool, device=adj_logits.device)
            adj_logits_masked = adj_logits[mask]
            adj_target_masked = adj_target[mask]
            adj_loss = F.smooth_l1_loss(adj_logits_masked, adj_target_masked) + 2*F.mse_loss(adj_logits_masked, adj_target_masked)
        
        # Optional: Add feature regularization
        processed_features = outputs.get("processed_features")
        if processed_features is not None:
            # Encourage feature diversity
            feature_var = torch.var(processed_features, dim=0).mean()
            regularization = -0.01 * feature_var  # Negative to encourage higher variance
            adj_loss += regularization
        
        return adj_loss


# Factory function to create the complete system
def create_jacobian_adaptive_adjacency_decoder(
    latent_dim: int,
    reference_decoder: "DecoderBase",
    reference_model: "GraphVAE",
    distance_feature_dim: int = 64,
    num_integration_points: int = 20
):
    """
    Factory function to create a Jacobian-adaptive adjacency decoder
    
    Args:
        latent_dim: Dimension of the latent space
        reference_decoder: The decoder whose Jacobian will define the metric
        reference_model: The GraphVAE model containing the reference decoder
        distance_feature_dim: Dimension of the distance feature representation
        num_integration_points: Number of points for numerical integration
        
    Returns:
        PreprocessedDecoder with Jacobian-based preprocessing
    """
    # Create Jacobian-based preprocessing layer
    preprocessor = JacobianMetricPreprocessingLayer(
        latent_dim=latent_dim,
        reference_decoder=reference_decoder,
        reference_model=reference_model,
        num_integration_points=num_integration_points,
        output_dim=distance_feature_dim,
        name="jacobian_metric_preprocessor"
    )
    
    # Create adaptive adjacency decoder
    decoder = AdaptiveJacobianAdjacencyDecoder(
        distance_feature_dim=distance_feature_dim,
        hidden_dims=[32, 16],
        use_attention=True,
        name="adaptive_jacobian_adj_decoder"
    )
    
    # Combine them
    return PreprocessedDecoder(decoder, preprocessor, "jacobian_adaptive_adjacency")

# Example for adding losses
# def create_jacobian_loss_function(decoder_instance, target_jacobian=None, regularization_type="norm"):
#     """
#     Factory function to create a Jacobian loss function for a specific decoder
    
#     Args:
#         decoder_instance: The decoder to compute Jacobian for
    
#     Returns:
#         Loss function that can be added to the decoder
#     """

#     def jacobian_loss(outputs, targets, z, **kwargs):
#         print(targets, outputs)
#         # Compute current Jacobian
#         current_jacobian = decoder_instance.compute_jacobian(z)
        
#         if regularization_type == "target" and target_jacobian is not None:
#             # Compare to target Jacobian
#             loss = nn.functional.mse_loss(current_jacobian, target_jacobian)
#         elif regularization_type == "frobenius":
#             # Frobenius norm regularization
#             loss = torch.norm(current_jacobian, p='fro', dim=(-2, -1)).mean()
#         else:  # "norm"
#             # L2 norm regularization (encourage smaller gradients)
#             loss = torch.norm(current_jacobian, dim=-1).mean()
        
#         return loss
    
#     return jacobian_loss

# class ConditionalJacobianLoss:
#     """
#     A Jacobian loss that changes behavior based on training conditions
#     """
#     def __init__(self, decoder_instance, threshold_epoch=100):
#         self.decoder_instance = decoder_instance
#         self.threshold_epoch = threshold_epoch
#         self.current_epoch = 0
    
#     def update_epoch(self, epoch):
#         self.current_epoch = epoch
    
#     def __call__(self, outputs, targets, z, **kwargs):
#         current_jacobian = self.decoder_instance.compute_jacobian(z)
        
#         if self.current_epoch < self.threshold_epoch:
#             # Early training: encourage larger gradients (avoid collapse)
#             loss = -torch.norm(current_jacobian, dim=-1).mean()
#         else:
#             # Later training: regularize gradients (smooth manifold)
#             loss = torch.norm(current_jacobian, dim=-1).mean()
        
#         return loss


# node_decoder = model_phase2.get_decoder("node_attr_decoder")

# jacobian_loss_weight = 1

# # Create Jacobian loss function
# jacobian_loss_fn = create_jacobian_loss_function(
#     node_decoder, 
#     regularization_type="norm"
# )

# model_phase2.add_custom_loss_to_decoder(
#     "node_attr_decoder", 
#     "node_attr_decoder_jacobian", 
#     jacobian_loss_fn, 
#     weight=jacobian_loss_weight
# )
class LatentDistanceDecoder(DecoderBase):
    """
    Fictive decoder that outputs latent codes but computes a meaningful loss 
    based on pairwise distances between connected nodes using Jacobian-based metrics.
    """
    def __init__(
        self, 
        latent_dim: int,
        reference_decoder_name: str = "node_attr_decoder",
        distance_mode: str = "linear_interpolation",  # or "direct"
        num_integration_points: int = 10,
        metric_regularization: float = 1e-6,
        cache_jacobians: bool = True,
        name: str = "latent_distance_decoder"
    ):
        super(LatentDistanceDecoder, self).__init__(latent_dim, name)
        
        self.reference_decoder_name = reference_decoder_name
        self.distance_mode = distance_mode
        self.num_integration_points = num_integration_points
        self.metric_regularization = metric_regularization
        self.cache_jacobians = cache_jacobians
        
        # Cache for Jacobian computations
        self._jacobian_cache = {}
        self._cache_valid = False
        self._reference_decoder = None
    
    def set_reference_decoder(self, decoder: "DecoderBase"):
        """
        Set the reference decoder used for Jacobian computation
        
        Args:
            decoder: The decoder to use for Jacobian computation (e.g., NodeAttributeDecoder)
        """
        self._reference_decoder = decoder
        self._jacobian_cache.clear()  # Clear cache when reference changes
        self._cache_valid = False
    
    def forward(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass - simply returns the latent codes (fictive decoder)
        
        Args:
            z: Latent node embeddings [num_nodes, latent_dim]
            
        Returns:
            Dict with latent codes
        """
        return {
            "latent_codes": z.clone()
        }
    
    def compute_metric_tensor(self, z: torch.Tensor, node_idx: int) -> torch.Tensor:
        """
        Compute the metric tensor at a given point in latent space using the 
        Jacobian of the reference decoder.
        
        The metric tensor G is computed as: G = J^T @ J
        where J is the Jacobian of the reference decoder output w.r.t. latent variables.
        
        Args:
            z: Latent variables [num_nodes, latent_dim]
            node_idx: Index of the node to compute metric for
            
        Returns:
            Metric tensor [latent_dim, latent_dim]
        """
        if self._reference_decoder is None:
            # Fallback to identity metric (Euclidean)
            return torch.eye(self.latent_dim, device=z.device, dtype=z.dtype)
        
        # Check cache first
        cache_key = f"{node_idx}_{hash(z.data.tobytes()) if hasattr(z.data, 'tobytes') else id(z)}"
        if self.cache_jacobians and cache_key in self._jacobian_cache:
            return self._jacobian_cache[cache_key]
        
        try:
            # Compute Jacobian of reference decoder at this point        
            jacobian = self._reference_decoder.compute_jacobian(z, node_idx)
            
            #print(f"Debug: Jacobian shape: {jacobian.shape}, z shape: {z.shape}, latent_dim: {self.latent_dim}")
            
            # Handle different Jacobian shapes based on decoder type
            if jacobian.dim() == 2:
                # Shape: [output_dim, latent_dim]
                if jacobian.size(1) != self.latent_dim:
                    raise ValueError(f"Jacobian latent dimension {jacobian.size(1)} doesn't match expected {self.latent_dim}")
                # Metric tensor: G = J^T @ J -> [latent_dim, latent_dim]
                metric_tensor = torch.mm(jacobian.t(), jacobian)
            elif jacobian.dim() == 3:
                # Shape: [num_nodes, output_dim, latent_dim] - take the specific node
                if node_idx >= jacobian.size(0):
                    raise ValueError(f"Node index {node_idx} out of bounds for Jacobian with {jacobian.size(0)} nodes")
                node_jacobian = jacobian[node_idx]  # [output_dim, latent_dim]
                if node_jacobian.size(1) != self.latent_dim:
                    raise ValueError(f"Jacobian latent dimension {node_jacobian.size(1)} doesn't match expected {self.latent_dim}")
                metric_tensor = torch.mm(node_jacobian.t(), node_jacobian)
            elif jacobian.dim() == 1:
                # Handle case where Jacobian is flattened or single dimension
                if jacobian.size(0) == self.latent_dim:
                    # Treat as single gradient vector, create outer product
                    metric_tensor = torch.outer(jacobian, jacobian)
                else:
                    raise ValueError(f"1D Jacobian size {jacobian.size(0)} doesn't match latent_dim {self.latent_dim}")
            else:
                raise ValueError(f"Unexpected Jacobian shape: {jacobian.shape}")
            
            # Ensure metric tensor has correct shape
            if metric_tensor.shape != (self.latent_dim, self.latent_dim):
                raise ValueError(f"Metric tensor has wrong shape: {metric_tensor.shape}, expected ({self.latent_dim}, {self.latent_dim})")
            
            # Add regularization for numerical stability
            metric_tensor += self.metric_regularization * torch.eye(
                self.latent_dim, device=z.device, dtype=z.dtype
            )
            
            # Cache the result
            if self.cache_jacobians:
                self._jacobian_cache[cache_key] = metric_tensor
            
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
        Compute distance using linear interpolation with Jacobian-based metric:
        ∫₀¹ √((u-v)ᵀG(x(t))(u-v))dt where x(t) = tu + (1-t)v
        and G(x) is computed from the Jacobian at point x.
        
        Args:
            u, v: Latent vectors [latent_dim]
            z_full: Full latent matrix [num_nodes, latent_dim] (for Jacobian computation)
            node_idx_u, node_idx_v: Node indices for u and v
            
        Returns:
            Distance scalar
        """
        def integrand(t):
            # Interpolated point
            x_t = t * u + (1 - t) * v
            
            # Create temporary z matrix with interpolated point
            # We'll use the midpoint node index for metric computation
            mid_idx = (node_idx_u + node_idx_v) // 2
            z_temp = z_full.clone()
            z_temp[mid_idx] = x_t
            
            try:
                # Compute metric tensor at interpolated point
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

    
    def direct_distance_with_jacobian(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        z_full: torch.Tensor,
        node_idx_u: int,
        node_idx_v: int
    ) -> torch.Tensor:
        """
        Compute distance using metric at midpoint between u and v
        
        Args:
            u, v: Latent vectors [latent_dim]
            z_full: Full latent matrix [num_nodes, latent_dim]
            node_idx_u, node_idx_v: Node indices for u and v
            
        Returns:
            Distance scalar
        """
        # Compute midpoint
        mid_point = (u + v) / 2
        mid_idx = (node_idx_u + node_idx_v) // 2
        
        # Create temporary z matrix with midpoint
        z_temp = z_full.clone()
        z_temp[mid_idx] = mid_point
        
        try:
            # Compute metric tensor at midpoint
            G = self.compute_metric_tensor(z_temp, mid_idx)
            
            # Compute distance: sqrt((u-v)^T @ G @ (u-v))
            diff = u - v
            quadratic_form = torch.sum(diff * (G @ diff))
            return torch.sqrt(torch.clamp(quadratic_form, min=1e-8))
            
        except Exception as e:
            # Fallback to Euclidean distance
            return torch.norm(u - v)
    
    def compute_pairwise_distance(
        self,
        z: torch.Tensor,
        node_i: int,
        node_j: int
    ) -> torch.Tensor:
        """
        Compute distance between two specific nodes
        
        Args:
            z: Latent embeddings [num_nodes, latent_dim]
            node_i, node_j: Indices of the two nodes
            
        Returns:
            Distance scalar
        """
        if self.distance_mode == "linear_interpolation":
            return self.linear_interpolation_distance_with_jacobian(
                z[node_i], z[node_j], z, node_i, node_j
            )
        else:  # direct mode
            return self.direct_distance_with_jacobian(
                z[node_i], z[node_j], z, node_i, node_j
            )
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss based on pairwise distances between connected nodes.
        The loss sums up the estimated distances for all existing edges in the graph.
        
        Args:
            outputs: Output from forward pass (contains latent_codes)
            targets: Target data (should contain edge_index and optionally edge_labels or adj_matrix)
            
        Returns:
            Distance-based loss
        """
        z = outputs["latent_codes"]
        
        # Clear cache at the beginning of each loss computation
        if not self._cache_valid:
            self._jacobian_cache.clear()
            self._cache_valid = True
        
        total_distance = 0.0
        num_edges = 0
        
        # Check if we have edge_index (sparse format) or adj_matrix (dense format)
        if "edge_index" in targets:
            # Sparse format - use edge_index
            edge_index = targets["edge_index"]
            edge_labels = targets.get("edge_labels", None)
            
            src_nodes, dst_nodes = edge_index[0], edge_index[1]
            
            # Iterate through all edges
            for idx in tqdm(range(edge_index.size(1))):
                src_idx = src_nodes[idx].item()
                dst_idx = dst_nodes[idx].item()
                
                # Skip self-loops if desired
                if src_idx == dst_idx:
                    continue
                
                # Check if this edge exists (if edge_labels provided)
                if edge_labels is not None:
                    edge_weight = edge_labels[idx].item()
                    if edge_weight <= 0:  # Skip non-existing edges
                        continue
                else:
                    edge_weight = 1.0  # Assume all edges in edge_index exist
                
                # Compute distance between connected nodes
                distance = self.compute_pairwise_distance(z, src_idx, dst_idx)
                total_distance += edge_weight * distance
                num_edges += edge_weight
        
        elif "adj_matrix" in targets:
            # Dense format - use adjacency matrix
            adj_matrix = targets["adj_matrix"]
            num_nodes = adj_matrix.size(0)
            
            # Iterate through upper triangle of adjacency matrix
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    edge_weight = adj_matrix[i, j].item()
                    
                    if edge_weight > 0:  # Edge exists
                        # Compute distance between connected nodes
                        distance = self.compute_pairwise_distance(z, i, j)
                        total_distance += edge_weight * distance
                        num_edges += edge_weight
        
        else:
            raise ValueError("Targets must contain either 'edge_index' or 'adj_matrix'")
        
        # Normalize by number of edges and apply weight
        if num_edges > 0:
            avg_distance = total_distance / num_edges
            loss = avg_distance
        else:
            # No edges found - return zero loss
            loss = torch.tensor(0.0, device=z.device, requires_grad=True)
        
        return loss
    
    def compute_jacobian(self, z: torch.Tensor, node_idx: int = None) -> torch.Tensor:
        """
        Compute Jacobian for this decoder (which is just identity since we output z)
        
        Args:
            z: Latent variables [num_nodes, latent_dim]
            node_idx: Optional node index
            
        Returns:
            Identity Jacobian
        """
        if node_idx is not None:
            # Return identity matrix for single node
            return torch.eye(self.latent_dim, device=z.device, dtype=z.dtype)
        else:
            # Return identity for all nodes
            batch_size = z.size(0)
            return torch.eye(self.latent_dim, device=z.device, dtype=z.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    
    def invalidate_cache(self):
        """Invalidate the Jacobian cache"""
        self._jacobian_cache.clear()
        self._cache_valid = False

# This latter implementation uses grid cache to speed up computations.
# In a near future, it will replace the former one.
class LatentDistanceDecoder(DecoderBase):
    """
    Fictive decoder that outputs latent codes but computes a meaningful loss 
    based on pairwise distances between connected nodes using Jacobian-based metrics.
    Enhanced with simple but effective caching.
    """
    def __init__(
        self, 
        latent_dim: int,
        reference_decoder_name: str = "node_attr_decoder",
        distance_mode: str = "linear_interpolation",  # or "direct"
        num_integration_points: int = 10,
        metric_regularization: float = 1e-6,
        cache_jacobians: bool = True,
        # Simple caching parameters
        cache_tolerance: float = 1e-4,  # How close points need to be to reuse cache
        max_cache_size: int = 1000,  # Maximum cached entries
        name: str = "latent_distance_decoder"
    ):
        super(LatentDistanceDecoder, self).__init__(latent_dim, name)
        
        self.reference_decoder_name = reference_decoder_name
        self.distance_mode = distance_mode
        self.num_integration_points = num_integration_points
        self.metric_regularization = metric_regularization
        self.cache_jacobians = cache_jacobians
        self.cache_tolerance = cache_tolerance
        self.max_cache_size = max_cache_size
        
        # Simple cache: store (z_point_hash, metric_tensor) pairs
        self._metric_cache = {}  # Key: simple hash -> (z_point, metric_tensor_detached)
        self._jacobian_cache = {}
        self._cache_valid = False
        self._reference_decoder = None
    
    def set_reference_decoder(self, decoder: "DecoderBase"):
        """Set the reference decoder used for Jacobian computation"""
        self._reference_decoder = decoder
        self._jacobian_cache.clear()
        self._metric_cache.clear()
        self._cache_valid = False
    
    def _get_cache_key(self, z_point: torch.Tensor, node_idx: int) -> str:
        """Generate a simple cache key from z_point and node_idx"""
        # Round to cache_tolerance precision to increase cache hits
        z_rounded = torch.round(z_point / self.cache_tolerance) * self.cache_tolerance
        z_str = "_".join([f"{x:.6f}" for x in z_rounded.cpu().numpy()])
        return f"{node_idx}_{z_str}"
    
    def _find_cached_metric(self, z_point: torch.Tensor, node_idx: int) -> Optional[torch.Tensor]:
        """Find cached metric tensor for similar z_point"""
        cache_key = self._get_cache_key(z_point, node_idx)
        
        if cache_key in self._metric_cache:
            cached_z, cached_metric = self._metric_cache[cache_key]
            # Double-check distance (in case of hash collisions)
            if torch.norm(z_point - cached_z) < self.cache_tolerance:
                return cached_metric.to(device=z_point.device, dtype=z_point.dtype)
        
        return None
    
    def _cache_metric(self, z_point: torch.Tensor, node_idx: int, metric_tensor: torch.Tensor):
        """Cache a metric tensor"""
        if len(self._metric_cache) >= self.max_cache_size:
            # Simple LRU: remove oldest entry
            oldest_key = next(iter(self._metric_cache))
            del self._metric_cache[oldest_key]
        
        cache_key = self._get_cache_key(z_point, node_idx)
        self._metric_cache[cache_key] = (z_point.detach().clone(), metric_tensor.detach().clone())
    
    def forward(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass - simply returns the latent codes (fictive decoder)"""
        return {"latent_codes": z.clone()}
    
    def compute_metric_tensor(self, z: torch.Tensor, node_idx: int) -> torch.Tensor:
        """
        Compute the metric tensor at a given point in latent space using the 
        Jacobian of the reference decoder. Uses simple caching for efficiency.
        """
        z_point = z[node_idx]
        
        # Try to find cached result first
        cached_metric = self._find_cached_metric(z_point, node_idx)
        if cached_metric is not None:
            # CRITICAL: Preserve gradients by adding zero times actual computation
            actual_metric = self._compute_metric_raw(z, node_idx)
            return cached_metric + 0.0 * actual_metric
        
        # Compute fresh metric tensor
        metric_tensor = self._compute_metric_raw(z, node_idx)
        
        # Cache the result (detached)
        if self.cache_jacobians:
            self._cache_metric(z_point, node_idx, metric_tensor)
        
        return metric_tensor
    
    def _compute_metric_raw(self, z: torch.Tensor, node_idx: int) -> torch.Tensor:
        """Raw computation of metric tensor without caching"""
        if self._reference_decoder is None:
            return torch.eye(self.latent_dim, device=z.device, dtype=z.dtype)
        
        # Check old-style jacobian cache first
        cache_key = f"{node_idx}_{hash(z.data.tobytes()) if hasattr(z.data, 'tobytes') else id(z)}"
        if self.cache_jacobians and cache_key in self._jacobian_cache:
            return self._jacobian_cache[cache_key]
        
        try:
            # Compute Jacobian of reference decoder at this point        
            jacobian = self._reference_decoder.compute_jacobian(z, node_idx)
            
            # Handle different Jacobian shapes based on decoder type
            if jacobian.dim() == 2:
                # Shape: [output_dim, latent_dim]
                if jacobian.size(1) != self.latent_dim:
                    raise ValueError(f"Jacobian latent dimension {jacobian.size(1)} doesn't match expected {self.latent_dim}")
                # Metric tensor: G = J^T @ J -> [latent_dim, latent_dim]
                metric_tensor = torch.mm(jacobian.t(), jacobian)
            elif jacobian.dim() == 3:
                # Shape: [num_nodes, output_dim, latent_dim] - take the specific node
                if node_idx >= jacobian.size(0):
                    raise ValueError(f"Node index {node_idx} out of bounds for Jacobian with {jacobian.size(0)} nodes")
                node_jacobian = jacobian[node_idx]  # [output_dim, latent_dim]
                if node_jacobian.size(1) != self.latent_dim:
                    raise ValueError(f"Jacobian latent dimension {node_jacobian.size(1)} doesn't match expected {self.latent_dim}")
                metric_tensor = torch.mm(node_jacobian.t(), node_jacobian)
            elif jacobian.dim() == 1:
                # Handle case where Jacobian is flattened or single dimension
                if jacobian.size(0) == self.latent_dim:
                    # Treat as single gradient vector, create outer product
                    metric_tensor = torch.outer(jacobian, jacobian)
                else:
                    raise ValueError(f"1D Jacobian size {jacobian.size(0)} doesn't match latent_dim {self.latent_dim}")
            else:
                raise ValueError(f"Unexpected Jacobian shape: {jacobian.shape}")
            
            # Ensure metric tensor has correct shape
            if metric_tensor.shape != (self.latent_dim, self.latent_dim):
                raise ValueError(f"Metric tensor has wrong shape: {metric_tensor.shape}, expected ({self.latent_dim}, {self.latent_dim})")
            
            # Add regularization for numerical stability
            metric_tensor += self.metric_regularization * torch.eye(
                self.latent_dim, device=z.device, dtype=z.dtype
            )
            
            # Cache the result (old-style cache)
            if self.cache_jacobians:
                self._jacobian_cache[cache_key] = metric_tensor
            
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
        Compute distance using linear interpolation with Jacobian-based metric:
        ∫₀¹ √((u-v)ᵀG(x(t))(u-v))dt where x(t) = tu + (1-t)v
        and G(x) is computed from the Jacobian at point x.
        """
        def integrand(t):
            # Interpolated point
            x_t = t * u + (1 - t) * v
            
            # Create temporary z matrix with interpolated point
            # We'll use the midpoint node index for metric computation
            mid_idx = (node_idx_u + node_idx_v) // 2
            z_temp = z_full.clone()
            z_temp[mid_idx] = x_t
            
            try:
                # Compute metric tensor at interpolated point
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
    
    def direct_distance_with_jacobian(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        z_full: torch.Tensor,
        node_idx_u: int,
        node_idx_v: int
    ) -> torch.Tensor:
        """
        Compute distance using metric at midpoint between u and v
        """
        # Compute midpoint
        mid_point = (u + v) / 2
        mid_idx = (node_idx_u + node_idx_v) // 2
        
        # Create temporary z matrix with midpoint
        z_temp = z_full.clone()
        z_temp[mid_idx] = mid_point
        
        try:
            # Compute metric tensor at midpoint
            G = self.compute_metric_tensor(z_temp, mid_idx)
            
            # Compute distance: sqrt((u-v)^T @ G @ (u-v))
            diff = u - v
            quadratic_form = torch.sum(diff * (G @ diff))
            return torch.sqrt(torch.clamp(quadratic_form, min=1e-8))
            
        except Exception as e:
            # Fallback to Euclidean distance
            return torch.norm(u - v)
    
    def compute_pairwise_distance(
        self,
        z: torch.Tensor,
        node_i: int,
        node_j: int
    ) -> torch.Tensor:
        """Compute distance between two specific nodes"""
        if self.distance_mode == "linear_interpolation":
            return self.linear_interpolation_distance_with_jacobian(
                z[node_i], z[node_j], z, node_i, node_j
            )
        else:  # direct mode
            return self.direct_distance_with_jacobian(
                z[node_i], z[node_j], z, node_i, node_j
            )
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        negative_distance_weight: float = 1,
    ) -> torch.Tensor:
        """
        Compute loss based on:
        - Pairwise distances between connected nodes (positive edges)
        - Penalty for non-connected nodes being too close in the latent space (negative edges)
        """
        z = outputs["latent_codes"]
        
        # Clear cache
        if not self._cache_valid:
            self._jacobian_cache.clear()
            self._cache_valid = True
        
        total_pos_distance = 0.0
        total_neg_penalty = 0.0
        num_pos_edges = 0

        edge_set = set()  # To avoid sampling existing edges as negatives

        if "edge_index" in targets:
            edge_index = targets["edge_index"]
            num_negative_samples = edge_index.size(1)
            edge_labels = targets.get("edge_labels", None)
            src_nodes, dst_nodes = edge_index[0], edge_index[1]
            
            for idx in tqdm(range(edge_index.size(1)), desc="Positive Edges"):
                src_idx = src_nodes[idx].item()
                dst_idx = dst_nodes[idx].item()
                
                if src_idx == dst_idx:
                    continue

                edge_set.add((src_idx, dst_idx))
                edge_set.add((dst_idx, src_idx))  # Undirected
                
                if edge_labels is not None:
                    edge_weight = edge_labels[idx].item()
                    if edge_weight <= 0:
                        continue
                else:
                    edge_weight = 1.0
                
                distance = self.compute_pairwise_distance(z, src_idx, dst_idx)
                total_pos_distance += edge_weight * distance
                num_pos_edges += edge_weight

            # Sample negative edges (no connection in edge_index)
            num_nodes = z.size(0)
            neg_samples = 0
            with tqdm(total=num_negative_samples, desc="Negative Sampling") as pbar:
                while neg_samples < num_negative_samples:
                    i = torch.randint(0, num_nodes, (1,)).item()
                    j = torch.randint(0, num_nodes, (1,)).item()
                    if i == j or (i, j) in edge_set:
                        continue
                    distance = self.compute_pairwise_distance(z, i, j)
                    penalty = 1.0 / (1.0 + (distance)**2)
                    total_neg_penalty += penalty
                    neg_samples += 1
                    pbar.update(1)

        elif "adj_matrix" in targets:
            adj_matrix = targets["adj_matrix"]
            num_negative_samples = int((adj_matrix != 0).sum().item() / 2)
            num_nodes = adj_matrix.size(0)

            for i in tqdm(range(num_nodes), desc="Positive Adjacency"):
                for j in range(i + 1, num_nodes):
                    edge_weight = adj_matrix[i, j].item()
                    if edge_weight > 0:
                        edge_set.add((i, j))
                        edge_set.add((j, i))
                        distance = self.compute_pairwise_distance(z, i, j)
                        total_pos_distance += edge_weight * (distance**2)
                        num_pos_edges += edge_weight

            # Sample negative edges
            neg_samples = 0
            with tqdm(total=num_negative_samples, desc="Negative Sampling") as pbar:
                while neg_samples < num_negative_samples:
                    i = torch.randint(0, num_nodes, (1,)).item()
                    j = torch.randint(0, num_nodes, (1,)).item()
                    if i == j or (i, j) in edge_set:
                        continue
                    distance = self.compute_pairwise_distance(z, i, j)
                    penalty = 1.0 / (1.0 + distance)
                    total_neg_penalty += penalty
                    neg_samples += 1
                    pbar.update(1)

        else:
            raise ValueError("Targets must contain either 'edge_index' or 'adj_matrix'")
        
        # Compute final loss
        if num_pos_edges > 0:
            avg_pos_distance = total_pos_distance / num_pos_edges
            pos_loss = avg_pos_distance
        else:
            pos_loss = torch.tensor(0.0, device=z.device, requires_grad=True)

        neg_loss = negative_distance_weight * (total_neg_penalty / num_negative_samples) #* ((2 * num_negative_samples) / (num_nodes*(num_nodes-1)))
        print("Loss: pos=", pos_loss, " neg=", neg_loss, " (", (neg_loss*100)/(pos_loss + neg_loss) , "%) total=", pos_loss + neg_loss)
        
        #return pos_loss * (neg_loss.detach() / (pos_loss.detach() + 1e-8)) + neg_loss
        return pos_loss + neg_loss
    
    def compute_jacobian(self, z: torch.Tensor, node_idx: int = None) -> torch.Tensor:
        """
        Compute Jacobian for this decoder (which is just identity since we output z)
        """
        if node_idx is not None:
            # Return identity matrix for single node
            return torch.eye(self.latent_dim, device=z.device, dtype=z.dtype)
        else:
            # Return identity for all nodes
            batch_size = z.size(0)
            return torch.eye(self.latent_dim, device=z.device, dtype=z.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    
    def invalidate_cache(self):
        """Invalidate all caches"""
        self._jacobian_cache.clear()
        self._metric_cache.clear()
        self._cache_valid = False
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get caching statistics"""
        return {
            "metric_cache_size": len(self._metric_cache),
            "jacobian_cache_size": len(self._jacobian_cache)
        }
    

class ManifoldHeatKernelDecoder(DecoderBase):
    """
    Fictive decoder that outputs latent codes but computes a meaningful loss 
    based on manifold heat kernel divergence with graph Laplacian.
    Uses proper Riemannian distance computation via line integration.
    """
    def __init__(
        self, 
        latent_dim: int,
        reference_decoder_name: str = "node_attr_decoder",
        heat_time: Union[float, List[float]] = 1.0,  # Time parameter for heat kernel
        num_eigenvalues: int = 50,  # Number of eigenvalues to compute
        num_integration_points: int = 10,  # For Riemannian distance integration
        metric_regularization: float = 1e-6,
        cache_jacobians: bool = True,
        # Simple caching parameters
        cache_tolerance: float = 1e-4,
        max_cache_size: int = 1000,
        # Heat kernel specific parameters
        laplacian_regularization: float = 1e-8,
        heat_kernel_approximation: str = "spectral",  # "spectral" or "finite_difference"
        finite_diff_steps: int = 100,
        # Manifold Laplacian construction
        manifold_neighbors: int = 10,  # How many nearest neighbors to connect in manifold
        gaussian_bandwidth: float = 1.0,  # Bandwidth for Gaussian weights
        name: str = "manifold_heat_kernel_decoder"
    ):
        super(ManifoldHeatKernelDecoder, self).__init__(latent_dim, name)
        
        self.reference_decoder_name = reference_decoder_name
        self.heat_times = [heat_time] if isinstance(heat_time, (float, int)) else list(heat_time)
        self.num_eigenvalues = num_eigenvalues
        self.num_integration_points = num_integration_points
        self.metric_regularization = metric_regularization
        self.cache_jacobians = cache_jacobians
        self.cache_tolerance = cache_tolerance
        self.max_cache_size = max_cache_size
        self.laplacian_regularization = laplacian_regularization
        self.heat_kernel_approximation = heat_kernel_approximation
        self.finite_diff_steps = finite_diff_steps
        self.manifold_neighbors = manifold_neighbors
        self.gaussian_bandwidth = gaussian_bandwidth
        
        # Simple cache: store (z_point_hash, metric_tensor) pairs
        self._metric_cache = {}
        self._jacobian_cache = {}
        self._riemannian_distance_cache = {}
        self._heat_kernel_cache = {}
        self._cache_valid = False
        self._reference_decoder = None
    
    def set_reference_decoder(self, decoder: "DecoderBase"):
        """Set the reference decoder used for Jacobian computation"""
        self._reference_decoder = decoder
        self._clear_all_caches()
    
    def _clear_all_caches(self):
        """Clear all caches"""
        self._jacobian_cache.clear()
        self._metric_cache.clear()
        self._riemannian_distance_cache.clear()
        self._heat_kernel_cache.clear()
        self._cache_valid = False
    
    def _get_cache_key(self, z_point: torch.Tensor, node_idx: int) -> str:
        """Generate a simple cache key from z_point and node_idx"""
        z_rounded = torch.round(z_point / self.cache_tolerance) * self.cache_tolerance
        z_str = "_".join([f"{x:.6f}" for x in z_rounded.cpu().numpy()])
        return f"{node_idx}_{z_str}"
    
    def _find_cached_metric(self, z_point: torch.Tensor, node_idx: int) -> Optional[torch.Tensor]:
        """Find cached metric tensor for similar z_point"""
        cache_key = self._get_cache_key(z_point, node_idx)
        
        if cache_key in self._metric_cache:
            cached_z, cached_metric = self._metric_cache[cache_key]
            if torch.norm(z_point - cached_z) < self.cache_tolerance:
                return cached_metric.to(device=z_point.device, dtype=z_point.dtype)
        
        return None
    
    def _cache_metric(self, z_point: torch.Tensor, node_idx: int, metric_tensor: torch.Tensor):
        """Cache a metric tensor"""
        if len(self._metric_cache) >= self.max_cache_size:
            oldest_key = next(iter(self._metric_cache))
            del self._metric_cache[oldest_key]
        
        cache_key = self._get_cache_key(z_point, node_idx)
        self._metric_cache[cache_key] = (z_point.detach().clone(), metric_tensor.detach().clone())
    
    def forward(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass - simply returns the latent codes (fictive decoder)"""
        return {"latent_codes": z.clone()}
    
    def compute_metric_tensor(self, z: torch.Tensor, node_idx: int) -> torch.Tensor:
        """
        Compute the Riemannian metric tensor at a given point in latent space
        """
        z_point = z[node_idx]
        
        # Try to find cached result first
        cached_metric = self._find_cached_metric(z_point, node_idx)
        if cached_metric is not None:
            # Preserve gradients by adding zero times actual computation
            actual_metric = self._compute_metric_raw(z, node_idx)
            return cached_metric + 0.0 * actual_metric
        
        # Compute fresh metric tensor
        metric_tensor = self._compute_metric_raw(z, node_idx)
        
        # Cache the result (detached)
        if self.cache_jacobians:
            self._cache_metric(z_point, node_idx, metric_tensor)
        
        return metric_tensor
    
    def _compute_metric_raw(self, z: torch.Tensor, node_idx: int) -> torch.Tensor:
        """Raw computation of metric tensor without caching"""
        if self._reference_decoder is None:
            return torch.eye(self.latent_dim, device=z.device, dtype=z.dtype)
        
        try:
            # Compute Jacobian of reference decoder at this point        
            jacobian = self._reference_decoder.compute_jacobian(z, node_idx)
            
            # Handle different Jacobian shapes
            if jacobian.dim() == 2:
                if jacobian.size(1) != self.latent_dim:
                    raise ValueError(f"Jacobian latent dimension {jacobian.size(1)} doesn't match expected {self.latent_dim}")
                metric_tensor = torch.mm(jacobian.t(), jacobian)
            elif jacobian.dim() == 3:
                if node_idx >= jacobian.size(0):
                    raise ValueError(f"Node index {node_idx} out of bounds for Jacobian with {jacobian.size(0)} nodes")
                node_jacobian = jacobian[node_idx]
                if node_jacobian.size(1) != self.latent_dim:
                    raise ValueError(f"Jacobian latent dimension {node_jacobian.size(1)} doesn't match expected {self.latent_dim}")
                metric_tensor = torch.mm(node_jacobian.t(), node_jacobian)
            elif jacobian.dim() == 1:
                if jacobian.size(0) == self.latent_dim:
                    metric_tensor = torch.outer(jacobian, jacobian)
                else:
                    raise ValueError(f"1D Jacobian size {jacobian.size(0)} doesn't match latent_dim {self.latent_dim}")
            else:
                raise ValueError(f"Unexpected Jacobian shape: {jacobian.shape}")
            
            # Add regularization for numerical stability
            metric_tensor += self.metric_regularization * torch.eye(
                self.latent_dim, device=z.device, dtype=z.dtype
            )
            
            return metric_tensor
            
        except Exception as e:
            print(f"Warning: Failed to compute Jacobian metric, falling back to identity: {e}")
            return torch.eye(self.latent_dim, device=z.device, dtype=z.dtype)
    
    def compute_riemannian_distance(
        self, 
        u: torch.Tensor, 
        v: torch.Tensor, 
        z_full: torch.Tensor,
        node_idx_u: int,
        node_idx_v: int
    ) -> torch.Tensor:
        """
        Compute proper Riemannian distance using line integration:
        d(u,v) = ∫₀¹ √((u-v)ᵀG(x(t))(u-v))dt where x(t) = tu + (1-t)v
        """
        # Check cache first
        #cache_key = f"dist_{node_idx_u}_{node_idx_v}_{hash((u.data.tobytes(), v.data.tobytes()))}"
        u_str = "_".join([f"{x:.6f}" for x in u.detach().cpu().numpy()])
        v_str = "_".join([f"{x:.6f}" for x in v.detach().cpu().numpy()])
        cache_key = f"dist_{node_idx_u}_{node_idx_v}_{u_str}_{v_str}"
        if cache_key in self._riemannian_distance_cache:
            cached_dist = self._riemannian_distance_cache[cache_key]
            # Preserve gradients
            actual_dist = self._compute_riemannian_distance_raw(u, v, z_full, node_idx_u, node_idx_v)
            return cached_dist + 0.0 * actual_dist
        
        distance = self._compute_riemannian_distance_raw(u, v, z_full, node_idx_u, node_idx_v)
        
        # Cache result
        if self.cache_jacobians and len(self._riemannian_distance_cache) < self.max_cache_size:
            self._riemannian_distance_cache[cache_key] = distance.detach().clone()
        
        return distance
    
    def _compute_riemannian_distance_raw(
        self, 
        u: torch.Tensor, 
        v: torch.Tensor, 
        z_full: torch.Tensor,
        node_idx_u: int,
        node_idx_v: int
    ) -> torch.Tensor:
        """Raw computation of Riemannian distance"""
        def integrand(t):
            # Interpolated point
            x_t = t * u + (1 - t) * v
            
            # Create temporary z matrix with interpolated point
            # Use midpoint node index for metric computation
            mid_idx = (node_idx_u + node_idx_v) // 2
            z_temp = z_full.clone()
            z_temp[mid_idx] = x_t
            
            try:
                # Compute metric tensor at interpolated point
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
    
    def compute_manifold_laplacian(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the Laplace-Beltrami operator on the manifold using proper Riemannian distances
        FIXED VERSION: Fully differentiable using Gaussian RBF weights
        """
        num_nodes = z.size(0)
        device = z.device
        
        # Compute all pairwise Riemannian distances
        print("Computing pairwise Riemannian distances...")
        distances = torch.zeros(num_nodes, num_nodes, device=device, dtype=z.dtype)
        
        for i in tqdm(range(num_nodes), desc="Computing distances"):
            for j in range(i + 1, num_nodes):
                dist = self.compute_riemannian_distance(z[i], z[j], z, i, j)
                distances[i, j] = dist
                distances[j, i] = dist  # Symmetric
        
        # FIXED: Use differentiable Gaussian RBF weights instead of k-NN
        # This creates a fully connected graph with exponentially decaying weights
        weights = torch.exp(-distances**2 / (2 * self.gaussian_bandwidth**2))
        
        # Zero out diagonal (no self-connections)
        weights = weights * (1 - torch.eye(num_nodes, device=device))
        
        # Optional: Threshold very small weights to maintain some sparsity
        threshold = 1e-4
        weights = torch.where(weights > threshold, weights, torch.zeros_like(weights))
        
        # Create Laplacian: L = D - W where D is degree matrix
        degree = torch.sum(weights, dim=1)
        L_manifold = torch.diag(degree) - weights
        
        # Add regularization
        L_manifold += self.laplacian_regularization * torch.eye(num_nodes, device=device)
        
        return L_manifold
    
    def compute_heat_kernel_spectral(self, laplacian: torch.Tensor, t: Union[float, List[float]]) -> torch.Tensor:
        """
        Compute heat kernel using spectral decomposition: K(t) = exp(-t*L)
        """
        try:
            # Eigendecomposition of Laplacian
            eigenvals, eigenvecs = torch.linalg.eigh(laplacian)
            
            # Clamp eigenvalues to avoid numerical issues
            eigenvals = torch.clamp(eigenvals, min=0.0)
            
            # Take only the first num_eigenvalues for efficiency
            if self.num_eigenvalues < eigenvals.size(0):
                eigenvals = eigenvals[:self.num_eigenvalues]
                eigenvecs = eigenvecs[:, :self.num_eigenvalues]
            
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
                        
        except Exception as e:
            print(f"Warning: Spectral heat kernel computation failed: {e}")
            # Fallback to identity
            return torch.eye(laplacian.size(0), device=laplacian.device, dtype=laplacian.dtype)
    
    def compute_heat_kernel_finite_diff(self, laplacian: torch.Tensor, t: Union[float, List[float]]) -> torch.Tensor:
        """
        Compute heat kernel using finite difference approximation of exp(-t*L)
        """
        def compute_single_kernel(t_i):
            dt = t_i / self.finite_diff_steps
            K = torch.eye(laplacian.size(0), device=laplacian.device, dtype=laplacian.dtype)
            for _ in range(self.finite_diff_steps):
                K = K - dt * (laplacian @ K)
            return K

        if isinstance(t, (float, int)):
            return compute_single_kernel(t)
        return [compute_single_kernel(t_i) for t_i in t]
    
    def compute_graph_laplacian(self, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
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
        
        # Compute degree matrix
        degree = torch.sum(adj, dim=1)
        D = torch.diag(degree)
        
        # Laplacian: L = D - A
        laplacian = D - adj
        
        # Add small regularization
        laplacian += self.laplacian_regularization * torch.eye(num_nodes, device=adj.device)
        
        return laplacian
    
    def compute_heat_kernel_divergence(
        self,
        K_manifold: Union[torch.Tensor, List[torch.Tensor]],
        K_graph:    Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Compute divergence between manifold heat kernel and graph heat kernel
        Using Frobenius norm of the difference
        """
        # If we got lists of kernels, compute per‐t and average:
        if isinstance(K_manifold, (list, tuple)):
            divergences = [
                self.compute_heat_kernel_divergence(Km, Kg)
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
        return torch.norm(diff, p='fro')
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        heat_kernel_weight: float = 1e6,
    ) -> torch.Tensor:
        """
        Compute loss based on heat kernel divergence between manifold and graph
        """
        z = outputs["latent_codes"]
        num_nodes = z.size(0)
        
        # Clear cache if needed
        if not self._cache_valid:
            self._clear_all_caches()
            self._cache_valid = True
        
        print("Computing graph Laplacian...")
        # Compute graph Laplacian
        L_graph = self.compute_graph_laplacian(targets)
        
        print("Computing manifold Laplacian...")
        # Compute manifold Laplacian (this will compute Riemannian distances)
        L_manifold = self.compute_manifold_laplacian(z)
        
        print("Computing heat kernels...")
        # Compute heat kernels
        if self.heat_kernel_approximation == "spectral":
            K_graph = self.compute_heat_kernel_spectral(L_graph, self.heat_times)
            K_manifold = self.compute_heat_kernel_spectral(L_manifold, self.heat_times)
        else:  # finite_difference
            K_graph = self.compute_heat_kernel_finite_diff(L_graph, self.heat_times)
            K_manifold = self.compute_heat_kernel_finite_diff(L_manifold, self.heat_times)
        
        print("Computing heat kernel divergence...")
        # Compute divergence
        divergence = self.compute_heat_kernel_divergence(K_manifold, K_graph)
        
        # Total loss
        total_loss = heat_kernel_weight * divergence
        
        print(f"Heat kernel divergence: {(divergence.item()):.6f}")
        print(f"Total loss: {(total_loss.item()):.6f}")
        
        return total_loss
    
    def compute_jacobian(self, z: torch.Tensor, node_idx: int = None) -> torch.Tensor:
        """
        Compute Jacobian for this decoder (identity since we output z)
        """
        if node_idx is not None:
            return torch.eye(self.latent_dim, device=z.device, dtype=z.dtype)
        else:
            batch_size = z.size(0)
            return torch.eye(self.latent_dim, device=z.device, dtype=z.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    
    def invalidate_cache(self):
        """Invalidate all caches"""
        self._clear_all_caches()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get caching statistics"""
        return {
            "metric_cache_size": len(self._metric_cache),
            "jacobian_cache_size": len(self._jacobian_cache),
            "riemannian_distance_cache_size": len(self._riemannian_distance_cache),
            "heat_kernel_cache_size": len(self._heat_kernel_cache)
        }
    
    def get_heat_kernels(self, z: torch.Tensor, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Utility method to get both heat kernels for analysis
        """
        L_graph = self.compute_graph_laplacian(targets)
        L_manifold = self.compute_manifold_laplacian(z)
        
        if self.heat_kernel_approximation == "spectral":
            K_graph = self.compute_heat_kernel_spectral(L_graph, self.heat_times)
            K_manifold = self.compute_heat_kernel_spectral(L_manifold, self.heat_times)
        else:
            K_graph = self.compute_heat_kernel_finite_diff(L_graph, self.heat_times)
            K_manifold = self.compute_heat_kernel_finite_diff(L_manifold, self.heat_times)
        
        return {
            "graph_laplacian": L_graph,
            "manifold_laplacian": L_manifold,
            "graph_heat_kernel": K_graph,
            "manifold_heat_kernel": K_manifold,
            "riemannian_distances": self._get_distance_matrix(z)
        }
    
    def _get_distance_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """Get the full Riemannian distance matrix (for analysis)"""
        num_nodes = z.size(0)
        distances = torch.zeros(num_nodes, num_nodes, device=z.device, dtype=z.dtype)
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = self.compute_riemannian_distance(z[i], z[j], z, i, j)
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances