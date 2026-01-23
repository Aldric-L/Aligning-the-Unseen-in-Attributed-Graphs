import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from tqdm import tqdm
from torch.autograd.functional import jacobian
import weakref
import time
from torch_geometric.nn import GCNConv

from framework.preprocessing import PreprocessingLayer
from framework.utils import get_adjacency_matrix_from_tensors
from framework.autograd import _mlp_forward_with_jacobian

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
        # base_loss = self.compute_loss(outputs, targets)
        # loss_components = {'base_loss': base_loss}

        loss_output = self.compute_loss(outputs, targets)
        if isinstance(loss_output, dict):
            loss_components = {'base_loss': loss_output['final_loss']}
            loss_components.update({k: v for k, v in loss_output.items() if k != 'final_loss'})
        else:
            loss_components = {'base_loss': loss_output}
        total_loss = loss_components['base_loss']

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
        #z_detached = z.detach().clone().requires_grad_(True)
        
        # Forward pass to get node feature predictions
        outputs = self.forward(z)
        node_features = outputs["node_features"]
        
        # Handle single node case
        if node_idx is not None:
            node_features = node_features[node_idx]
            
            # Initialize Jacobian matrix (output_dim × latent_dim)
            jacobian = torch.zeros(self.output_dim, z.size(1), device=z.device)
            
            # Compute Jacobian row by row
            for i in range(self.output_dim):
                # Zero gradients
                if z.grad is not None:
                    z.grad.zero_()
                
                # Backward for this feature dimension
                node_features[i].backward(retain_graph=True)
                
                # Store gradient (Jacobian row)
                jacobian[i] = z.grad[node_idx].clone()
        
        else:
            # Full Jacobian for all nodes
            batch_size = z.size(0)
            
            # Initialize Jacobian tensor (nodes × output_dim × latent_dim)
            jacobian = torch.zeros(batch_size, self.output_dim, z.size(1), device=z.device)
            
            # For each node and feature dimension
            for n in range(batch_size):
                for i in range(self.output_dim):
                    # Zero gradients
                    if z.grad is not None:
                        z.grad.zero_()
                    
                    # Backward for this specific element
                    node_features[n, i].backward(retain_graph=True)
                    
                    # Store gradient
                    jacobian[n, i] = z.grad[n].clone()
        
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
        

class NodeAttributeVariationalDecoder(DecoderBase):
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
        name: str = "node_attr_decoder",
        loss_options: dict = {
        "lambda_comp_variance": 5,
        "lambda_decoder_variance":100.0,
        "debug": False},
        clip_var: float = -1
    ):
        super(NodeAttributeVariationalDecoder, self).__init__(latent_dim, name)
        
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = activation
        self.final_activation = final_activation
        
        # Build MLP layers
        layers = []
        in_features = latent_dim
        
        for dim in hidden_dims:
            l = nn.Linear(in_features, dim)
            torch.nn.init.xavier_uniform_(l.weight)
            # It's common to initialize biases to zero
            if l.bias is not None:
                torch.nn.init.constant_(l.bias, 0)
            layers.append(l)
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = dim
        
        #layers.append(nn.Linear(in_features, output_dim))
        self.mlp = nn.Sequential(*layers)

        self.mean_head = nn.Linear(in_features, output_dim)
        torch.nn.init.xavier_uniform_(self.mean_head.weight)
        if self.mean_head.bias is not None:
            torch.nn.init.constant_(self.mean_head.bias, 0)

        self.logvar_head = nn.Linear(in_features, output_dim)
        torch.nn.init.xavier_uniform_(self.logvar_head.weight)
        if self.logvar_head.bias is not None:
            torch.nn.init.constant_(self.logvar_head.bias, 0)

        self.loss_options = loss_options
        self.clip_var = clip_var
    
    def forward(self, z: torch.Tensor, sample:bool = False,
        eps_var: float = 1e-8, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Reconstruct node attributes from latent embeddings
        
        Args:
            z: Latent node embeddings [num_nodes, latent_dim]
            
        Returns:
            Dict with node feature predictions
        """
        h = self.mlp(z)
        mu_x = self.mean_head(h)
        logvar_x = self.logvar_head(h)

        if torch.isnan(h).any() or torch.isinf(h).any():
            print("[Variational Decoder DEBUG] NaNs or Infs in h!", h.min().item(), h.max().item())
        if torch.isnan(mu_x).any() or torch.isinf(mu_x).any():
            print("[Variational Decoder DEBUG] NaNs or Infs in mu_x!", mu_x.min().item(), mu_x.max().item())
        if torch.isnan(logvar_x).any() or torch.isinf(logvar_x).any():
            print("[Variational Decoder DEBUG] NaNs or Infs in logvar_x!", logvar_x.min().item(), logvar_x.max().item())

        if self.clip_var != -1:
            logvar_x = torch.clip(logvar_x, min=-self.clip_var, max=self.clip_var)
        
        # Apply final activation if specified
        if self.final_activation is not None:
            mu_x = self.final_activation(mu_x)
        
        recon = torch.distributions.Normal(mu_x, torch.exp(0.5 * logvar_x + eps_var)).rsample() if sample else mu_x 

        return {
            "node_features": recon,
            "node_features_mu": mu_x,
            "node_features_logvar": logvar_x
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        eps_var: float = 1e-8
    ) -> torch.Tensor:
        """
        Loss = 
        1) - E_q[log p(x|z)]                            # Gaussian NLL
        + 2) lambda_comp_variance * mean_i[(Var_d mu_i - Var_d x_i)^2]
        + 3) lambda_decoder_variance * mean_i[(Mean_d sigma^2_i - (Var_d x_i - Var_d mu_i))^2]
        """
        lambda_comp_variance = self.loss_options.get("lambda_comp_variance", 5)
        lambda_decoder_variance = self.loss_options.get("lambda_decoder_variance", 100)
        debug = self.loss_options.get("debug", False)
        mu     = outputs["node_features_mu"]      # [B, D]
        logvar = outputs["node_features_logvar"]  # [B, D]
        x      = targets["node_features"]         # [B, D]
        B, D   = x.shape

        if torch.isnan(mu).any() or torch.isinf(mu).any():
            print("[LOSS DEBUG] NaNs or Infs in mu!", mu.min().item(), mu.max().item())
        if torch.isnan(logvar).any() or torch.isinf(logvar).any():
            print("[LOSS DEBUG] NaNs or Infs in logvar!", logvar.min().item(), logvar.max().item())

        # 1) Gaussian NLL
        std   = torch.exp(0.5 * logvar) + eps_var
        if torch.isnan(std).any() or torch.isinf(std).any():
            print("[LOSS DEBUG] NaNs or Infs in std!", std.min().item(), std.max().item())
        dist  = torch.distributions.Normal(mu, std)
        log_p = dist.log_prob(x)                  # [B, D]
        recon_nll = -log_p.sum(dim=1).mean()      # scalar

        # empirical variance of each target vector
        x_mean  = x.mean(dim=1, keepdim=True)     # [B,1]
        var_emp = torch.sum((x - x_mean)**2, dim=1) / D  # [B]

        # 2) Component-wise variance of mu
        mu_mean = mu.mean(dim=1, keepdim=True)    # [B,1]
        var_mu  = torch.sum((mu - mu_mean)**2, dim=1) / D  # [B]
        comp_var_pen = torch.mean((var_mu - var_emp)**2)   # scalar

        # 3) Decoder-noise variance: match E[sigma^2] to (var_emp - var_mu)
        sigma2      = torch.exp(logvar)                  # [B, D]
        mean_sigma2 = sigma2.mean(dim=1)                 # [B]
        # residual variance that the noise head must explain
        residual = torch.clamp(var_emp - var_mu, min=eps_var)  # [B]
        dec_var_pen = torch.mean((mean_sigma2 - residual)**2)  # scalar

        if debug:
            print(f"lambda_decoder_variance = {lambda_decoder_variance}, lambda_comp_variance = {lambda_comp_variance}")
            print(f"NLL={recon_nll.item():.2f}, comp_var_pen={lambda_comp_variance*comp_var_pen.item():.2f} ({comp_var_pen.item():.2f}), var_pen={lambda_decoder_variance*dec_var_pen.item():.2f} ({dec_var_pen.item():.2f}), final pen={lambda_comp_variance*comp_var_pen.item() + lambda_decoder_variance*dec_var_pen.item():.2f}")

        return (
            recon_nll
            + lambda_comp_variance    * comp_var_pen
            + lambda_decoder_variance * dec_var_pen
        )
        
    def compute_jacobian(self, z: torch.Tensor, node_idx: Optional[int] = None, mode="total") -> torch.Tensor:
        """Ultra-fast analytical jacobian - no autodiff needed"""
        z0 = z.detach()

        # Get weights/biases
        W_mu = self.mean_head.weight  # [D, H]
        W_lv = self.logvar_head.weight  # [D, H]
        b_lv = self.logvar_head.bias
        if node_idx is not None or z0.ndim == 1:
            # Single node case
            idx = 0 if z0.ndim == 1 else node_idx
            z_node = z0[idx:idx+1]  # [1, L]

            # Forward pass through MLP storing activations
            h, J_h = _mlp_forward_with_jacobian(x=z_node,
                                                mlp=self.mlp,
                                                training_mode=self.training,
                                                dropout=getattr(self, "dropout", 0))  # h: [1, H], J_h: [1, H, L]
            h = h[0]       # [H]
            J_h = J_h[0]   # [H, L]

            # Compute sigma
            lv_node = (W_lv @ h) + b_lv  # [D]
            sigma_node = torch.exp(0.5 * lv_node)  # [D]

            # Jacobian computation
            J_mu = W_mu @ J_h  # [D, L]
            factor = 0.5 * sigma_node[:, None] * W_lv  # [D, H]
            J_sigma = factor @ J_h  # [D, L]

            if mode == "total":
                return J_mu + J_sigma
            return J_mu, J_sigma

        else:
            # Batch case
            h, J_h = _mlp_forward_with_jacobian(x=z0,
                                                mlp=self.mlp,
                                                training_mode=self.training,
                                                dropout=getattr(self, "dropout", 0))  # h: [N, H], J_h: [N, H, L]

            # Compute sigma for all nodes
            lv = torch.matmul(h, W_lv.t()) + b_lv  # [N, D]
            sigma = torch.exp(0.5 * lv)  # [N, D]

            # Batch jacobian computation
            J_mu = torch.einsum('dh,nhl->ndl', W_mu, J_h)  # [N, D, L]
            factor = 0.5 * sigma.unsqueeze(2) * W_lv.unsqueeze(0)  # [N, D, H]
            J_sigma = torch.einsum('ndh,nhl->ndl', factor, J_h)  # [N, D, L]

            if mode == "total":
                return J_mu + J_sigma
            return J_mu, J_sigma

    
    def check_full_rank(self):
        total_is_full_rank = True
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                weight_matrix = module.weight
                rank = torch.linalg.matrix_rank(weight_matrix)
                rows, cols = weight_matrix.shape
                min_dim = min(rows, cols)
                is_full_rank = rank == min_dim
                total_is_full_rank &= is_full_rank
                print(f"Layer: {name}")
                print(f"Weight matrix shape: {weight_matrix.shape}")
                print(f"Rank: {rank.item()}")
                print(f"Min dimension: {min_dim}")
                print(f"Is full rank? {'✅ Yes' if is_full_rank else '❌ No'}")
                print("-" * 30)
        return total_is_full_rank
    


class InnerProductAdjacencyDecoder(DecoderBase):
    """
    Standard inner-product decoder for Variational Graph Autoencoders (VGAE).
    Reconstructs the adjacency matrix A via A_hat = sigmoid(Z * Z^T).
    """
    def __init__(self, latent_dim: int, name: str = "adj_decoder", dropout: float = 0.0):
        super(InnerProductAdjacencyDecoder, self).__init__(latent_dim, name)
        self.dropout = dropout

    def forward(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Computes the reconstructed adjacency logits.

        Args:
            z: Latent embeddings [num_nodes, latent_dim]

        Returns:
            Dict containing:
                - 'adj_logits': Raw logits (Z * Z^T) [num_nodes, num_nodes]
                - 'adj_probs': Sigmoid probabilities [num_nodes, num_nodes]
        """
        # Apply dropout to Z if configured (rare in vanilla VGAE but sometimes useful)
        z = F.dropout(z, self.dropout, training=self.training)
        
        # Standard inner product decoder: Z * Z^T
        adj_logits = torch.matmul(z, z.t())
        
        return {
            'adj_logits': adj_logits,
            'adj_probs': torch.sigmoid(adj_logits)
        }

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], 
                     pos_weight: float = 1.0, norm: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Computes binary cross entropy loss for adjacency reconstruction.
        
        Note: The standard VGAE loss usually weights the positive examples (edges) 
        higher because graphs are sparse.
        
        Args:
            outputs: Must contain 'adj_logits'
            targets: Must contain 'adj_matrix' (dense or sparse)
            pos_weight: Weight for positive class (edges) to handle sparsity.
                        Usually calculated as: (num_nodes^2 - num_edges) / num_edges
            norm: Normalization factor for the loss.
                  Usually calculated as: num_nodes^2 / ((num_nodes^2 - num_edges) * 2)

        Returns:
            Dict with 'final_loss' and 'bce_loss'.
        """
        adj_logits = outputs['adj_logits']

        if "edge_index" in targets:
            target_adj = get_adjacency_matrix_from_tensors(targets["edge_index"], targets["edge_labels"])
        elif "adj_matrix" in targets:
            target_adj = targets["adj_matrix"]
        else:
            raise ValueError("Targets must contain either 'edge_index' or 'adj_matrix'")
        
        # Ensure target is dense for BCEWithLogitsLoss
        if target_adj.is_sparse:
            target_adj = target_adj.to_dense()

        # Handle pos_weight as a tensor for PyTorch API compatibility
        if not isinstance(pos_weight, torch.Tensor):
            pos_weight_tensor = torch.tensor([pos_weight], device=adj_logits.device)
        else:
            pos_weight_tensor = pos_weight

        # Weighted Binary Cross Entropy with Logits
        # We multiply by norm (as per Kingma/Welling) to keep scale consistent with KL
        loss = norm * F.binary_cross_entropy_with_logits(
            adj_logits, 
            target_adj, 
            pos_weight=pos_weight_tensor
        )
        
        return {
            'final_loss': loss,
            'bce_unweighted': loss / norm  # Useful for debugging unscaled loss
        }
    
class GraphGCNDecoder(DecoderBase):
    """
    GCN-based Decoder for Variational Graph Autoencoders (VGAE).
    
    Unlike the standard inner-product decoder (which reconstructs Adjacency A),
    a GCN decoder is typically used to reconstruct Node Features (X) by 
    propagating the latent variables Z over the graph structure.
    
    Forward Path: Z, Edge_Index -> GCN Layers -> X_hat
    """
    def __init__(
        self, 
        latent_dim: int, 
        hidden_dims: List[int], 
        output_dim: int, 
        name: str = "node_attr_decoder", 
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU()
    ):
        """
        Args:
            latent_dim: Dimension of input latent embeddings (Z)
            hidden_dims: List of hidden dimensions for the decoder layers.
                         Typically the reverse of the encoder.
            output_dim: Dimension of the original node features (X) to reconstruct.
            dropout: Dropout probability.
            activation: Activation function.
        """
        super(GraphGCNDecoder, self).__init__(latent_dim, name)
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = activation
        
        self.gcn_layers = nn.ModuleList()
        
        # Build decoder layers (Latent -> Hidden -> ... -> Output)
        in_channels = latent_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            self.gcn_layers.append(GCNConv(in_channels, hidden_dim))
            in_channels = hidden_dim
            
        # Final reconstruction layer mapping to original feature space
        self.final_layer = GCNConv(in_channels, output_dim)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Reconstructs node features using the latent embeddings and graph topology.

        Args:
            z: Latent embeddings [num_nodes, latent_dim]
            edge_index: Graph connectivity [2, num_edges] (Required for GCN)

        Returns:
            Dict containing:
                - 'recon_features': Reconstructed node features [num_nodes, output_dim]
        """
        x = z
        
        # Iterate through hidden GCN layers
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final projection to feature space
        # Note: We usually don't apply activation/dropout to the final reconstruction 
        # unless features are bounded (e.g., sigmoid for binary features)
        recon_x = self.final_layer(x, edge_index)
        
        return {
            'recon_features': recon_x
        }

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], 
                     **kwargs) -> Dict[str, torch.Tensor]:
        """
        Computes Mean Squared Error (MSE) for feature reconstruction.

        Args:
            outputs: Must contain 'recon_features'
            targets: Must contain 'node_features' (the original X)

        Returns:
            Dict with 'final_loss' and 'mse_loss'.
        """
        if 'recon_features' not in outputs:
            raise ValueError("GCN Decoder outputs must contain 'recon_features'")
            
        recon_x = outputs['recon_features']
        
        if 'node_features' in targets:
            target_x = targets['node_features']
        elif 'x' in targets:
            target_x = targets['x']
        else:
            raise ValueError("Targets must contain 'node_features' or 'x' for GCN decoding.")

        # Compute MSE Loss
        loss = F.mse_loss(recon_x, target_x)
        
        return {
            'final_loss': loss,
            'mse_loss': loss
        }

from framework.geometry import compute_graph_laplacian_from_targets


class ManifoldHeatKernelDecoder(DecoderBase):
    """
    Fictive decoder that outputs latent codes but computes a meaningful loss 
    based on manifold heat kernel divergence with graph Laplacian.
    """
    def __init__(
        self, 
        latent_dim: int,
        distance_mode: str = "linear_interpolation",  # or "direct" or "dijkstra"
        num_integration_points: int = 25,
        name: str = "manifold_heat_kernel_decoder",
        num_heat_time: int = 20,  # Time parameter for heat kernel
        num_eigenvalues: int = 50,  # Number of eigenvalues to compute
        laplacian_regularization: float = 1e-8,
        manifold_neighbors: int = 7,  # How many nearest neighbors to connect in manifold
        ema_lag_factor: float = 0.08,
        max_ema_epochs : int = 300,
        ema_inactivity_threshold: int = 20,
        dist_distorsion_penalty: float = 0.0,
        retain_high_freq_threshold: float = 0.9,
        suppress_low_freq_threshold: float = 5e-3,
    ):
        super(ManifoldHeatKernelDecoder, self).__init__(latent_dim, name)
        
        self.distance_mode = distance_mode
        self.num_integration_points = num_integration_points
        self.model = None  # Weak reference to the model instance

        self.num_heat_time = num_heat_time
        self.num_eigenvalues = num_eigenvalues
        self.laplacian_regularization = laplacian_regularization
        self.manifold_neighbors = manifold_neighbors
        self.sigma_ema = None
        self.L_graph = None
        self.K_graph = None
        self.lag_factor = ema_lag_factor
        self.first_sigma = None
        self.heat_times = None
        self.freeze_sigma = 0
        self.sigma_inactivity_threshold = ema_inactivity_threshold
        self.max_ema_epochs = max_ema_epochs
        self.ema_epochs = 0
        self.mean_distance = None
        self.dist_distorsion_penalty = dist_distorsion_penalty
        self.retain_high_freq_threshold = retain_high_freq_threshold
        self.suppress_low_freq_threshold = suppress_low_freq_threshold

    def giveVAEInstance(self, model):
        self.model = weakref.ref(model)

    def forward(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass - simply returns the latent codes (fictive decoder)"""
        return {"latent_codes": z.clone()}
        
    def _riemannian_distance(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        #print(f"[riemannian_distance] Computing pairwise distances for batch size {z.shape[0]}")
        #start_time = time.time()
        if self.model is None or self.model() is None:
            raise ValueError("Model instance not set. Call giveVAEInstance(model) first.")
        model = self.model()

        distances = []
        if z1.ndim == 1:
            z1 = z1.unsqueeze(0)
        if z2.ndim == 1:
            z2 = z2.unsqueeze(0)
        if self.distance_mode == "linear_interpolation":
            return model.get_latent_manifold().linear_interpolation_distance(z1, z2, num_points=self.num_integration_points)
        else:
            for i, (u_vec, v_vec) in enumerate(zip(z1, z2)):
                if self.distance_mode == "linear_interpolation":
                    #d = DistanceApproximations.linear_interpolation_distance(model.get_latent_manifold(), u_vec, v_vec, 
                    #                                                         num_points=self.num_integration_points)
                    d = model.get_latent_manifold().linear_interpolation_distance(u_vec, v_vec, 
                                                                                    num_points=self.num_integration_points)
                else:
                    d = model.get_latent_manifold().exact_geodesic_distance(u_vec, v_vec)
                #print(f"  [distance] Sample {i}: {d.item():.4f}")
                distances.append(d)
        #print(f"[riemannian_distance] Done in {time.time() - start_time:.4f}s")
        return torch.stack(distances)
    
    def compute_distance_matrix(self, z: torch.Tensor, batch_size=8) -> torch.Tensor:
        num_nodes = z.size(0)
        device = z.device
        #print("Computing pairwise Riemannian distances...")
        if self.distance_mode == "dijkstra" or self.distance_mode == "Dijkstra":
            distances = self.model().get_latent_manifold().get_grid_as_graph().compute_shortest_paths(
                self.model().get_latent_manifold()._clamp_point_to_bounds(z),
                weight_type="geodesic",  
                max_grid_neighbors=8,     # Connect to up to 8 nearest grid nodes
                num_threads=None
            )
        else:
            distances = torch.zeros(num_nodes, num_nodes, device=z.device)
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
                u_batch = z[current_row_indices]
                v_batch = z[current_col_indices]

                # Compute distances for this mini-batch
                current_batch_distances = self._riemannian_distance(
                    u_batch, v_batch
                )

                # Populate the distance matrix with the results of this batch
                distances[current_row_indices, current_col_indices] = current_batch_distances
                distances[current_col_indices, current_row_indices] = current_batch_distances # Symmetric

        return distances
    
    def get_spectral_heat_times(self, eigenvalues, num_times=5):
        # Filter out the zero eigenvalue (connected component)
        valid_eigs = eigenvalues[eigenvalues > 1e-5]
        
        lambda_max = valid_eigs.max()
        lambda_min = valid_eigs.min() # This is roughly lambda_2
        
        # Invert to get time scales
        t_min = 1.0 / lambda_max
        t_max = 1.0 / lambda_min
        
        # Create log-spaced times
        # We often multiply t_max by a factor (e.g., 2 or 4) to ensure full global coverage
        return torch.logspace(torch.log10(t_min), torch.log10(t_max * 4), num_times)
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute loss based on heat kernel divergence between manifold and graph
        """
        z = outputs["latent_codes"]
        distances = self.compute_distance_matrix(z)

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

        if self.heat_times is None:
            with torch.no_grad():
                self.L_graph = compute_graph_laplacian_from_targets(targets, normalize=True, laplacian_regularization=self.laplacian_regularization)
                if self.heat_times is None:
                    self.heat_times = self.get_spectral_heat_times(eigenvalues=torch.clamp(torch.linalg.eigvalsh(self.L_graph),min=0.0), num_times=self.num_heat_time)
                print("Selected heat times:", self.heat_times)
                #self.K_graph = compute_heat_kernel_from_laplacian(self.L_graph, self.heat_times)

        dist_sq = distances.pow(2).unsqueeze(-1)
        coeffs = (4 * torch.pi * self.heat_times).pow(-self.latent_dim / 2)
        exp_term = torch.exp(-dist_sq / (4 * self.heat_times))
        kernel_stack = coeffs * exp_term
        #manifold_heat_kernels = list(torch.unbind(kernel_stack, dim=-1))

        # Compute the divergence between the manifold and graph heat kernels
        divergence = (adj-kernel_stack.sum(dim=-1)).pow(2).sum()
        
        print(f"Current HK loss: {(divergence.item()):.6f}")
        return {'final_loss': divergence}


    # def compute_loss(
    #     self,
    #     outputs: Dict[str, torch.Tensor],
    #     targets: Dict[str, torch.Tensor],
    # ) -> torch.Tensor:
    #     """
    #     Compute loss based on heat kernel divergence between manifold and graph
    #     """
    #     z = outputs["latent_codes"]
    #     distances = self.compute_distance_matrix(z)
    #     if self.L_graph is None or self.heat_times is None:
    #         with torch.no_grad():
    #             self.L_graph = compute_graph_laplacian_from_targets(targets, normalize=True, laplacian_regularization=self.laplacian_regularization)
    #             if self.heat_times is None:
    #                 self.heat_times = self.get_spectral_heat_times(eigenvalues=torch.clamp(torch.linalg.eigvalsh(self.L_graph),min=0.0), num_times=self.num_heat_time)
    #             print("Selected heat times:", self.heat_times)
    #             self.K_graph = compute_heat_kernel_from_laplacian(self.L_graph, self.heat_times)

    #     dist_sq = distances.pow(2).unsqueeze(-1)
    #     coeffs = (4 * torch.pi * self.heat_times).pow(-self.latent_dim / 2)
    #     exp_term = torch.exp(-dist_sq / (4 * self.heat_times))
    #     kernel_stack = coeffs * exp_term
    #     manifold_heat_kernels = list(torch.unbind(kernel_stack, dim=-1))

    #     # Compute the divergence between the manifold and graph heat kernels
    #     divergence = 0
    #     for i, (K_manifold, K_graph) in enumerate(zip(manifold_heat_kernels, self.K_graph)):
    #         #print("For heat time", self.heat_times[i].item(), ":", torch.norm(K_manifold - K_graph, p=1))
    #         divergence += torch.norm(K_manifold - K_graph, p=1)
    #     print(f"Current HK loss: {(divergence.item()):.6f}")
    #     return {'final_loss': divergence}

# class ManifoldHeatKernelDecoder(DecoderBase):
#     """
#     Fictive decoder that outputs latent codes but computes a meaningful loss 
#     based on manifold heat kernel divergence with graph Laplacian.
#     """
#     def __init__(
#         self, 
#         latent_dim: int,
#         distance_mode: str = "linear_interpolation",  # or "direct" or "dijkstra"
#         num_integration_points: int = 25,
#         name: str = "manifold_heat_kernel_decoder",
#         num_heat_time: int = 20,  # Time parameter for heat kernel
#         num_eigenvalues: int = 50,  # Number of eigenvalues to compute
#         laplacian_regularization: float = 1e-8,
#         manifold_neighbors: int = 7,  # How many nearest neighbors to connect in manifold
#         ema_lag_factor: float = 0.08,
#         max_ema_epochs : int = 300,
#         ema_inactivity_threshold: int = 20,
#         dist_distorsion_penalty: float = 0.0,
#         retain_high_freq_threshold: float = 0.9,
#         suppress_low_freq_threshold: float = 5e-3,
#     ):
#         super(ManifoldHeatKernelDecoder, self).__init__(latent_dim, name)
        
#         self.distance_mode = distance_mode
#         self.num_integration_points = num_integration_points
#         self.model = None  # Weak reference to the model instance

#         self.num_heat_time = num_heat_time
#         self.num_eigenvalues = num_eigenvalues
#         self.laplacian_regularization = laplacian_regularization
#         self.manifold_neighbors = manifold_neighbors
#         self.sigma_ema = None
#         self.L_graph = None
#         self.K_graph = None
#         self.lag_factor = ema_lag_factor
#         self.first_sigma = None
#         self.heat_times = None
#         self.freeze_sigma = 0
#         self.sigma_inactivity_threshold = ema_inactivity_threshold
#         self.max_ema_epochs = max_ema_epochs
#         self.ema_epochs = 0
#         self.mean_distance = None
#         self.dist_distorsion_penalty = dist_distorsion_penalty
#         self.retain_high_freq_threshold = retain_high_freq_threshold
#         self.suppress_low_freq_threshold = suppress_low_freq_threshold

#     def giveVAEInstance(self, model):
#         self.model = weakref.ref(model)

#     def forward(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
#         """Forward pass - simply returns the latent codes (fictive decoder)"""
#         return {"latent_codes": z.clone()}
        
#     def _riemannian_distance(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
#         #print(f"[riemannian_distance] Computing pairwise distances for batch size {z.shape[0]}")
#         #start_time = time.time()
#         if self.model is None or self.model() is None:
#             raise ValueError("Model instance not set. Call giveVAEInstance(model) first.")
#         model = self.model()

#         distances = []
#         if z1.ndim == 1:
#             z1 = z1.unsqueeze(0)
#         if z2.ndim == 1:
#             z2 = z2.unsqueeze(0)
#         if self.distance_mode == "linear_interpolation":
#             return model.get_latent_manifold().linear_interpolation_distance(z1, z2, num_points=self.num_integration_points)
#         else:
#             for i, (u_vec, v_vec) in enumerate(zip(z1, z2)):
#                 if self.distance_mode == "linear_interpolation":
#                     #d = DistanceApproximations.linear_interpolation_distance(model.get_latent_manifold(), u_vec, v_vec, 
#                     #                                                         num_points=self.num_integration_points)
#                     d = model.get_latent_manifold().linear_interpolation_distance(u_vec, v_vec, 
#                                                                                     num_points=self.num_integration_points)
#                 else:
#                     d = model.get_latent_manifold().exact_geodesic_distance(u_vec, v_vec)
#                 #print(f"  [distance] Sample {i}: {d.item():.4f}")
#                 distances.append(d)
#         #print(f"[riemannian_distance] Done in {time.time() - start_time:.4f}s")
#         return torch.stack(distances)
    
#     def compute_distance_matrix(self, z: torch.Tensor, batch_size=8) -> torch.Tensor:
#         num_nodes = z.size(0)
#         device = z.device
#         #print("Computing pairwise Riemannian distances...")
#         if self.distance_mode == "dijkstra" or self.distance_mode == "Dijkstra":
#             distances = self.model().get_latent_manifold().get_grid_as_graph().compute_shortest_paths(
#                 self.model().get_latent_manifold()._clamp_point_to_bounds(z),
#                 weight_type="geodesic",  
#                 max_grid_neighbors=8,     # Connect to up to 8 nearest grid nodes
#                 num_threads=6
#             )
#         else:
#             distances = torch.zeros(num_nodes, num_nodes, device=z.device)
#             row_indices, col_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
#             total_pairs = len(row_indices)

#             # Process in mini-batches
#             for i in tqdm(range(0, total_pairs, batch_size), desc="Computing distances in batches"):
#                 batch_start = i
#                 batch_end = min(i + batch_size, total_pairs)

#                 # Get the current batch of indices
#                 current_row_indices = row_indices[batch_start:batch_end]
#                 current_col_indices = col_indices[batch_start:batch_end]

#                 # Extract the corresponding node embeddings for these pairs
#                 u_batch = z[current_row_indices]
#                 v_batch = z[current_col_indices]

#                 # Compute distances for this mini-batch
#                 current_batch_distances = self._riemannian_distance(
#                     u_batch, v_batch
#                 )

#                 # Populate the distance matrix with the results of this batch
#                 distances[current_row_indices, current_col_indices] = current_batch_distances
#                 distances[current_col_indices, current_row_indices] = current_batch_distances # Symmetric

#         return distances
    
#     def compute_loss(
#         self,
#         outputs: Dict[str, torch.Tensor],
#         targets: Dict[str, torch.Tensor],
#     ) -> torch.Tensor:
#         """
#         Compute loss based on heat kernel divergence between manifold and graph
#         """
#         z = outputs["latent_codes"]
#         num_nodes = z.size(0)
#         ht_recomputed = False
        
#         distances = self.compute_distance_matrix(z)
#         if self.mean_distance is None:
#             with torch.no_grad():
#                 self.mean_distance = distances.mean()

#         if (self.freeze_sigma < self.sigma_inactivity_threshold and self.ema_epochs < self.max_ema_epochs)  or self.sigma_ema is None:
#             with torch.no_grad():
#                 sigma = compute_sigma_with_knn(distances=distances, knn_for_sigma=self.manifold_neighbors)
#             if self.sigma_ema is None:
#                 self.sigma_ema = sigma
#                 self.first_sigma = sigma
#             elif self.freeze_sigma < self.sigma_inactivity_threshold and self.ema_epochs < self.max_ema_epochs:
#                 new_sigma = (1 - self.lag_factor) * self.sigma_ema + self.lag_factor * sigma
#                 if abs(self.sigma_ema - new_sigma) / self.sigma_ema < 5e-3:
#                     self.freeze_sigma += 1
#                     if self.freeze_sigma >= self.sigma_inactivity_threshold:
#                         print("Freezing sigma for non-interesting changes.")
#                         L_manifold = compute_manifold_laplacian(distances=distances,
#                                             sigma=self.sigma_ema,
#                                             laplacian_regularization=self.laplacian_regularization)
#                         with torch.no_grad():
#                             self.heat_times, diag = compute_heat_time_scale_from_laplacian(L=L_manifold, num_times=self.num_heat_time, 
#                                                                                            retain_high_freq_threshold=self.retain_high_freq_threshold, 
#                                                                                            suppress_low_freq_threshold = self.suppress_low_freq_threshold)
#                             print("Selected heat times:", self.heat_times)
#                             self.K_graph = compute_heat_kernel_from_laplacian(self.L_graph, self.heat_times)
#                             ht_recomputed = True
#                         K_manifold = compute_heat_kernel_from_laplacian(L_manifold, self.heat_times)
#                     else:
#                         self.sigma_ema = new_sigma
#                 else:
#                     self.sigma_ema = new_sigma
#                     self.freeze_sigma = 0
#             print("Current sigma: ", sigma.item(), "Selected sigma: ", self.sigma_ema.item())
#             self.ema_epochs += 1

#         if not ht_recomputed:
#             L_manifold = compute_manifold_laplacian(distances=distances,
#                                                     sigma=self.sigma_ema,
#                                                     laplacian_regularization=self.laplacian_regularization)

#         #print(f"Sparsity Percentage: {torch.sum(L_manifold <= 1e-5).item()/L_manifold.numel():.2f}%")

#         if self.heat_times is None:
#             with torch.no_grad():
#                 self.heat_times, diag = compute_heat_time_scale_from_laplacian(L=L_manifold, num_times=self.num_heat_time, 
#                                                                                retain_high_freq_threshold=self.retain_high_freq_threshold, 
#                                                                                suppress_low_freq_threshold = self.suppress_low_freq_threshold)
#                 print("Selected heat times:", self.heat_times)
#                 #print("Diagnostics:", diag)
        
#         # Compute graph Laplacian
#         if self.L_graph is None:
#             with torch.no_grad():
#                 self.L_graph = compute_graph_laplacian_from_targets(targets, normalize=True, laplacian_regularization=self.laplacian_regularization)
#                 self.K_graph = compute_heat_kernel_from_laplacian(self.L_graph, self.heat_times)

#         K_manifold = compute_heat_kernel_from_laplacian(L_manifold, self.heat_times)

#         if not ht_recomputed and self.ema_epochs > 2 and self.ema_epochs % 10 == 0 and self.ema_epochs < self.max_ema_epochs:
#             diagnostics, kept_mask = check_heat_kernels_informativeness_fast(
#                 K_manifold,
#                 trace_low_frac=1e-3,    # tune to your problem
#                 trace_high_frac=0.98,
#                 var_eps=1e-8,
#                 diag_offdiag_ratio_min=0.05,
#                 diag_offdiag_ratio_max=20.0,
#                 rowstd_eps=1e-4,
#                 use_power_iter=False,   # keep cheap; set True if you want slightly stronger check
#                 verbose=False
#             )

#             if kept_mask.sum().item() > self.num_heat_time / 2:
#                 # fall back: recompute heat_times (call your preprocessing) or relax thresholds
#                 print("Warning: More than half of heat kernels flagged degenerate. Recomputing heat times.")
#                 print("Previous heat times:", self.heat_times)
#                 with torch.no_grad():
#                     self.heat_times, diag = compute_heat_time_scale_from_laplacian(L=L_manifold, num_times=self.num_heat_time, 
#                                                                                    retain_high_freq_threshold=self.retain_high_freq_threshold, 
#                                                                                    suppress_low_freq_threshold = self.suppress_low_freq_threshold)
#                     print("Selected heat times:", self.heat_times)
#                     self.K_graph = compute_heat_kernel_from_laplacian(self.L_graph, self.heat_times)
#                 K_manifold = compute_heat_kernel_from_laplacian(L_manifold, self.heat_times)
#                 self.freeze_sigma = 0

#         divergence = compute_heat_kernel_divergence(K_manifold, self.K_graph)
        
#         with torch.no_grad():
#             L_manifold_reference = compute_manifold_laplacian(distances=distances,
#                                                               sigma=self.first_sigma,
#                                                               laplacian_regularization=self.laplacian_regularization,
#                                                               debug=False)
#             K_manifold_reference = compute_heat_kernel_from_laplacian(L_manifold_reference, self.heat_times)

#             divergence_reference = compute_heat_kernel_divergence(K_manifold_reference, self.K_graph)
        
#         if self.dist_distorsion_penalty > 0:
#             dist_div = (distances.mean() - self.mean_distance)**2
#             print(f"Current Laplacian loss: {(divergence.item()):.6f} - With referent sigma: {(divergence_reference.item()):.6f} - Distance deviation loss: {(dist_div.item()):.6f}")
#             return {'final_loss': divergence + self.dist_distorsion_penalty * dist_div, 'dyn_loss': divergence, 'ref_loss': divergence_reference, 'sigma': self.sigma_ema, "dist_loss": dist_div}
        
#         print(f"Current Laplacian loss: {(divergence.item()):.6f} - With referent sigma: {(divergence_reference.item()):.6f}")
#         return {'final_loss': divergence, 'dyn_loss': divergence, 'ref_loss': divergence_reference, 'sigma': self.sigma_ema}



# DEPRECATED
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

