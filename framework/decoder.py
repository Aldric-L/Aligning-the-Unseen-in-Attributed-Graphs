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

from framework.utils import heat_kernel_distance, get_adjacency_matrix_from_tensors
from framework.preprocessing import PreprocessingLayer, DistancePreprocessingLayer, JacobianMetricPreprocessingLayer
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
        name: str = "node_attr_decoder"
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
        self.logvar_head = nn.Linear(in_features, output_dim)
    
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
        lambda_comp_variance: float = 5,
        lambda_decoder_variance: float = 100.0,
        eps_var: float = 1e-8
    ) -> torch.Tensor:
        """
        Loss = 
        1) - E_q[log p(x|z)]                            # Gaussian NLL
        + 2) lambda_comp_variance * mean_i[(Var_d mu_i - Var_d x_i)^2]
        + 3) lambda_decoder_variance * mean_i[(Mean_d sigma^2_i - (Var_d x_i - Var_d mu_i))^2]
        """
        mu     = outputs["node_features_mu"]      # [B, D]
        logvar = outputs["node_features_logvar"]  # [B, D]
        x      = targets["node_features"]         # [B, D]
        B, D   = x.shape

        # 1) Gaussian NLL
        std   = torch.exp(0.5 * logvar) + eps_var
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

        #print(f"NLL={recon_nll.item():.2f}, comp_var_pen={lambda_comp_variance*comp_var_pen.item():.2f} ({comp_var_pen.item():.2f}), var_pen={lambda_decoder_variance*dec_var_pen.item():.2f} ({dec_var_pen.item():.2f}), final pen={lambda_comp_variance*comp_var_pen.item() + lambda_decoder_variance*dec_var_pen.item():.2f}")

        return (
            recon_nll
            + lambda_comp_variance    * comp_var_pen
            + lambda_decoder_variance * dec_var_pen
        )

        
    # def compute_jacobian(
    #     self, 
    #     z: torch.Tensor, 
    #     node_idx: Optional[int] = None,
    # ) -> torch.Tensor:
    #     """
    #     Compute Jacobian of the sampled output w.r.t. z:
    #       recon = mu(z) + sigma(z) * eps
    #     so J = J_mu + J_sigma.
    #     """
    #     # detach input
    #     z0 = z.detach()
    #     with torch.no_grad():
    #         outputs = self.forward(z0, sample=True)
    #         mu     = outputs["node_features_mu"]
    #         logvar = outputs["node_features_logvar"]
    #         recon  = outputs["node_features"]
        
    #     def f_mu(z_in):
    #         return self.forward(z_in, sample=False)["node_features_mu"]
        
    #     def f_sigma(z_in):
    #         lv = self.forward(z_in, sample=False)["node_features_logvar"]
    #         return torch.exp(0.5 * lv)
        
    #     if node_idx is not None or z0.ndim == 1:
    #         # single node
    #         idx = 0 if z0.ndim == 1 else node_idx
    #         z_node = z0[idx].requires_grad_(True)

    #         J_mu    = jacobian(lambda zz: f_mu(z0.clone().scatter(0, idx, zz))[idx], z_node)
    #         J_sigma = jacobian(lambda zz: f_sigma(z0.clone().scatter(0, idx, zz))[idx], z_node)

    #         # J = J_mu + J_sigma
    #         return J_mu + J_sigma

    #     else:
    #         # full batch
    #         z0 = z0.requires_grad_(True)
    #         J_mu_full    = jacobian(f_mu,    z0, vectorize=True)
    #         J_sigma_full = jacobian(f_sigma, z0, vectorize=True)
    #         # both have shape [N, out_dim, N, latent_dim]

    #         N, D, _, L = J_mu_full.shape
    #         idx = torch.arange(N, device=J_mu_full.device)      # [N]
    #         # J_mu_full[idx, :, idx, :] has shape [N, D, L]
    #         return J_mu_full[idx, :, idx, :] + J_sigma_full[idx, :, idx, :]

    # def compute_jacobian(self, z: torch.Tensor,  node_idx: Optional[int] = None, chunk_size: int = 32):
    #     N = z.size(0)
    #     results = []
        
    #     for start in range(0, N, chunk_size):
    #         end = min(start + chunk_size, N)
    #         chunk_result = self._compute_jacobian(z[start:end], node_idx=node_idx)
    #         results.append(chunk_result)
        
    #     return torch.cat(results, dim=0)

    # def _compute_jacobian(self, z: torch.Tensor, node_idx: Optional[int] = None) -> torch.Tensor:
    #     """
    #     Optimized: compute jacobian of recon = mu + sigma * eps
    #     by computing J_h once for the shared mlp features h = self.mlp(z).
    #     Assumes mean_head and logvar_head are nn.Linear layers.
    #     """
    #     # detach input for the reference recon etc.
    #     z0 = z.detach()
    #     with torch.no_grad():
    #         outputs = self.forward(z0, sample=True)
    #         mu     = outputs["node_features_mu"]
    #         logvar = outputs["node_features_logvar"]
    #         recon  = outputs["node_features"]

    #     W_mu = self.mean_head.weight      # [D, H]
    #     b_mu = self.mean_head.bias
    #     W_lv = self.logvar_head.weight    # [D, H]
    #     b_lv = self.logvar_head.bias

    #     def f_h(z_in):
    #         # return pre-head features h for each node
    #         # ensure shape: (N, H) for batch input, (1,H) if single vector
    #         return self.mlp(z_in)

    #     if node_idx is not None or z0.ndim == 1:
    #         # single node case
    #         idx = 0 if z0.ndim == 1 else node_idx
    #         z_node = z0[idx].requires_grad_(True)  # shape [L]

    #         # jacobian of h wrt this node's latent z: shape [H, L]
    #         J_h = jacobian(lambda zz: f_h(zz.unsqueeze(0))[0], z_node, create_graph=False)  # [H, L]

    #         # compute mu and sigma at this node
    #         h_node = f_h(z0[idx:idx+1])[0]                # [H]
    #         lv_node = (W_lv @ h_node) + b_lv              # [D]
    #         sigma_node = torch.exp(0.5 * lv_node)         # [D]

    #         # J_mu = W_mu @ J_h  -> [D, L]
    #         J_mu = W_mu @ J_h

    #         # factor = 0.5 * sigma[:,None] * W_lv  -> [D, H]
    #         factor = 0.5 * sigma_node[:, None] * W_lv      # broadcast

    #         # J_sigma = factor @ J_h -> [D, L]
    #         J_sigma = factor @ J_h

    #         return J_mu + J_sigma

    #     else:
    #         # full-batch: compute J_h once vectorized
    #         z0 = z0.requires_grad_(True)   # shape [N, L]

    #         # J_h_full: [N, H, N, L] (vectorize=True often reduces memory/time)
    #         J_h_full = jacobian(f_h, z0, vectorize=True, create_graph=False)

    #         N, H, _, L = J_h_full.shape
    #         idx = torch.arange(N, device=J_h_full.device)

    #         # extract diagonal blocks: [N, H, L]
    #         J_h = J_h_full[idx, :, idx, :]

    #         # compute h, lv, sigma for each node (N,H), (N,D), (N,D)
    #         h = f_h(z0)                       # [N, H]
    #         lv = torch.matmul(h, W_lv.t()) + b_lv  # [N, D]
    #         sigma = torch.exp(0.5 * lv)             # [N, D]

    #         # J_mu: einsum 'dh,nhl->ndl'  -> [N, D, L]
    #         J_mu = torch.einsum('dh,nhl->ndl', W_mu, J_h)

    #         # factor per node: [N, D, H] = 0.5 * sigma[:, :, None] * W_lv[None, :, :]
    #         factor = 0.5 * sigma.unsqueeze(2) * W_lv.unsqueeze(0)  # [N, D, H]

    #         # J_sigma: einsum 'ndh,nhl->ndl' -> [N, D, L]
    #         J_sigma = torch.einsum('ndh,nhl->ndl', factor, J_h)

    #         return J_mu + J_sigma
        
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

class LatentDistanceDecoder(DecoderBase):
    """
    Fictive decoder that outputs latent codes but computes a meaningful loss 
    based on pairwise distances between connected nodes using Jacobian-based metrics.
    """
    def __init__(
        self, 
        latent_dim: int,
        distance_mode: str = "linear_interpolation",  # or "direct"
        num_integration_points: int = 25,
        metric_regularization: float = 1e-6,
        name: str = "latent_distance_decoder"
    ):
        super(LatentDistanceDecoder, self).__init__(latent_dim, name)
        
        self.distance_mode = distance_mode
        self.num_integration_points = num_integration_points
        self.metric_regularization = metric_regularization
        self.model = None  # Weak reference to the model instance

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
    
    def compute_loss(
                self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                negative_distance_weight: float = 1.0,
            ) -> torch.Tensor:
        if self.model is None or self.model() is None:
            raise ValueError("Model instance not set. Call giveVAEInstance(model) first.")

        z = outputs["latent_codes"]   # (N, D)
        N = z.size(0)
        # Removed: batched_dist = torch.vmap(self._riemannian_distance, in_dims=(0,0), out_dims=0)

        pos_pairs = []
        pos_weights = []

        if "edge_index" in targets:
            src, dst = targets["edge_index"]
            edge_labels = targets.get("edge_labels", None)
            for idx, (i, j) in enumerate(zip(src.tolist(), dst.tolist())):
                if i == j:
                    continue
                w = edge_labels[idx].item() if edge_labels is not None else 1.0
                if w > 0:
                    pos_pairs.append((i, j))
                    pos_weights.append(w)

        elif "adj_matrix" in targets:
            A = targets["adj_matrix"]
            ui, uj = torch.triu_indices(N, N, offset=1)
            weights = A[ui, uj]
            mask = weights > 0
            for i, j, w in zip(ui[mask].tolist(), uj[mask].tolist(), weights[mask].tolist()):
                pos_pairs.append((i, j))
                pos_weights.append(w)

        else:
            raise ValueError("Targets must contain either 'edge_index' or 'adj_matrix'")

        # Batch positive distances
        if pos_pairs:
            pos_pairs = torch.tensor(pos_pairs, device=z.device)
            pos_weights = torch.tensor(pos_weights, device=z.device)
            z_i = z[pos_pairs[:,0]]
            z_j = z[pos_pairs[:,1]]
            # Directly call _riemannian_distance (it handles its own batching)
            pos_dists = self._riemannian_distance(z_i, z_j) 
            total_pos   = (pos_weights * pos_dists).sum()
            sum_weights = pos_weights.sum()
            pos_loss    = total_pos / sum_weights
        else:
            pos_loss    = torch.tensor(0.0, device=z.device, requires_grad=True)
            sum_weights = 1.0

        # Build non-edge mask & sample negatives
        mask_nonedge = torch.ones((N, N), dtype=torch.bool, device=z.device)
        if pos_pairs.numel() > 0:
            mask_nonedge[pos_pairs[:,0], pos_pairs[:,1]] = False
            mask_nonedge[pos_pairs[:,1], pos_pairs[:,0]] = False
        mask_nonedge.fill_diagonal_(False)

        if negative_distance_weight > 0:
            num_neg = len(pos_pairs) if "edge_index" in targets else int((targets["adj_matrix"] != 0).sum().item() / 2)
            all_pairs = torch.nonzero(mask_nonedge, as_tuple=False)
            idxs      = torch.randint(0, all_pairs.size(0), (num_neg,), device=z.device)
            neg_pairs = all_pairs[idxs]

            z_i_neg = z[neg_pairs[:,0]]
            z_j_neg = z[neg_pairs[:,1]]
            # Directly call _riemannian_distance
            neg_dists = self._riemannian_distance(z_i_neg, z_j_neg)

            neg_penalties = 1.0 / (1.0 + neg_dists**2)

            neg_loss   = negative_distance_weight * neg_penalties.mean()
        else:
            neg_loss = torch.tensor(0.0, device=z.device, requires_grad=True)
        total_loss = pos_loss + neg_loss

        print(f"Loss: pos={pos_loss.item():.4f}, neg={neg_loss.item():.4f}, total={total_loss.item():.4f}")
        return total_loss


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
        metric_regularization: float = 1e-6,
        name: str = "manifold_heat_kernel_decoder",
        # Heat kernel specific parameters
        heat_time: Union[float, List[float]] = 1.0,  # Time parameter for heat kernel
        num_eigenvalues: int = 50,  # Number of eigenvalues to compute
        laplacian_regularization: float = 1e-8,
        heat_kernel_approximation: str = "spectral",  # "spectral" or "finite_difference"
        finite_diff_steps: int = 100,
        # Manifold Laplacian construction
        manifold_neighbors: int = 10,  # How many nearest neighbors to connect in manifold
        gaussian_bandwidth: float = 1.0,  # Bandwidth for Gaussian weights
        lag_factor: float = 0.5,
    ):
        super(ManifoldHeatKernelDecoder, self).__init__(latent_dim, name)
        
        self.distance_mode = distance_mode
        self.num_integration_points = num_integration_points
        self.metric_regularization = metric_regularization
        self.model = None  # Weak reference to the model instance

        self.heat_times = [heat_time] if isinstance(heat_time, (float, int)) else list(heat_time)
        self.num_eigenvalues = num_eigenvalues
        self.laplacian_regularization = laplacian_regularization
        self.heat_kernel_approximation = heat_kernel_approximation
        self.finite_diff_steps = finite_diff_steps
        self.manifold_neighbors = manifold_neighbors
        self.gaussian_bandwidth = gaussian_bandwidth
        self.sigma_ema = None
        self.lag_factor = lag_factor

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

    def compute_manifold_laplacian(self, z: torch.Tensor, batch_size=8, normalize:bool=True, knn_for_sigma: int =7) -> torch.Tensor:
        """
        Compute the Laplace-Beltrami operator on the manifold using proper Riemannian distances
        Fully differentiable using Gaussian RBF weights
        """
        num_nodes = z.size(0)
        device = z.device
        
        # Compute all pairwise Riemannian distances
        print("Computing pairwise Riemannian distances...")
        # distances = torch.zeros(num_nodes, num_nodes, device=device, dtype=z.dtype)
        
        # for i in tqdm(range(num_nodes), desc="Computing distances"):
        #     for j in range(i + 1, num_nodes):
        #         dist = self._riemannian_distance(z[i], z[j])
        #         distances[i, j] = dist
        #         distances[j, i] = dist  # Symmetric
        if self.distance_mode == "dijkstra" or self.distance_mode == "Dijkstra":
            distances = self.model().get_latent_manifold().get_grid_as_graph().compute_shortest_paths(
                self.model().get_latent_manifold()._clamp_point_to_bounds(z),
                weight_type="geodesic",  # Uses your metric tensors
                max_grid_neighbors=8,     # Connect to up to 8 nearest grid nodes
                num_threads=6
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

        with torch.no_grad():
            if knn_for_sigma >= num_nodes**0.5:
                knn_for_sigma = max(1, num_nodes**0.5 - 1)
            # sort row-wise (excluding diagonal)
            #sorted_dists, _ = torch.sort(distances + torch.eye(num_nodes, device=device) * 1e9, dim=1)
            # k-th nearest: index knn_for_sigma (0-based if we excluded diag by big number)
            #sigma_i = sorted_dists[:, knn_for_sigma].clamp(min=1e-8)  # shape (N,)
            #sigma_i = sigma_i.clamp(min=1e-4).view(num_nodes, 1)
            # build symmetric, locally-scaled Gaussian kernel
            #sigma_matrix = sigma_i * sigma_i.t()  # σ_i * σ_j
            #print("Min/Max sigma_matrix:", torch.min(sigma_matrix), torch.max(sigma_matrix))
            alpha = self.lag_factor  # small update
            # if self.sigma_ema is None:
            #      self.sigma_ema = sigma_matrix
            # else:
            #     sigma_matrix = (1 - alpha) * self.sigma_ema + alpha * sigma_matrix
            #     self.sigma_ema = sigma_matrix
            # print("Min/Max sigma_matrix (corrected):", torch.min(sigma_matrix), torch.max(sigma_matrix))
            sigma = torch.median(distances)
            print("Selected sigma=", sigma)
            if self.sigma_ema is None:
                 self.sigma_ema = sigma
            else:
                sigma = (1 - alpha) * self.sigma_ema + alpha * sigma
                self.sigma_ema = sigma
            print("Corrected sigma=", sigma)
        #weights = torch.exp(-distances**2 / (2 * self.gaussian_bandwidth**2))
        #weights = torch.exp(- (distances ** 2) / (sigma_matrix + 1e-8))
        weights = torch.exp(-distances**2 / (2 * sigma**2))

        # zero out diagonal (no self-weight)
        #weights.fill_diagonal_(0.0)
        
        # Zero out diagonal (no self-connections)
        weights = weights * (1 - torch.eye(num_nodes, device=device))
        
        # Optional: Threshold very small weights to maintain some sparsity
        threshold = 1e-4
        weights = torch.where(weights > threshold, weights, torch.zeros_like(weights))
        
        # Create Laplacian: L = D - W where D is degree matrix
        degree = torch.sum(weights, dim=1)
        if normalize:
            d_inv_sqrt = torch.where(degree > 0, torch.pow(degree + 1e-8, -0.5), torch.zeros_like(degree))
            D_inv_sqrt = torch.diag(d_inv_sqrt)
            L_manifold = torch.eye(weights.size(0)) - D_inv_sqrt @ weights @ D_inv_sqrt
            # print("Normal laplacian:", torch.diag(degree) - weights)
            # print("Normalized laplacian:", L_manifold)
            # print("D_inv_sqrt:", D_inv_sqrt)
            # print("Any NaN in L_sym?", torch.isnan(L_manifold).any().item())
            # print("Any Inf in L_sym?", torch.isinf(L_manifold).any().item())
        else:
            L_manifold = torch.diag(degree) - weights

        # Add regularization
        L_manifold += self.laplacian_regularization * torch.eye(num_nodes, device=device)
        
        return L_manifold
    
    def compute_heat_kernel_spectral(self, laplacian: torch.Tensor, t: Union[float, List[float]], eigenvals=None, eigenvecs=None) -> torch.Tensor:
        """
        Compute heat kernel using spectral decomposition: K(t) = exp(-t*L)
        """
        try:
            # if eigenvals is None or eigenvecs is None:
            #     # Eigendecomposition of Laplacian
            #     eigenvals, eigenvecs = torch.linalg.eigh(laplacian)
                
            #     # Clamp eigenvalues to avoid numerical issues
            #     eigenvals = torch.clamp(eigenvals, min=0.0)
            
            # # Take only the first num_eigenvalues for efficiency
            # if self.num_eigenvalues < eigenvals.size(0):
            #     eigenvals = eigenvals[:self.num_eigenvalues]
            #     eigenvecs = eigenvecs[:, :self.num_eigenvalues]
            
            # Compute heat kernel: K(t) = V * exp(-t*Λ) * V^T
            if isinstance(t, (float, int)):
                #exp_eigenvals = torch.exp(-t * eigenvals)
                return torch.matrix_exp(-t * laplacian)
                return eigenvecs @ torch.diag(exp_eigenvals) @ eigenvecs.t()
            else:
                heat_kernels = []
                for t_i in t:
                    #exp_eigenvals = torch.exp(-t_i * eigenvals)
                    #K_t = eigenvecs @ torch.diag(exp_eigenvals) @ eigenvecs.t()
                    K_t = torch.matrix_exp(-t_i * laplacian)
                    heat_kernels.append(K_t)
                return heat_kernels
                        
        except Exception as e:
            print(f"Warning: Spectral heat kernel computation failed: {e}")
            # Fallback to identity
            return torch.eye(laplacian.size(0), device=laplacian.device, dtype=laplacian.dtype)
    
    def compute_graph_laplacian(self, targets: Dict[str, torch.Tensor], normalize: bool = True) -> torch.Tensor:
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
        if normalize:
            d_inv_sqrt = torch.where(degree > 0, torch.pow(degree + 1e-8, -0.5), torch.zeros_like(degree))
            D_inv_sqrt = torch.diag(d_inv_sqrt)
            laplacian = torch.eye(adj.size(0)) - D_inv_sqrt @ adj @ D_inv_sqrt
            # print("Any NaN in L_sym?", torch.isnan(laplacian).any().item())
            # print("Any Inf in L_sym?", torch.isinf(laplacian).any().item())
        else:
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
        heat_kernel_weight: float = 1,
    ) -> torch.Tensor:
        """
        Compute loss based on heat kernel divergence between manifold and graph
        """
        z = outputs["latent_codes"]
        num_nodes = z.size(0)
        
        #print("Computing graph Laplacian...")
        start_time = time.time()
        # Compute graph Laplacian
        L_graph = self.compute_graph_laplacian(targets)
        #print(f"[TIMER] Done in {time.time() - start_time:.4f}s")
        
        #print("Computing manifold Laplacian...")
        # Compute manifold Laplacian (this will compute Riemannian distances)
        start_time = time.time()
        L_manifold = self.compute_manifold_laplacian(z)
        #print(f"[TIMER] Done in {time.time() - start_time:.4f}s")

        #eigenvals, eigenvecs = torch.linalg.eigh(L_manifold)   
        #print("eig min:", eigenvals.min().item(), "eig max:", eigenvals.max().item()) 
        #eigenvals = torch.clamp(eigenvals, min=0.0)
        num_times = 20

        # # Keep only first m eigenpairs if specified
        # if self.num_eigenvalues is not None:
        #     eigenvals = eigenvals[:self.num_eigenvalues]
        #     eigenvecs = eigenvecs[:, :self.num_eigenvalues]

        # # Keep strictly positive eigenvalues (avoid division by 0)
        # pos_mask = eigenvals > 1e-5
        # if pos_mask.sum() == 0:
        #     raise RuntimeError("No positive eigenvalues found (graph might be fully disconnected).")

        # eigenvals_pos = eigenvals[pos_mask]
        # lambda_min = eigenvals_pos.min()
        # lambda_max = eigenvals_pos.max()

        # # Normalize eigenvalues by max
        # lambdas_tilde = eigenvals_pos / lambda_max
        # lambda_tilde_min = lambdas_tilde.min().item()

        # # Log-spaced times in normalized scale
        # t_tilde = torch.logspace(0, torch.log10(torch.tensor(1.0/lambda_tilde_min)), steps=num_times)

        # # Rescale by lambda_max
        # heat_times = t_tilde / lambda_max

        # heat_times = torch.logspace(-0.5, 1.5, steps=num_times)
        heat_times = torch.logspace(0, 1.2, steps=num_times)

        # lambda_min = eigenvals[eigenvals > 1e-6].min().item()
        # lambda_max = eigenvals.max().item()

        # # Set attenuation thresholds
        # alpha_min = 0.8  # retain 80% of high-frequencies at t_min
        # alpha_max = 0.1 # retain 10% of low-frequencies at t_max

        # # Compute t_min and t_max
        # t_min = -np.log(alpha_min) / lambda_max
        # t_max = -np.log(alpha_max) / lambda_min

        # Generate logspaced times between t_min and t_max
        # heat_times = torch.logspace(start=np.log10(t_min), end=np.log10(t_max), steps=num_times).tolist()
        # print(heat_times)
        
        #print("Computing heat kernels...")
        start_time = time.time()
        # Compute heat kernels
        K_graph = self.compute_heat_kernel_spectral(L_graph, heat_times)
        K_manifold = self.compute_heat_kernel_spectral(L_manifold, heat_times)
        #print(f"[TIMER] Done in {time.time() - start_time:.4f}s")
        
        #print("Computing heat kernel divergence...")
        # Compute divergence
        start_time = time.time()
        divergence = self.compute_heat_kernel_divergence(K_manifold, K_graph)
        #print(f"[TIMER] Done in {time.time() - start_time:.4f}s")
        
        # Total loss
        total_loss = heat_kernel_weight * divergence
        
        #print(f"Heat kernel divergence: {(divergence.item()):.6f}")
        print(f"Total Laplacian loss: {(total_loss.item()):.6f}")
        
        return total_loss


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

