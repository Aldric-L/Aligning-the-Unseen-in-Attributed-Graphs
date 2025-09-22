import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod

import gc

from framework.encoder import Encoder
from framework.decoder import DecoderBase
from framework.prior import Prior, GaussianPrior
from framework.KLAnnealingScheduler import KLAnnealingScheduler
from framework.autograd import StopGradient, ReplaceForward
from framework.rbf import PrecisionNetwork
from framework.boundedManifold import BoundedManifold

class VAE(nn.Module):
    """
    Flexible VAE model supporting multiple decoders and pluggable priors
    """
    def __init__(
        self,
        encoder: "Encoder",
        decoders: List["DecoderBase"],
        prior: Optional["Prior"] = None,
        kl_scheduler: Optional["KLAnnealingScheduler"] = None,
        freeze_encoder: bool = False,
        compute_latent_manifold: bool = True
    ):
        """
        Args:
            encoder: Encoder modules
            decoders: List of decoder modules
            prior: Prior distribution (defaults to GaussianPrior)
            kl_scheduler: Optional KL annealing scheduler
            freeze_encoder: Whether to freeze the encoder parameters
        """
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)
        self.decoder_dict = {decoder.name: decoder for decoder in decoders}
        self.prior = prior if prior is not None else GaussianPrior()
        self.prior.giveVAEInstance(self)  
        self.kl_scheduler = kl_scheduler if kl_scheduler else KLAnnealingScheduler()

        self.latent_manifold = None
        self.compute_latent_manifold = compute_latent_manifold
        
        # Set encoder freeze state
        self.set_encoder_freeze(freeze_encoder)
        self.isEncoderFrozen = freeze_encoder
        self.rbf_network = None
        self.rbf_target = None
        
    def set_encoder_freeze(self, freeze: bool = True):
        """
        Freeze or unfreeze encoder parameters
        
        Args:
            freeze: Whether to freeze encoder parameters
        """
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
            if freeze:
                param.grad = None

        self.isEncoderFrozen = freeze
            
    def encode(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode graph into latent space
        
        Args:
            x: Node features [num_nodes, input_dim]
            
        Returns:
            mu: Mean of latent distribution
            params: Distribution parameters (e.g., logvar for Gaussian)
        """
        encoder_kwargs = {k: v for k, v in kwargs.items() 
                            if k in self.encoder.forward.__code__.co_varnames}
        return self.encoder(x, **encoder_kwargs)
    
    def reparameterize(self, mu: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from latent distribution
        
        Args:
            mu: Mean of latent distribution
            params: Distribution parameters
            
        Returns:
            Sampled latent variables
        """
        return self.prior.sample(mu, params)
    
    def decode(self, z: torch.Tensor, **kwargs) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Decode latent variables through all active decoders
        
        Args:
            z: Latent variables [num_nodes, latent_dim]
            **kwargs: Additional decoder-specific arguments
            
        Returns:
            Dict of outputs from each decoder
        """
        all_outputs = {}
        
        for decoder in self.decoders:
            decoder_kwargs = {k: v for k, v in kwargs.items() 
                              if k in decoder.forward.__code__.co_varnames}
            outputs = decoder(z, **decoder_kwargs)
            all_outputs[decoder.name] = outputs
        
        return all_outputs
    
    def forward(self, x: torch.Tensor, encoder_freeze=None, **kwargs) -> Dict[str, Any]:
        """
        Full forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            **kwargs: Additional encoder/decoder arguments
            
        Returns:
            Dict with outputs and latent variables
        """
        # Encode
        encoder_kwargs = {k: v for k, v in kwargs.items() 
                            if k in self.encoder.forward.__code__.co_varnames}
        should_freeze = (encoder_freeze is None and self.isEncoderFrozen) or (encoder_freeze is not None and encoder_freeze)
        if should_freeze:
            with torch.no_grad():
                mu, params = self.encode(x, **encoder_kwargs)
            mu, params = mu.detach(), params.detach()
        else:
            mu, params = self.encode(x, **encoder_kwargs)

        if self.compute_latent_manifold:
            if self.get_latent_manifold() is None:
                manifold_bounds = torch.tensor([[-5, 5]], dtype=torch.float32)
                manifold_bounds = manifold_bounds.repeat(mu.shape[1], 1)
            else:
                manifold_bounds = BoundedManifold.hypercube_bounds(mu, margin=0.2, relative=True)
            self.construct_latent_manifold(manifold_bounds, self.training)
        # Sample latent variables
        z = self.reparameterize(mu, params)
        
        # Decode
        decoder_kwargs = {**kwargs}
        outputs = self.decode(z, **decoder_kwargs)
        
        return {
            "mu": StopGradient.apply(mu) if should_freeze else mu,
            "params": StopGradient.apply(params) if should_freeze else params,
            "z": StopGradient.apply(z.detach()) if should_freeze else z,
            "outputs": outputs
        }
    
    def compute_loss(
        self, 
        outputs: Dict[str, Any], 
        targets: Dict[str, Dict[str, torch.Tensor]],
        decoder_weights: Optional[Dict[str, float]] = None,
        active_decoders: Optional[List[str]] = None,
        use_custom_losses: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss from VAE components
        
        Args:
            outputs: Outputs from forward pass
            targets: Target values for each decoder
            decoder_weights: Optional weights for each decoder loss
            active_decoders: Optional list of decoder names to include in loss computation
            use_custom_losses: Whether to include custom decoder losses
            
        Returns:
            Dict with total loss and component losses
        """
        # Default weights if not provided
        if decoder_weights is None:
            decoder_weights = {decoder.name: 1.0 for decoder in self.decoders}
        
        # Use all decoders if active_decoders not specified
        if active_decoders is None:
            active_decoders = list(self.decoder_dict.keys())
        
        # KL divergence loss using pluggable prior
        mu, params = outputs["mu"], outputs["params"]
        kl_loss = self.prior.kl_divergence(mu, params)
        
        # Get current KL weight from scheduler
        kl_weight = self.kl_scheduler.get_weight()
        
        # Compute individual decoder losses
        decoder_losses = {}
        detailed_losses = {}
        weighted_recon_loss = 0
        
        z = outputs["z"]  # Get latent variables for custom losses
        
        for decoder in self.decoders:
            name = decoder.name
            if name not in outputs["outputs"] or name not in active_decoders:
                continue
                
            # Get decoder outputs and corresponding targets
            decoder_outputs = outputs["outputs"][name]
            decoder_targets = targets.get(name, {})
            
            # Compute loss for this decoder (including custom losses if enabled)
            if use_custom_losses and hasattr(decoder, 'compute_total_loss'):
                loss_breakdown = decoder.compute_total_loss(
                    decoder_outputs, decoder_targets, z=z
                )
                decoder_loss = loss_breakdown['total']
                detailed_losses[name] = loss_breakdown
            else:
                decoder_loss = decoder.compute_loss(decoder_outputs, decoder_targets)
                if decoder_loss.ndim > 0:
                    decoder_loss = decoder_loss.sum() 
                detailed_losses[name] = {'base_loss': decoder_loss, 'total': decoder_loss}
            
            if decoder_loss.ndim > 0:
                decoder_loss = decoder_loss.sum() 
            decoder_losses[name] = decoder_loss
            
            # Add weighted loss to total reconstruction loss
            decoder_weight = decoder_weights.get(name, 1.0)
            if decoder_weight == -1:
                weighted_recon_loss += decoder_loss / decoder_loss.detach().item()
            else:
                weighted_recon_loss += decoder_weight * decoder_loss
        
        # Compute total loss
        total_loss = weighted_recon_loss + kl_weight * kl_loss
        
        return {
            "total_loss": total_loss,
            "kl_loss": kl_loss,
            "kl_weight": kl_weight,
            "recon_loss": weighted_recon_loss,
            "decoder_losses": decoder_losses,
            "detailed_losses": detailed_losses
        }
    
    def add_decoder(self, decoder: "DecoderBase"):
        """
        Add a new decoder to the model
        
        Args:
            decoder: Decoder module to add
        """
        self.decoders.append(decoder)
        self.decoder_dict[decoder.name] = decoder
    
    def remove_decoder(self, name: str) -> bool:
        """
        Remove a decoder by name
        
        Args:
            name: Name of the decoder to remove
            
        Returns:
            True if decoder was removed, False if not found
        """
        if name in self.decoder_dict:
            decoder = self.decoder_dict[name]
            self.decoders = nn.ModuleList([d for d in self.decoders if d.name != name])
            self.decoder_dict.pop(name)
            return True
        return False
    
    def get_decoder(self, name: str) -> Optional["DecoderBase"]:
        """
        Get decoder by name
        
        Args:
            name: Name of the decoder
            
        Returns:
            Decoder module or None if not found
        """
        return self.decoder_dict.get(name)
    
    def get_active_decoders(self) -> List[str]:
        """
        Get names of all active decoders
        
        Returns:
            List of decoder names
        """
        return list(self.decoder_dict.keys())
    
    def compute_jacobian(self, z: torch.Tensor, decoder_name: str, node_idx: Optional[int] = None, **kwargs) -> torch.Tensor:
        """
        Compute Jacobian of a specific decoder with respect to latent space
        
        Args:
            z: Latent variables [num_nodes, latent_dim]
            decoder_name: Name of the decoder to compute Jacobian for
            node_idx: Optional node index to compute Jacobian for
            **kwargs: Additional encoder/decoder arguments
            
        Returns:
            Jacobian matrix
        """
        decoder = self.get_decoder(decoder_name)
        if decoder is None:
            raise ValueError(f"Decoder '{decoder_name}' not found")
            
        if hasattr(decoder, "compute_jacobian"):
            jacobian_kwargs = {k: v for k, v in kwargs.items() 
                                if k in decoder.compute_jacobian.__code__.co_varnames}
            return decoder.compute_jacobian(z, node_idx,  **jacobian_kwargs)
        else:
            raise NotImplementedError(f"Decoder '{decoder_name}' does not implement compute_jacobian method")
    
    # Convenience methods for managing custom losses
    def add_custom_loss_to_decoder(self, decoder_name: str, loss_name: str, 
                                  loss_fn: Callable, weight: float = 1.0):
        """
        Add a custom loss to a specific decoder
        
        Args:
            decoder_name: Name of the decoder
            loss_name: Name for the custom loss
            loss_fn: Loss function
            weight: Weight for the loss
        """
        decoder = self.get_decoder(decoder_name)
        if decoder is None:
            raise ValueError(f"Decoder '{decoder_name}' not found")
        decoder.add_custom_loss(loss_name, loss_fn, weight)
    
    def remove_custom_loss_from_decoder(self, decoder_name: str, loss_name: str):
        """
        Remove a custom loss from a specific decoder
        
        Args:
            decoder_name: Name of the decoder
            loss_name: Name of the custom loss to remove
        """
        decoder = self.get_decoder(decoder_name)
        if decoder is not None:
            decoder.remove_custom_loss(loss_name)
    
    def set_custom_loss_active(self, decoder_name: str, loss_name: str, active: bool = True):
        """
        Activate/deactivate a custom loss for a specific decoder
        
        Args:
            decoder_name: Name of the decoder
            loss_name: Name of the custom loss
            active: Whether the loss should be active
        """
        decoder = self.get_decoder(decoder_name)
        if decoder is not None:
            decoder.set_custom_loss_active(loss_name, active)
    
    def set_custom_loss_weight(self, decoder_name: str, loss_name: str, weight: float):
        """
        Update the weight of a custom loss for a specific decoder
        
        Args:
            decoder_name: Name of the decoder
            loss_name: Name of the custom loss
            weight: New weight value
        """
        decoder = self.get_decoder(decoder_name)
        if decoder is not None:
            decoder.set_custom_loss_weight(loss_name, weight)
    
    def _local_decoder_point_metric(self, z, decoder_name: Optional[str] = None, rbf: bool = False) -> torch.Tensor:
        """
        Vectorized wrapper to compute the Riemannian metric from the jacobian matrix of decoder_name
        
        Args:
            z: torch tensor with shape (batch_size, n) or (n,) for single point
            decoder_name: Optional decoder name
            rbf: use RBF network? be carefull, this require the RBF network to have been trained and the decoder 
                 to have in its jacobian function a kwarg 'mode="separate"'
            
        Returns:
            torch.Tensor: Metric tensor(s)
                - If z.shape = (n,): returns (n, n) metric tensor
                - If z.shape = (batch_size, n): returns (batch_size, n, n) metric tensors
        """
        # Handle single point case by adding batch dimension
        single_point = z.dim() == 1
        if single_point:
            z = z.unsqueeze(0)  # (n,) -> (1, n)
        
        batch_size, n = z.shape
        
        try:
            # Compute Jacobian for all points in batch
            # J should have shape (batch_size, output_dim, n)
            if rbf is True and self.rbf_target == decoder_name and self.rbf_network is not None:
                J_mu, J_sigma = self.compute_jacobian(z, decoder_name, mode="separate")
                J_rbf = self.rbf_network.compute_precision_jacobian_vectorized(z).detach()
                #J_sigma_rbf = torch.bmm(J_rbf, J_sigma)
                # Usage
                J_sigma_rbf = ReplaceForward.apply(J_sigma, J_rbf) 
                J = J_mu + J_sigma_rbf
            else:
                J = self.compute_jacobian(z, decoder_name)
            
            # Compute metric tensor G = J^T @ J for each point in batch
            # Using batch matrix multiplication
            G = torch.bmm(J.transpose(-2, -1), J)  # (batch_size, n, n)
            
            # Remove batch dimension if input was single point
            if single_point:
                G = G.squeeze(0)  # (1, n, n) -> (n, n)
                
        except Exception as e:
            print(f"Error computing metric for batch: {e}")
            raise RuntimeError("Riemannian metric computation failed.")
        
        return G

    def construct_latent_manifold(self, bounds: torch.Tensor, force : bool = False, cache : bool = True, 
                                  decoder_name: Optional[str] = None, rbf: bool = True) -> BoundedManifold:
        """
        Construct the latent manifold representation

        Args:
            bounds: bounds of the bounded manifold
            force: if True, we construct a new latent manifold in place of the previous one
            decoder_name: Optional name of the decoder to construct the manifold from
            rbf: bool if True, we will request the VAE to use the RBF network for the variance component.

        Returns:
            BoundedManifold object 
        """
        if self.latent_manifold is None:
            if decoder_name is None:
                decoder_name = list(self.decoder_dict.keys())[0]
            def local_point_metric(z: torch.Tensor) -> torch.Tensor:
                """Local point metric function for the manifold"""
                return self._local_decoder_point_metric(z, decoder_name, rbf)
            self.latent_manifold = BoundedManifold(local_point_metric, bounds=bounds, 
                                                   grid_size=100, device=next(self.parameters()).device)
        elif self.latent_manifold is not None and force is True:
            self.latent_manifold.clear()
            
        return self.latent_manifold

    def get_latent_manifold(self):
        """
        Get the latent manifold representation if available

        Returns:
            BoundedManifold object or None if not initialized
        """
        return self.latent_manifold
    
    def set_compute_latent_manifold(self, compute: bool = True):
        """
        Set whether to compute the latent manifold during forward pass

        Args:
            compute: If True, compute the latent manifold
        """
        self.compute_latent_manifold = compute

    def train_rbf_layer(self, decoder_name: str, decoder_variance_target_name:str, latent_samples: torch.FloatTensor, 
                      n_centers: int =5, a: float=1.0, n_epochs: int=100, lr: float =0.01,
                      verbose: float = True, force: bool = True):
        """
        Train the RBF weights to better fit the latent samples
        
        This is a simple optimization that maximizes precision in areas with data,
        and minimizes it in areas without data.
        """
        if self.rbf_network is None or force is True or self.rbf_target != decoder_name:
            if force is True or self.rbf_target != decoder_name:
                self.rbf_network = None
            self.rbf_target = decoder_name
            self.rbf_network = PrecisionNetwork(latent_dim=self.encoder.latent_dim, data_dim=self.encoder.input_dim, n_centers=n_centers, a=a)
            self.rbf_network.fit(latent_samples)
        optimizer = torch.optim.Adam([self.rbf_network.rbf_layer.weights], lr=lr)

        log_variances = self.decode(latent_samples)[decoder_name][decoder_variance_target_name]

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Compute precision for all points
            precision_estimates = self.rbf_network(latent_samples)
            loss = torch.nn.functional.mse_loss(-torch.log(precision_estimates + 1e-8), log_variances.detach())
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0 and verbose:
                print(f"[RBF Training] Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

    def get_rbf_estimate(self, z: torch.Tensor) -> torch.Tensor:
        """
        Get RBF estimate of the variance through the selected decoder
        
        Args:
            z: Latent variables [num_nodes, latent_dim]
        """
        if self.rbf_network is None:
            raise RuntimeError("RBF network is not set.")
        
        if z.dim() == 1:
            z = z.unsqueeze(0)

        return self.rbf_network(z)
