import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import torch.optim as optim
from tqdm import tqdm

from framework.encoder import Encoder
from framework.decoder import DecoderBase


class KLAnnealingScheduler:
    """
    Scheduler for KL annealing during training
    """
    def __init__(
        self, 
        kl_weight: float = 1.0,
        anneal_type: str = 'linear',
        anneal_start: float = 0.0,
        anneal_end: float = 1.0,
        anneal_steps: int = 1000
    ):
        """
        Args:
            kl_weight: Maximum KL weight
            anneal_type: Type of annealing ('linear', 'sigmoid', 'cyclical')
            anneal_start: Starting value for annealing
            anneal_end: Final value for annealing
            anneal_steps: Number of steps to reach final value
        """
        self.kl_weight = kl_weight
        self.anneal_type = anneal_type
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end
        self.anneal_steps = anneal_steps
        self.current_step = 0
    
    def step(self):
        """
        Increment step counter
        """
        self.current_step += 1
    
    def get_weight(self) -> float:
        """
        Get current KL weight based on annealing schedule
        
        Returns:
            Current KL weight
        """
        progress = min(1.0, self.current_step / self.anneal_steps)
        
        if self.anneal_type == 'linear':
            weight = self.anneal_start + progress * (self.anneal_end - self.anneal_start)
        
        elif self.anneal_type == 'sigmoid':
            # Sigmoid annealing
            steepness = 5.0  # Controls steepness of sigmoid
            midpoint = 0.5   # Where the sigmoid is centered
            
            # Apply sigmoid function
            sigmoid_progress = 1 / (1 + np.exp(-steepness * (progress - midpoint)))
            weight = self.anneal_start + sigmoid_progress * (self.anneal_end - self.anneal_start)
        
        elif self.anneal_type == 'cyclical':
            # Cyclical annealing (useful for avoiding posterior collapse)
            cycle_length = self.anneal_steps / 4  # 4 cycles in total annealing period
            within_cycle_progress = (self.current_step % cycle_length) / cycle_length
            
            # Ramp up within each cycle, then stay at maximum
            if within_cycle_progress < 0.5:
                cycle_weight = within_cycle_progress * 2
            else:
                cycle_weight = 1.0
                
            weight = self.anneal_start + cycle_weight * (self.anneal_end - self.anneal_start)
        
        else:
            weight = self.kl_weight  # No annealing
        
        return weight * self.kl_weight


class GraphVAE(nn.Module):
    """
    Flexible Graph VAE model supporting multiple decoders and phased training
    """
    def __init__(
        self,
        encoder: "Encoder",
        decoders: List["DecoderBase"],
        kl_scheduler: Optional["KLAnnealingScheduler"] = None,
        freeze_encoder: bool = False
    ):
        """
        Args:
            encoder: Graph encoder module
            decoders: List of decoder modules
            kl_scheduler: Optional KL annealing scheduler
            freeze_encoder: Whether to freeze the encoder parameters
        """
        super(GraphVAE, self).__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)
        self.decoder_dict = {decoder.name: decoder for decoder in decoders}
        self.kl_scheduler = kl_scheduler if kl_scheduler else KLAnnealingScheduler()
        
        # Set encoder freeze state
        self.set_encoder_freeze(freeze_encoder)
        
    def set_encoder_freeze(self, freeze: bool = True):
        """
        Freeze or unfreeze encoder parameters
        
        Args:
            freeze: Whether to freeze encoder parameters
        """
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
            
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode graph into latent space
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        return self.encoder(x, edge_index)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from latent distribution
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent variables
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
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
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        Full forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            **kwargs: Additional decoder-specific arguments
            
        Returns:
            Dict with outputs and latent variables
        """
        # Encode
        mu, logvar = self.encode(x, edge_index)
        
        # Sample latent variables
        z = self.reparameterize(mu, logvar)
        
        # Decode
        decoder_kwargs = {"edge_index": edge_index, **kwargs}
        outputs = self.decode(z, **decoder_kwargs)
        
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
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
        
        # KL divergence loss
        mu, logvar = outputs["mu"], outputs["logvar"]
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
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
                detailed_losses[name] = {'base_loss': decoder_loss, 'total': decoder_loss}
            
            decoder_losses[name] = decoder_loss
            
            # Add weighted loss to total reconstruction loss
            decoder_weight = decoder_weights.get(name, 1.0)
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
    
    def compute_jacobian(self, z: torch.Tensor, decoder_name: str, node_idx: int = None) -> torch.Tensor:
        """
        Compute Jacobian of a specific decoder with respect to latent space
        
        Args:
            z: Latent variables [num_nodes, latent_dim]
            decoder_name: Name of the decoder to compute Jacobian for
            node_idx: Optional node index to compute Jacobian for
            
        Returns:
            Jacobian matrix
        """
        decoder = self.get_decoder(decoder_name)
        if decoder is None:
            raise ValueError(f"Decoder '{decoder_name}' not found")
            
        if hasattr(decoder, "compute_jacobian"):
            return decoder.compute_jacobian(z, node_idx)
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
