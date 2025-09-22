import torch
from abc import ABC, abstractmethod

class Prior(ABC):
    """Abstract base class for priors"""

    def giveVAEInstance(self, model):
        self.model = model
    
    @abstractmethod
    def sample(self, mu: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Sample from the prior distribution"""
        pass
    
    @abstractmethod
    def kl_divergence(self, mu: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence with respect to the prior"""
        pass
    
    @abstractmethod
    def get_param_size(self) -> int:
        """Return the number of parameters needed (e.g., 1 for logvar, 2 for concentration)"""
        pass

class GaussianPrior(Prior):
    """Gaussian prior N(0, I)"""
    
    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if mu.device != logvar.device:
            logvar = logvar.to(mu.device)
        
        if mu.requires_grad:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    def get_param_size(self) -> int:
        return 1  # logvar


