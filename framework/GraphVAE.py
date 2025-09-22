from framework.VAE import VAE
from framework.KLAnnealingScheduler import KLAnnealingScheduler
from framework.encoder import Encoder
from framework.decoder import DecoderBase
from framework.prior import Prior

from typing import List, Optional


class GraphVAE(VAE):

    """
    Graph Variational Autoencoder (GraphVAE) extending the base VAE class
    for graph-structured data.
    
    Inherits from VAE and can be customized with graph-specific encoders and decoders.
    """
    
    def __init__(self, encoder: "Encoder", decoders: List["DecoderBase"], 
                 kl_scheduler: Optional["KLAnnealingScheduler"] = None,
                 freeze_encoder: bool = False,
                 prior: Optional["Prior"] = None,
                 compute_latent_manifold: bool = True):
        super(GraphVAE, self).__init__(encoder, decoders, kl_scheduler=kl_scheduler, freeze_encoder=freeze_encoder, prior=prior, compute_latent_manifold=compute_latent_manifold)
