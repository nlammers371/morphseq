"""This module leverages the sequential nature of timelapse frames to traine a VAE with a latent space that captures
    time- and perturbation-based morphological similarities

Available samplers (NL: TBD whether all of these work)
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    :nosignatures:

"""

from .seq_vae_config import SeqVAEConfig
from .seq_vae_model import SeqVAE

__all__ = ["SeqVAE", "SeqVAEConfig"]