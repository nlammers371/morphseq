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

from .morph_iaf_vae_config import MorphIAFVAEConfig
from .morph_iaf_vae_model import MorphIAFVAE

__all__ = ["MorphIAFVAE", "MorphIAFVAEConfig"]