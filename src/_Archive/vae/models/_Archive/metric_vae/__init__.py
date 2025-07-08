"""This module introduces metric constraints to the standard VAE architecture

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

from .metric_vae_config import MetricVAEConfig
from .metric_vae_model import MetricVAE

__all__ = ["MetricVAE", "MetricVAEConfig"]