""" 
This is the heart of pythae! 
Here are implemented some of the most common (Variational) Autoencoders models.

By convention, each implemented model is stored in a folder located in :class:`pythae.models`
and named likewise the model. The following modules can be found in this folder:

- | *modelname_config.py*: Contains a :class:`ModelNameConfig` instance inheriting
    from either :class:`~pythae.models.base.AEConfig` for Autoencoder models or 
    :class:`~pythae.models.base.VAEConfig` for Variational Autoencoder models. 
- | *modelname_model.py*: An implementation of the model inheriting either from
    :class:`~pythae.models.AE` for Autoencoder models or 
    :class:`~pythae.models.base.VAE` for Variational Autoencoder models. 
- *modelname_utils.py* (optional): A module where utils methods are stored.
"""

from .base import BaseAE, BaseAEConfig
from .metric_vae import MetricVAE, MetricVAEConfig
from .vae import VAE, VAEConfig
from .seq_vae import SeqVAE, SeqVAEConfig
from .morph_iaf_vae import MorphIAFVAE, MorphIAFVAEConfig

__all__ = [
    "BaseAE",
    "BaseAEConfig",
    "VAE",
    "VAEConfig",
    "MetricVAE",
    "MetricVAEConfig",
    "SeqVAE",
    "SeqVAEConfig",
    "MorphIAFVAE",
    "MorphIAFVAEConfig"
]
