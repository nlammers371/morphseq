from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from ..base.base_config import BaseAEConfig


@dataclass
class VAEConfig(BaseAEConfig):
    """VAE config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
    """
    orth_flag: bool = False
    n_conv_layers: int = 5  # number of convolutional layers
    n_out_channels: int = 16
    beta: float = 1.0  # tunes the weight of the KL normalization term
    reconstruction_loss: Literal["bce", "mse"] = "mse"
