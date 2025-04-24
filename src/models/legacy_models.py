import os
from typing import Optional
from torch import Tensor
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from pythae.data.datasets import BaseDataset
from src.models.legacy_model_configs import VAEConfig
from src.models.model_components.legacy_components import EncoderConvVAE, DecoderConvVAE
from src.models.model_utils import ModelOutput

class VAE(nn.Module):
    """Vanilla Variational Autoencoder model."""

    def __init__(
        self,
        config: VAEConfig,
        encoder: Optional[EncoderConvVAE] = None,
        decoder: Optional[DecoderConvVAE] = None,
    ):
        super().__init__()
        self.config = config
        self.encoder = encoder or EncoderConvVAE(config.ddconfig)
        self.decoder = decoder or DecoderConvVAE(config.ddconfig)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """
        encoder_output = self.encoder(x)
        mu, logvar = encoder_output.embedding, encoder_output.log_covariance
        z = self.reparametrize(mu, logvar)
        recon_x = self.decoder(z)["reconstruction"]

        output = ModelOutput(
            mu=mu,
            logvar=logvar,
            recon_x=recon_x,
            z=z,
            x=x.detach()
        )
        return output

    @staticmethod
    def reparametrize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std


