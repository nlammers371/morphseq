import os
from typing import Optional
from torch import Tensor
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from pythae.data.datasets import BaseDataset
from src.models.model_configs import VAEConfig, morphVAEConfig
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


class morphVAE(nn.Module):
    """VAE that incorporates split latent space and metric learning."""

    def __init__(
        self,
        config: morphVAEConfig,
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
        self.vanilla = False
        if (len(x.shape) != 5):
            self.vanilla = True

        if self.vanilla: # do normal VAE pass if not training
            encoder_output = self.encoder(x)
            mu, logvar = encoder_output.embedding, encoder_output.log_covariance
            z = self.reparametrize(mu, logvar)
            recon_x = self.decoder(z)["reconstruction"]


        elif self.config.lossconfig.target == "NT-Xent":

            # 1) split out the two views
            x0, x1 = x.unbind(dim=1)  # each is (B, C, H, W)

            # 2) stack them into a single 2B batch, *block-wise*
            x_all = torch.cat([x0, x1], dim=0)  # (2B, C, H, W)

            # 3) run everything in one shot
            enc = self.encoder(x_all)
            mu = enc.embedding  # (2B, D)
            logvar = enc.log_covariance  # (2B, D)
            B = x0.shape[0]
            z = self.reparametrize(mu[:B], logvar[:B]) # we only need the actual samples (not positive pairs)
            recon_x = self.decoder(z)["reconstruction"]

        else:
            raise NotImplementedError

        output = ModelOutput(
                        mu=mu,
                        logvar=logvar,
                        recon_x=recon_x,
                        z=z
                    )

        return output

    @staticmethod
    def reparametrize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std


