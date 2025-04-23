import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELossBasic(nn.Module):

    def __init__(self, kl_weight=1.0, reconstruction_loss="mse"):
        super().__init__()

        self.kl_weight = kl_weight
        self.reconstruction_loss = reconstruction_loss
        # self.reduction = reduction

    def forward(self, x, recon_x, log_var, mu):

        if self.model_config.reconstruction_loss == "mse":
            recon_loss = (
                0.5
                * F.mse_loss(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ).sum(dim=-1)
            )

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        return recon_loss.mean(dim=0) + self.beta*KLD.mean(dim=0),  recon_loss.mean(dim=0), KLD.mean(dim=0)