import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELossBasic(nn.Module):

    def __init__(self, kld_weight=1.0, reconstruction_loss="mse"):
        super().__init__()

        self.kld_weight = kld_weight
        self.reconstruction_loss = reconstruction_loss

    def forward(self, model_output):

        x = model_output.x
        recon_x = model_output.recon_x
        logvar = model_output.logvar
        mu = model_output.mu

        if self.reconstruction_loss == "mse":
            recon_loss = (
                0.5
                * F.mse_loss(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ).sum(dim=-1)
            )

        elif self.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        return recon_loss.mean(dim=0) + self.kld_weight*KLD.mean(dim=0),  recon_loss.mean(dim=0), KLD.mean(dim=0)