import torch
from src.losses import *
from src.losses import vae_loss_basic
from src.vae.models import VAE

# 1) your registry mapping names → classes
OBJECTIVE_REGISTRY = {
    "VAELossBasic": vae_loss_basic.VAELossBasic,
    # "bce": torch.nn.BCELoss,
    # "LPIPSWithDiscriminator": LPIPSWithDiscriminator,
    # etc…
}

# DATASET_REGISTRY = {
#     "BasicDataset":
# }

# MODEL_REGISTRY = {
#     "VAE":
# }