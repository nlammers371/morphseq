from typing import Tuple
from pydantic.dataclasses import dataclass
from dataclasses import field
import torch
import math

@dataclass
class LegacyArchitecture:
    latent_dim: int = 64

    n_out_channels: int = 16
    n_conv_layers: int = 5
    orth_flag: bool = False
    kernel_size: int = 4
    stride: int = 2
    input_dim: Tuple[int, int, int] = (1, 288, 128)

@dataclass
class SplitArchitecture(LegacyArchitecture):

    frac_nuisance_latents: float = 0.05

    @property
    def latent_dim_bio(self) -> int:
        # at least 1, rounding up the nuisance count
        bio = self.latent_dim - math.ceil(self.frac_nuisance_latents * self.latent_dim)
        return max(bio, 1)

    @property
    def latent_dim_nuisance(self) -> int:
        return self.latent_dim - self.latent_dim_bio #np.max([np.floor(self.frac_nuisance_latents * self.latent_dim), 1]).astype(int)

    @property
    def biological_indices(self):
        return torch.arange(self.latent_dim_nuisance, self.latent_dim, dtype=torch.int64)

    @property
    def nuisance_indices(self):
        return torch.arange(0, self.latent_dim_nuisance, dtype=torch.int64)