from typing import Tuple
from pydantic.dataclasses import dataclass
import numpy as np

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
    def n_nuisance_latents(self) -> str:
        return np.max([np.floor(self.frac_nuisance_latents * self.latent_dim), 1]).astype(int)
