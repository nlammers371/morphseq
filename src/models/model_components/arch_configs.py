from typing import Tuple, List, Literal
from pydantic.dataclasses import dataclass
from dataclasses import field
import torch
import math

@dataclass
class LegacyArchitecture:

    name: Literal["convAE"] = "convVAE"
    latent_dim: int = 64

    n_out_channels: int = 16
    n_conv_layers: int = 5
    orth_flag: bool = False
    kernel_size: int = 4
    stride: int = 2
    input_dim: Tuple[int, int, int] = (1, 288, 128)


timm_dict = {
    "Efficient-B0-RA": "efficientnet_b0.ra_in1k",   # E-B0-RA
    "Efficient-B4": "efficientnet_b4",           # E-B4
    "ConvNeXt-Tiny": "convnext_tiny_in22k_ft_in1k",
    "Swin-Tiny": "swin_tiny_patch4_window7_224",
    "MaxViT-Tiny": "maxvit_tiny_tf_512.in1k",
    "DeiT-Tiny": "deit_tiny_patch16_224",  # alt
    "RegNet-Y": "regnety_400mf ",
    "ViT-Tiny": "vit_tiny_patch16_224"}
# alt
@dataclass
class TimmArchitecture:

    name: Literal["Efficient-B0-RA", "Efficient-B4", "ConvNeXt-Tiny",
                  "Swin-Tiny", "MaxViT-Tiny", "DeiT-Tiny", "RegNet-Y", "ViT-Tiny"] = "Efficient-B0-RA"
    latent_dim: int = 64
    orth_flag: bool = False
    use_pretrained_weights: bool = True
    dec_use_local_attn: bool = False  # for decoder
    input_dim: Tuple[int, int, int] = (1, 288, 128)

    @property
    def timm_name(self):
        return timm_dict[self.name]


# @dataclass
# class SplitArchitecture(LegacyArchitecture):
#
#     name: Literal["convAESplit"] = "convVAESplit"
#     frac_nuisance_latents: float = 0.2
#
#     @property
#     def latent_dim_bio(self) -> int:
#         # at least 1, rounding up the nuisance count
#         bio = self.latent_dim - math.ceil(self.frac_nuisance_latents * self.latent_dim)
#         return max(bio, 1)
#
#     @property
#     def latent_dim_nuisance(self) -> int:
#         return self.latent_dim - self.latent_dim_bio #np.max([np.floor(self.frac_nuisance_latents * self.latent_dim), 1]).astype(int)
#
#     @property
#     def biological_indices(self):
#         return torch.arange(self.latent_dim_nuisance, self.latent_dim, dtype=torch.int64)
#
#     @property
#     def nuisance_indices(self):
#         return torch.arange(0, self.latent_dim_nuisance, dtype=torch.int64)


@dataclass
class ArchitectureAELDM: # wraps native attributes from LDM repo

    name: Literal["ldmVAE"] = "ldmVAE"
    # Attributes
    double_z: bool = True
    z_channels: int = 64
    resolution: List[int] = field(default_factory=lambda: [288, 128])
    in_channels: int = 1
    out_ch: int = 1
    ch: int = 128
    ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4, 4])
    num_res_blocks: int = 2
    attn_resolutions: List[int] = field(default_factory=lambda: [16, 8])
    dropout: float = 0.0
    # NL addition
    latent_dim: int = 64

    freeze_encoder_trunk: bool = True


# @dataclass
# class SplitArchitectureAELDM(ArchitectureAELDM): # adds split logic
#
#     name: Literal["ldmVAESplit"] = "ldmVAESplit"
#     frac_nuisance_latents: float = 0.2
#
#     @property
#     def latent_dim_bio(self) -> int:
#         # at least 1, rounding up the nuisance count
#         bio = self.latent_dim - math.ceil(self.frac_nuisance_latents * self.latent_dim)
#         return max(bio, 1)
#
#     @property
#     def latent_dim_nuisance(self) -> int:
#         return self.latent_dim - self.latent_dim_bio #np.max([np.floor(self.frac_nuisance_latents * self.latent_dim), 1]).astype(int)
#
#     @property
#     def biological_indices(self):
#         return torch.arange(self.latent_dim_nuisance, self.latent_dim, dtype=torch.int64)
#
#     @property
#     def nuisance_indices(self):
#         return torch.arange(0, self.latent_dim_nuisance, dtype=torch.int64)