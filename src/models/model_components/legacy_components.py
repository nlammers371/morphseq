import sys
sys.path.append("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")
# from src.vae.models.nn import BaseEncoder, BaseDecoder
from src.models.model_utils import ModelOutput
import torch.nn as nn
import torch
import numpy as np
from typing import Tuple
from pydantic.dataclasses import dataclass
from src.functions.utilities import conv_output_shape
from scipy.stats import ortho_group

@dataclass
class LegacyArchitecture:
    latent_dim: int = 64
    n_out_channels: int = 16
    n_conv_layers: int = 5
    orth_flag: bool = False
    kernel_size: int = 4
    stride: int = 2
    input_dim: Tuple[int, int, int] = (1, 288, 128)

# Define an encoder class with tuneable variables for the number of convolutional layers ad the depth of the conv kernels
class EncoderConvVAE(nn.Module):
    def __init__(self, cfg: LegacyArchitecture) -> None:
        super().__init__()
        self.cfg = cfg
        self.conv_layers = self._make_conv_stack(
            in_ch=cfg.input_dim[0],
            layers=cfg.n_conv_layers,
            base_ch=cfg.n_out_channels,
        )

        # infer flattened dim automatically
        with torch.no_grad():
            dummy = torch.zeros(1, *cfg.input_dim)
            self._feat_dim = self.conv_layers(dummy).numel()

        self.embedding = nn.Linear(self._feat_dim, cfg.latent_dim)
        self.log_var   = nn.Linear(self._feat_dim, cfg.latent_dim)

        if cfg.orth_flag:
            self._init_orthonormal(cfg.latent_dim)
        else:
            self.project = None

    def _make_conv_stack(self, in_ch, layers, base_ch):
        blocks, ch = [], in_ch
        for n in range(layers):
            out_ch = base_ch * (2**n)
            k, s, p = (5,1,2) if (n==0 and layers==7) else (self.cfg.kernel_size, self.cfg.stride, 1)
            blocks += [nn.Conv2d(ch, out_ch, k, s, p),
                       nn.BatchNorm2d(out_ch),
                       nn.ReLU()]
            ch = out_ch
        return nn.Sequential(*blocks)

    def _init_orthonormal(self, dim):
        W = ortho_group.rvs(dim=dim).astype('float32')
        self.register_buffer('A', torch.from_numpy(W))
        self.project = nn.Linear(dim, dim, bias=False)
        for p in self.project.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.conv_layers(x).view(x.size(0), -1)
        mu = self.embedding(h)
        logvar = self.log_var(h)
        if self.project:
            mu = self.project(mu)
        return ModelOutput(embedding=mu, log_covariance=logvar,
                           weight_matrix=(self.project.weight if self.project else None))


# Defines a "matched" decoder class that inherits key features from its paired encoder
class DecoderConvVAE(nn.Module):
    def __init__(self, cfg: LegacyArchitecture) -> None:
        super().__init__()
        self.cfg = cfg

        # 1) figure out the “base” spatial dimensions after encoding
        self.h_base, self.w_base = self._infer_encoded_hw(
            input_hw=cfg.input_dim[1:],
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            layers=cfg.n_conv_layers,
        )

        # 2) total feature dim coming out of the FC
        self.feature_dim = (
            cfg.n_out_channels * 2 ** (cfg.n_conv_layers - 1)
            * self.h_base
            * self.w_base
        )

        # 3) linear layer to expand z → feature vector
        self.fc = nn.Linear(cfg.latent_dim, self.feature_dim)

        # 4) build the deconv stack
        self.deconv_layers = self._make_deconv_stack(
            base_channels=cfg.n_out_channels,
            layers=cfg.n_conv_layers,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            final_channels=cfg.input_dim[0],
        )

    def _infer_encoded_hw(
        self,
        input_hw: Tuple[int,int],
        kernel_size: int,
        stride: int,
        layers: int,
    ) -> Tuple[int,int]:
        """Run the conv_output_shape formula up to 6 layers deep to get H, W."""
        h, w = input_hw
        for _ in range(min(layers, 6)):
            h, w = conv_output_shape((h, w), kernel_size=kernel_size, stride=stride, pad=1)
        return h, w

    def _make_deconv_stack(
        self,
        base_channels: int,
        layers: int,
        kernel_size: int,
        stride: int,
        final_channels: int,
    ) -> nn.Sequential:
        blocks = []
        for n in range(layers):
            # index in reverse
            idx = layers - n - 1
            in_ch  = base_channels * 2 ** idx
            out_ch = final_channels if (n == layers - 1) else base_channels * 2 ** (idx - 1)

            # special 7-layer case
            if n == layers - 1 and layers == 7:
                k, s, p = 5, 1, 2
            else:
                k, s, p = kernel_size, stride, 1

            blocks.append(nn.ConvTranspose2d(in_ch, out_ch, k, s, padding=p))

            if n == layers - 1:
                blocks.append(nn.Sigmoid())
            else:
                blocks.append(nn.BatchNorm2d(out_ch))
                blocks.append(nn.ReLU())

        return nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor) -> ModelOutput:
        # expand z → feature map
        batch = z.size(0)
        x = self.fc(z).view(
            batch,
            self.cfg.n_out_channels * 2 ** (self.cfg.n_conv_layers - 1),
            self.h_base,
            self.w_base,
        )
        recon = self.deconv_layers(x)
        return ModelOutput(reconstruction=recon)