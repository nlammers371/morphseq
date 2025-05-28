# decoders/unidec_lite.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from src.models.model_utils import ModelOutput
import timm

# ---------- building blocks ----------
class UpBlock(nn.Module):
    """Depth-wise conv → point-wise conv → BN → SiLU → pixel-shuffle ↑2."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.out_ch = out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch * 4, 1, bias=False),
            nn.BatchNorm2d(out_ch * 4),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return F.pixel_shuffle(x, 2)  # H,W ×2 ; C ÷4


# ---------- universal decoder ----------
class UniDecLite(nn.Module):
    """
    Universal decoder for 288×128 images (stride-32 encoders).
    Works with or without skip features and lets you schedule their influence
    through a scalar .skip_weight that you can update every epoch.
    """
    def __init__(
        self,
        enc_ch_last: int,         # channels of last encoder stage (e.g. 1280)
        z_dim: int,               # latent size
        out_ch: int = 3,
        skip_chs: Optional[List[int]] = None,  # NL: not used currently
        use_local_attn: bool = False           # set True if you later add Swin blk
    ):
        super().__init__()
        self.lat_h, self.lat_w = 9, 4          # for 288×128; compute if cfg varies
        self.skip_weight = 0.0                 # <-- will be scheduled during train

        # fc reshape from latent z
        self.fc = nn.Linear(z_dim, enc_ch_last * self.lat_h * self.lat_w)

        # five up-sampling stages
        self.up = nn.ModuleList([
            UpBlock(enc_ch_last, 512),
            UpBlock(512, 256),
            UpBlock(256, 128),
            UpBlock(128,  64),
            UpBlock( 64,  32)
        ])

        # 1×1 adapters for incoming skip maps (if provided)
        self.skip_proj = nn.ModuleList()
        for i, ch in enumerate(skip_chs or []):
            tgt = self.up[i].out_ch
            self.skip_proj.append(nn.Conv2d(ch, tgt, 1) if ch else None)

        # (optional) one local-attention block right after UpBlock4
        if use_local_attn:
            from timm.layers import WindowAttention
            self.local_attn = WindowAttention(
                dim=256, window_size=(7, 7), num_heads=4, qkv_bias=True)
            self.norm = nn.LayerNorm(256)
        else:
            self.local_attn = None

        self.to_rgb = nn.Conv2d(32, out_ch, 3, padding=1)

    # ---------- forward ----------
    def forward(self, z, skips: Optional[List[torch.Tensor]] = None):
        """
        z:     [B, z_dim]
        skips: [E4,E3,E2,E1] encoder feature maps OR None
        """
        x = self.fc(z).view(z.size(0), -1, self.lat_h, self.lat_w)

        for i, up in enumerate(self.up):
            x = up(x)

            # fuse skip i (if any) with current feature map
            if skips and i < len(skips) and skips[i] is not None and \
               self.skip_proj[i] is not None and self.skip_weight > 0:
                skip = self.skip_proj[i](skips[i])
                # linear blend → allows gradual fade-in
                x = torch.cat([x, self.skip_weight * skip], dim=1)

            # optional local attention after UpBlock4 (idx 1) for GAN sharpness
            if self.local_attn and i == 1:
                B, C, H, W = x.shape
                x_flat = x.flatten(2).transpose(1, 2)          # B, HW, C
                x_norm = self.norm(x_flat)
                x_attn = self.local_attn(x_norm) + x_flat
                x = x_attn.transpose(1, 2).view(B, C, H, W)

        recon = torch.tanh(self.to_rgb(x))
        return ModelOutput(reconstruction=recon)