from timm import create_model
from scipy.stats import ortho_group
# decoders/unidec_lite.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from src.models.model_utils import ModelOutput
from src.models.model_components.arch_configs import TimmArchitecture

PATCH_FAMILIES = ("vit", "deit", "mixer", "swin_mlp")   # edit as you add exotic tags

class TimmEncoder(nn.Module):
    """
    Generic encoder → (mu, logvar, optional skip list)
    Works with any timm backbone that either
      (a) supports features_only=True  **or**
      (b) is a patch-token Vision Transformer.
    """
    def __init__(self, cfg: TimmArchitecture) -> None:

        super().__init__()
        self.cfg = cfg
        self.model_name = cfg.timm_name.lower()
        # self.return_skips = False #return_skips
        self.use_pretrained_weights = cfg.use_pretrained_weights
        self.latent_dim = cfg.latent_dim

        # -------- 1) build backbone --------
        if self._is_patch_family():
            # ViT-style backbone (tokens out)
            self.backbone = create_model(self.model_name, pretrained=self.use_pretrained_weights, in_chans=self.cfg.input_dim[0])
            self.embed_dim = self.backbone.num_features  # e.g. 192 / 768
        else:
            # Conv / hierarchical backbone (maps out)
            self.backbone = create_model(
                self.model_name, pretrained=self.use_pretrained_weights, in_chans=self.cfg.input_dim[0],
                features_only=True, out_indices=None   # we’ll choose later
            )
            self.embed_dim = self.backbone.feature_info.channels()[-1]

        # -------- 2) heads to parameterise q(z|x) --------
        self.embedding     = nn.Linear(self.embed_dim, self.latent_dim)
        self.log_var = nn.Linear(self.embed_dim, self.latent_dim)

        # Global pool layer only needed for feature-map families
        if not self._is_patch_family():
            self.pool = nn.AdaptiveAvgPool2d(1)

        # flag for whether to force into orthonormal basis
        if cfg.orth_flag:
            self._init_orthonormal(cfg.latent_dim)
        else:
            self.project = None

    # helper function for othornormal basis
    def _init_orthonormal(self, dim):
        W = ortho_group.rvs(dim=dim).astype('float32')
        self.register_buffer('A', torch.from_numpy(W))
        self.project = nn.Linear(dim, dim, bias=False)
        for p in self.project.parameters():
            p.requires_grad = False

    # helper ------------------------------------------------
    def _is_patch_family(self):
        return any(self.model_name.startswith(p) for p in PATCH_FAMILIES)

    # ---------------------------------------
    def forward(self, x):
        if self._is_patch_family():                         # ViT / DeiT
            # ① forward to patch tokens
            tokens = self.backbone.forward_features(x)      # [B,N,C]
            cls_or_mean = tokens.mean(dim=1)                # mean-pool tokens
            mu, logvar = self.embedding(cls_or_mean), self.logvar(cls_or_mean)

            return ModelOutput(embedding=mu, log_covariance=logvar)


        else:                                               # Conv / Swin / MaxViT
            feats = self.backbone(x)                        # list of stage maps
            penult = feats[-1]                              # deepest
            vec = self.pool(penult).flatten(1)              # [B,C]
            mu, logvar = self.embedding(vec), self.log_var(vec)
            # if self.return_skips:
            #     # pick the four deepest for UNet decoder (E4..E1)
            #     skips = feats[-5:-1] if len(feats) >= 5 else None
            #     return mu, logvar, skips
            return ModelOutput(embedding=mu, log_covariance=logvar)

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


# Vanilla conv decodewr
# ---------- universal decoder ----------
class UniDecLite(nn.Module):
    """
    Universal decoder for 288×128 images (stride-32 encoders).
    Works with or without skip features and lets you schedule their influence
    through a scalar .skip_weight that you can update every epoch.
    """
    def __init__(self, cfg: TimmArchitecture, enc_ch_last: int) -> None:
        super().__init__()

        self.cfg = cfg
        self.enc_ch_last = enc_ch_last
        H, W = torch.tensor(self.cfg.input_dim[1]), torch.tensor(self.cfg.input_dim[2])
        self.lat_h, self.lat_w = torch.floor(H/32).int(), torch.floor(W/32).int()
        self.out_ch = cfg.input_dim[0] # for 288×128; compute if cfg varies
        # self.skip_weight = 0.0                 # <-- will be scheduled during train
        self.model_name = cfg.name.lower()
        self.use_pretrained_weights = cfg.use_pretrained_weights
        self.latent_dim = cfg.latent_dim
        self.use_local_attn = cfg.dec_use_local_attn
        # fc reshape from latent z
        self.fc = nn.Linear(self.latent_dim, self.enc_ch_last * self.lat_h * self.lat_w)

        # five up-sampling stages
        self.up = nn.ModuleList([
            UpBlock(self.enc_ch_last, 512),
            UpBlock(512, 256),
            UpBlock(256, 128),
            UpBlock(128,  64),
            UpBlock( 64,  32)
        ])

        # 1×1 adapters for incoming skip maps (if provided)
        # self.skip_proj = nn.ModuleList()
        # for i, ch in enumerate(skip_chs or []):
        #     tgt = self.up[i].out_ch
        #     self.skip_proj.append(nn.Conv2d(ch, tgt, 1) if ch else None)

        # (optional) one local-attention block right after UpBlock4
        if self.use_local_attn:
            from src.models.model_components.window_attention import WindowAttention
            self.local_attn = WindowAttention(
                dim=256, window_size=(7, 7), num_heads=4, qkv_bias=True)
            self.norm = nn.LayerNorm(256)
        else:
            self.local_attn = None

        self.to_img = nn.Conv2d(32, self.out_ch, 3, padding=1, bias=False)

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
            # if skips and i < len(skips) and skips[i] is not None and \
            #    self.skip_proj[i] is not None and self.skip_weight > 0:
            #     skip = self.skip_proj[i](skips[i])
            #     # linear blend → allows gradual fade-in
            #     x = torch.cat([x, self.skip_weight * skip], dim=1)

            # optional local attention after UpBlock4 (idx 1) for GAN sharpness
            if self.local_attn and i == 1:
                B, C, H, W = x.shape
                x_flat = x.flatten(2).transpose(1, 2)          # B, HW, C
                x_norm = self.norm(x_flat)
                x_attn = self.local_attn(x_norm) + x_flat
                x = x_attn.transpose(1, 2).view(B, C, H, W)

        recon = torch.sigmoid(self.to_img(x))
        return ModelOutput(reconstruction=recon)