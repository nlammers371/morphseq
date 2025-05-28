import torch
import torch.nn as nn
from timm import create_model
from src.models.model_components.arch_configs import LegacyArchitecture
from scipy.stats import ortho_group
from src.models.model_utils import ModelOutput

PATCH_FAMILIES = ("vit", "deit", "mixer", "swin_mlp")   # edit as you add exotic tags

class TimmEncoder(nn.Module):
    """
    Generic encoder → (mu, logvar, optional skip list)
    Works with any timm backbone that either
      (a) supports features_only=True  **or**
      (b) is a patch-token Vision Transformer.
    """
    def __init__(self, cfg: LegacyArchitecture) -> None:

        super().__init__()
        self.cfg = cfg
        self.model_name = cfg.encoder_name.lower()
        self.return_skips = False #return_skips
        self.use_pretrained_weights = cfg.use_pretrained_weights
        self.latent_dim = cfg.latent_dim

        # -------- 1) build backbone --------
        if self._is_patch_family():
            # ViT-style backbone (tokens out)
            self.backbone = create_model(self.encoder_name, pretrained=self.use_pretrained_weights)
            self.embed_dim = self.backbone.num_features  # e.g. 192 / 768
        else:
            # Conv / hierarchical backbone (maps out)
            self.backbone = create_model(
                self.encoder_name, pretrained=self.use_pretrained_weights,
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
            mu, logvar = self.embedding(vec), self.logvar(vec)
            # if self.return_skips:
            #     # pick the four deepest for UNet decoder (E4..E1)
            #     skips = feats[-5:-1] if len(feats) >= 5 else None
            #     return mu, logvar, skips
            return ModelOutput(embedding=mu, log_covariance=logvar)

