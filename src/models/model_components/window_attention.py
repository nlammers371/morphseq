# ----------------------------------------------------------------------------
# window_ops.py  ―  extracted verbatim from
# https://github.com/rwightman/pytorch-image-models  (Apache-2.0)
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

__all__ = [
    'window_partition',
    'window_reverse',
    'WindowAttention',
]

# --------------------------------------------------------------------------- #
# helper functions
# --------------------------------------------------------------------------- #

def window_partition(x: Tensor, window_size: Tuple[int, int]) -> Tensor:
    """Partition feature map into non-overlapping windows.

    Args:
        x:           (B, H, W, C) tensor
        window_size: (Wh, Ww)

    Returns:
        windows:     (num_windows*B, Wh*Ww, C)
    """
    B, H, W, C = x.shape
    Wh, Ww = window_size
    x = x.view(B,
               H // Wh, Wh,
               W // Ww, Ww,
               C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous() \
              .view(-1, Wh * Ww, C)
    return windows


def window_reverse(windows: Tensor,
                   window_size: Tuple[int, int],
                   H: int,
                   W: int) -> Tensor:
    """Reverse windows back to feature map.

    Args:
        windows:     (num_windows*B, Wh*Ww, C)
        window_size: (Wh, Ww)
        H, W:        original height & width

    Returns:
        x:           (B, H, W, C)
    """
    Wh, Ww = window_size
    B = int(windows.shape[0] / (H * W / Wh / Ww))
    x = windows.view(B,
                     H // Wh, W // Ww,
                     Wh, Ww, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous() \
         .view(B, H, W, -1)
    return x


# --------------------------------------------------------------------------- #
# main module
# --------------------------------------------------------------------------- #

class WindowAttention(nn.Module):
    """Window-based multi-head self attention (W-MSA) module with
    *relative position bias* (as in Swin Transformer).

    Args:
        dim:            input channel dimension
        window_size:    (Wh, Ww)
        num_heads:      number of attention heads
        qkv_bias:       add bias to qkv projections
        attn_drop:      dropout on attention weights
        proj_drop:      dropout on output projection
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int] = (7, 7),
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # relative position bias table
        Wh, Ww = self.window_size
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), num_heads)
        )  # (2*Wh-1)*(2*Ww-1), nH

        # get pair-wise relative position index
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)                                  # (2, Wh*Ww)
        rel_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]       # (2, Wh*Ww, Wh*Ww)
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()                      # (Wh*Ww, Wh*Ww, 2)
        rel_coords[:, :, 0] += Wh - 1
        rel_coords[:, :, 1] += Ww - 1
        rel_coords[:, :, 0] *= 2 * Ww - 1
        rel_pos_index = rel_coords.sum(-1)                                         # (Wh*Ww, Wh*Ww)
        self.register_buffer("rel_pos_index", rel_pos_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)
        nn.init.trunc_normal_(self.qkv.weight, std=.02)
        if qkv_bias:
            nn.init.constant_(self.qkv.bias, 0)
        nn.init.trunc_normal_(self.proj.weight, std=.02)
        nn.init.constant_(self.proj.bias, 0)

    # --------------------------------------------------------------------- #
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, H, W, C)  **or**  (B, N, C) where N = Wh*Ww
        """
        B_, N, C = x.shape

        # Q K V
        qkv = self.qkv(x) \
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads) \
            .permute(2, 0, 3, 1, 4)                                # (3, B, nH, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, nH, N, N)

        # add relative positional bias
        rel_bias = self.rel_pos_bias_table[self.rel_pos_index.view(-1)] \
            .view(N, N, -1) \
            .permute(2, 0, 1)  # (nH, N, N)
        attn = attn + rel_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # attention output
        x = (attn @ v) \
            .transpose(1, 2) \
            .reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x