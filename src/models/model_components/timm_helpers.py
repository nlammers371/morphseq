import torch.nn as nn
import torch

def vit_resize(model, new_hw):
    """
    Resize a timm Vision Transformer so it accepts images of size (H, W).

    • Works for square *or* rectangular inputs
    • Modifies the model in place and returns it
    """
    H, W = new_hw
    pe = model.patch_embed                   # PatchEmbed module

    # ------------------------------------------------------------------
    # 1) Update PatchEmbed meta
    # ------------------------------------------------------------------
    gh_old, gw_old = model.patch_embed.grid_size
    ph, pw = pe.patch_size if isinstance(pe.patch_size, tuple) else (pe.patch_size, pe.patch_size)
    pe.img_size  = (H, W)
    pe.grid_size = (H // ph, W // pw)        # new (gh, gw)

    # ------------------------------------------------------------------
    # 2) Interpolate positional embeddings
    # ------------------------------------------------------------------
    # pos_embed shape: (1, 1 + gh_old*gw_old, C)
    cls_tok, pos_tok = model.pos_embed[:, :1], model.pos_embed[:, 1:]
      # before we changed it this equals old (37,37)
    pos_tok = pos_tok.reshape(1, gh_old, gw_old, -1).permute(0, 3, 1, 2)     # (1,C,gh,gw)

    pos_tok = nn.functional.interpolate(
        pos_tok, size=pe.grid_size, mode='bicubic', align_corners=False)

    pos_tok = pos_tok.permute(0, 2, 3, 1).reshape(1, -1, pos_tok.shape[1])
    model.pos_embed = nn.Parameter(torch.cat([cls_tok, pos_tok], dim=1))

    return model