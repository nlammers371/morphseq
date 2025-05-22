# lpips_helper.py
import torch, lpips
from functools import lru_cache
from contextlib import contextmanager

# ------------------------------------------------------------------ #
# 1) build-once cache: LPIPS(net="alex") lives on *CPU* forever
# ------------------------------------------------------------------ #
@lru_cache()
def _build_lpips(backbone: str, device: torch.device):
    net = lpips.LPIPS(net=backbone)          # builds on CPU
    net = net.to(device)                     # move once if needed
    net.requires_grad_(False).eval()
    return net

# ------------------------------------------------------------------ #
# 3) public helper: returns a scalar LPIPS score (mean over batch)
# ------------------------------------------------------------------ #
def lpips_score(model_input, model_output, backbone="alex", batch_key = "data", use_gpu = False) -> torch.Tensor:
    """Compute LPIPS on CPU; tensors can be on GPU or CPU."""

    target = model_input[batch_key]
    recon = model_output.recon_x

    device = torch.device("cuda") if use_gpu and torch.cuda.is_available() \
        else torch.device("cpu")

    net = _build_lpips(backbone, device)                    # cached instance
           # == cpu
    recon = recon.detach().to(device, non_blocking=True)
    tgt   = target.detach().to(device, non_blocking=True)

    with torch.no_grad():
        return net(recon, tgt).mean()