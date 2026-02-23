# lpips_helper.py
import lpips
from functools import lru_cache
import piq, torch, torch.nn.functional as F
import torch.nn as nn
from torch import Tensor


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
    if len(target.shape) > 4:
        target, _ = target.unbind(dim=1)
    recon = model_output.recon_x

    device = torch.device("cuda") if use_gpu and torch.cuda.is_available() \
        else torch.device("cpu")

    net = _build_lpips(backbone, device)                    # cached instance
           # == cpu
    recon = recon.detach().to(device, non_blocking=True)
    tgt   = target.detach().to(device, non_blocking=True)

    with torch.no_grad():
        return net(recon, tgt).mean()


def ssim_score(model_input, model_output, batch_key = "data", use_gpu = True, n_channels=1) -> torch.Tensor:
    """Compute LPIPS on CPU; tensors can be on GPU or CPU."""

    target = model_input[batch_key]
    if len(target.shape) > 4:
        target, _ = target.unbind(dim=1)  # add channel dim if missing
    recon = model_output.recon_x

    device = torch.device("cuda") if use_gpu and torch.cuda.is_available() \
        else torch.device("cpu")

    ms_ssim_loss = piq.MultiScaleSSIMLoss(
        data_range=1.0,  # images in [0,1]
        kernel_size=7,  # default
        kernel_sigma=1.5,
        scale_weights=None  # default paper weights
    ).to(device=device, non_blocking=True)

    recon = recon.detach().to(device, non_blocking=True)
    tgt   = target.detach().to(device, non_blocking=True)
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            val_ms = ms_ssim_loss(recon, tgt)

    return val_ms


# helper to monitor latent curvature
class LatentCovarianceLoss(nn.Module):
    """
    Penalises deviation of the latent empirical covariance from the identity matrix.
    
    Returns a *scalar* suitable for logging or adding (with a weight) to the total loss.
    
    Args
    ----
    mode : {'fro', 'off'}   (default: 'fro')
        * 'fro'  – full Frobenius ‖Σ - I‖²_F  / D².  
        * 'off' – mean‐square of off-diagonal elements only.
    normalize : bool (default: True)
        Divide by dimension so the scale is roughly invariant to latent size.
    eps : float
        Numerical stabiliser for small batch sizes.
    """
    def __init__(self, latent_indices=None, mode: str = "off", normalize: bool = True, eps: float = 1e-8):
        super().__init__()
        assert mode in ("fro", "off")
        self.mode, self.normalize, self.eps, self.latent_indices = mode, normalize, eps, latent_indices

    def forward(self, model_output) -> Tensor:
        """
        Parameters
        ----------
        z : (B, D) tensor of latent samples *after* reparameterisation.

        Returns
        -------
        Tensor scalar
        """
        # extract latent vector
        z = model_output.mu
        if self.latent_indices is not None:
            z = z[:, self.latent_indices]

        # get covariance
        B, D = z.shape
        zc = z - z.mean(dim=0, keepdim=True)                   # centre
        cov = (zc.t() @ zc) / (B - 1 + self.eps)               # unbiased Σ̂
        eye = torch.eye(D, device=z.device, dtype=z.dtype)

        # calculate directional warping
        # cond = torch.linalg.eigvalsh(cov).max() / torch.linalg.eigvalsh(cov).min()
        eigvals = torch.linalg.eigvalsh(cov.float())
        eps = 1e-8  # or slightly smaller
        eigvals_clipped = eigvals.clamp(min=eps)
        cond = eigvals_clipped.max() / eigvals_clipped.min() 
        # calculate anisotropy
        if self.mode == "fro":
            diff = cov - eye
            loss = (diff ** 2).sum()
            if self.normalize:
                loss = loss / (D * D)
        else:  # 'off'
            off_diag_mask = torch.ones_like(cov) - eye
            loss = ((cov * off_diag_mask) ** 2).sum()
            if self.normalize:
                loss = loss / (D * (D - 1))

            loss = loss.sqrt()

        return loss, cond