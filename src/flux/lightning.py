import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Sequence, Dict, Any
from src.flux.model import MLPVelocityField


class HierarchicalClockNVF(pl.LightningModule):
    """PyTorch‑Lightning module that trains a neural‑velocity field with
    hierarchical (experiment → embryo) clock‑scaling.

    Velocity model
        v_pred = γ_{e,k} * f_θ(z)
        γ_{e,k} = exp(log_s_e + δ_{e,k})
    where
        log_s_e  : learnable per‑experiment log‑scale
        δ_{e,k}  : learnable embryo‑specific deviation with Gaussian shrinkage.
    """

    def __init__(
        self,
        dim: int,
        num_exp: int,
        num_embryo: int,
        infer_embryo_clock: bool = False,
        hidden: Sequence[int] = (256, 128, 128),
        sigma: float = 0.2,  # shrinkage std‑dev for δ
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        
        super().__init__()
        self.save_hyperparameters()

        # shared geometry network f_θ
        self.field = MLPVelocityField(dim, hidden)

        # hierarchical clock parameters
        self.log_s = nn.Parameter(torch.zeros(num_exp))   # experiment speed (log‑space)
        self.delta = nn.Parameter(torch.zeros(num_embryo))  # embryo deviation

        # hyper‑params
        self.sigma = sigma
        self.lr = lr
        self.weight_decay = weight_decay
        self.infer_embryo_clock = infer_embryo_clock

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, z: torch.Tensor, exp_idx: torch.Tensor, emb_idx: torch.Tensor) -> torch.Tensor:
        """Predict velocity for latent positions *z*.

        Args:
            z:        (B, d) latent embeddings
            exp_idx:  (B,) experiment indices (int)
            emb_idx:  (B,) embryo indices (int)
        Returns:
            (B, d) velocity predictions
        """
        if self.infer_embryo_clock:
            gamma = torch.exp(self.log_s[exp_idx] + self.delta[emb_idx])  # (B,)
        else:
            gamma = torch.exp(self.log_s[exp_idx])
            
        return self.field(z) * gamma.unsqueeze(-1)

    # --------------------------------------------------------
    # Loss helpers
    # --------------------------------------------------------
    def _loss(self, batch: Dict[str, torch.Tensor]):
        z0 = batch["z0"]            # (B, d)
        dz_target = batch["dz"]            # (B, d)
        exp_idx = batch["exp"]      # (B,)
        emb_idx = batch["emb"]      # (B,)
        
        # finite‑difference velocity
        dz_pred = self.forward(z0, exp_idx, emb_idx)

        loss_mse = torch.mean((dz_pred - dz_target) ** 2)
        loss_reg = torch.mean(self.delta ** 2) / (self.sigma ** 2)
        loss = loss_mse + loss_reg

        return loss, {
            "mse": loss_mse.detach(),
            "reg": loss_reg.detach(),
            "total": loss.detach(),
        }

    # --------------------------------------------------------
    # Lightning hooks
    # --------------------------------------------------------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, logs = self._loss(batch)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, logs = self._loss(batch)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
