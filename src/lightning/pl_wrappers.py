import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from typing import Any, Callable, Optional
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import functional as F

class LitModel(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable[..., Any],
        data_cfg: Any,
        lr: float,
        batch_key: str = "data",
    ):
        """
        model:            any nn.Module whose forward(x) returns either
                          - a dataclass/object with attributes
                            (recon_x, embedding, log_covariance, z)
                          - or a tuple you know how to unpack
        loss_fn:         a callable that takes (recon_x, x, log_var, mu, z)
                          or (model_output, batch) depending on model
        data_cfg:        your BaseDataConfig
        lr:              learning rate
        batch_key:       the key in your batch dict for inputs
        """
        super().__init__()                    # always call this first
        self.save_hyperparameters(ignore=["model", "loss_fn", "data_cfg"])

        self.model   = model
        self.loss_fn = loss_fn
        self.data_cfg = data_cfg
        self.lr       = lr
        self.batch_key = batch_key

    def forward(self, x: torch.Tensor) -> Any:
        return self.model(x)

    def _step(self, batch, batch_idx, stage: str):
        x = batch[self.batch_key]
        out = self(x)

        # 1) if model implements a compute_loss method, use it
        if hasattr(self.model, "compute_loss"):
            loss_output = self.model.compute_loss(x, out)
        else:
            loss_output  = self.loss_fn(model_input=batch,
                                          model_output=out,
                                          batch_key=self.batch_key)

        bsz = x.size(0)
        # log the main loss
        self.log(f"{stage}/loss", loss_output.loss, prog_bar=(stage=="train"), on_step=False, on_epoch=True, batch_size=bsz)

        # self.log("train/loss", loss_output.loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/recon_loss", loss_output.recon_loss, on_step=False, on_epoch=True, batch_size=bsz)
        self.log(f"{stage}/kld_loss", loss_output.KLD, on_step=False, on_epoch=True, batch_size=bsz)
        if "metric_loss" in loss_output:
            self.log(f"{stage}/metric_loss", loss_output.metric_loss, on_step=False, on_epoch=True, batch_size=bsz)

        return loss_output.loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        ds = self.data_cfg.create_dataset()
        # get indices for images to use for training
        train_indices = self.data_cfg.train_indices
        train_sampler = SubsetRandomSampler(train_indices)

        return DataLoader(
            ds,
            batch_size=self.data_cfg.batch_size,
            num_workers=self.data_cfg.num_workers,
            sampler=train_sampler,
            shuffle=False,
        )

    def val_dataloader(self):
        ds = self.data_cfg.create_dataset()
        eval_indices = self.data_cfg.eval_indices
        eval_sampler = SubsetRandomSampler(eval_indices)
        return DataLoader(
            ds,
            batch_size=self.data_cfg.batch_size,
            num_workers=self.data_cfg.num_workers,
            sampler=eval_sampler,
            shuffle=False,
        )

    # -------------------------------------------------------------------
    # prediction API: given a batch → return recon, recon-loss, mu, logσ²
    # -------------------------------------------------------------------
    def predict_step(self, batch, batch_idx, dataloader_idx=0, recon_loss_type="mse"):
        x = batch["data"]
        snip_ids = batch["label"][0]  # list[str] already

        out = self.model(x)  # your plain nn.Module
        recon_x = out.recon_x
        mu = out.mu
        log_var = out.logvar
        if recon_loss_type == "mse":
            recon_loss = F.mse_loss(
                recon_x.view(recon_x.size(0), -1),
                x.view(x.size(0), -1),
                reduction="none"
            ).sum(dim=1)  # (B,)
        else:
            raise NotImplementedError

        return {
            "snip_ids": snip_ids,  # list[str]
            "mode": self.current_mode,  # set by caller
            "recon": recon_x.cpu(),  # keep on CPU for plotting
            "orig": x.cpu(),
            "recon_loss": recon_loss.cpu(),
            "recon_loss_type": recon_loss_type,# (B,)
            "mu": mu.cpu(),  # (B, D)
            "log_var": log_var.cpu(),  # (B, D)
        }