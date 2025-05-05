import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from typing import Any, Callable, Optional, Dict
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import functional as F
from torch import nn
from src.models.ldm_models import AutoencoderKLModel
from src.lightning.pl_utils import ramp_weight, cosine_ramp_weight


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

        # 1) compute kld and pips weights
        kld_w = self._kld_weight()
        self.loss_fn.kld_weight = kld_w
        pips_w = self._pips_weight()
        self.loss_fn.pips_weight = pips_w

        if hasattr(self, "metric_weight"):
            metric_w = self._metric_weight()
            self.loss_fn.metric_weight = metric_w

        if hasattr(self.model, "compute_loss"):
            loss_output = self.model.compute_loss(x, out)
        else:
            # break_flag = self.current_epoch==3
            loss_output  = self.loss_fn(model_input=batch,
                                        model_output=out,
                                        batch_key=self.batch_key)

        bsz = x.size(0)

        # log weights, sync_dist=True
        self.log(f"{stage}/pips_weight", pips_w, on_step=False, on_epoch=True, rank_zero_only=True)#, sync_dist=True)
        self.log(f"{stage}/kld_weight", kld_w, on_step=False, on_epoch=True, rank_zero_only=True)#, sync_dist=True)

        # log the main loss
        self.log(f"{stage}/loss", loss_output.loss, prog_bar=(stage=="train"), on_step=False, on_epoch=True, batch_size=bsz, rank_zero_only=True)#, sync_dist=True)

        # self.log("train/loss", loss_output.loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/recon_loss", loss_output.recon_loss, on_step=False, on_epoch=True, batch_size=bsz, rank_zero_only=True)#, sync_dist=True)
        self.log(f"{stage}/pixel_loss", loss_output.pixel_loss, on_step=False, on_epoch=True, batch_size=bsz, rank_zero_only=True)#, sync_dist=True)
        self.log(f"{stage}/pips_loss", loss_output.pips_loss, on_step=False, on_epoch=True, batch_size=bsz, rank_zero_only=True)#, sync_dist=True)
        self.log(f"{stage}/kld_loss", loss_output.KLD, on_step=False, on_epoch=True, batch_size=bsz, rank_zero_only=True)#, sync_dist=True)

        if "metric_loss" in loss_output:
            self.log(f"{stage}/metric_weight", self.loss_fn.metric_weight, on_step=False, on_epoch=True, batch_size=bsz,
                     rank_zero_only=True)
            self.log(f"{stage}/metric_loss", loss_output.metric_loss, on_step=False, on_epoch=True, batch_size=bsz, rank_zero_only=True)

        if "pips_loss" in loss_output:
            self.log(f"{stage}/pips_loss", loss_output.pips_loss, on_step=False, on_epoch=True, batch_size=bsz, rank_zero_only=True)#, sync_dist=True)

        return loss_output.loss

    def _kld_weight(self) -> float:
        """Current β according to ramp-up schedule."""
        if self.loss_fn.schedule_kld:
            return cosine_ramp_weight(
                step_curr=self.current_epoch,
                **self.loss_fn.kld_cfg,
            )
        else:
            return self.loss_fn.kld_weight
        
    def _pips_weight(self) -> float:
        """Current β according to ramp-up schedule."""
        if self.loss_fn.schedule_pips:
            return cosine_ramp_weight(
                step_curr=self.current_epoch,
                **self.loss_fn.pips_cfg,
            )
        else:
            return self.loss_fn.pips_weight

    def _metric_weight(self) -> float:
        """Current β according to ramp-up schedule."""
        if self.loss_fn.schedule_metric:
            return cosine_ramp_weight(
                step_curr=self.current_epoch,
                **self.loss_fn.metric_cfg,
            )
        else:
            return self.loss_fn.metric_weight

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





class LitAutoencoderKL(pl.LightningModule):
    """
    Lightning wrapper that handles training, logging, checkpoints.
    """
    def __init__(
        self,
        model: AutoencoderKLModel,
        loss_fn: nn.Module,
        learning_rate: float,
        monitor: Optional[str] = None,
    ):
        super().__init__()
        # capture hyperparameters except large objects
        self.save_hyperparameters(ignore=["model", "loss_fn"])
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        if monitor:
            self.monitor = monitor

    def forward(self, batch: Dict[str, Any]) -> Any:
        x = self.model.get_input(batch)
        return self.model(x)

    def training_step(self, batch: Dict[str, Any], batch_idx: int, optimizer_idx: int = 0):
        x = self.model.get_input(batch)
        recon, posterior = self.model(x)
        loss, logs = self.loss_fn(x, recon, posterior,
                                  optimizer_idx, self.global_step,
                                  last_layer=self.model.get_last_layer(),
                                  split="train")
        # log main and any extra
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(logs, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        x = self.model.get_input(batch)
        recon, posterior = self.model(x)
        loss, logs = self.loss_fn(x, recon, posterior,
                                  0, self.global_step,
                                  last_layer=self.model.get_last_layer(),
                                  split="val")
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log_dict(logs)

    def configure_optimizers(self):
        lr = self.learning_rate
        # separate optimizers as in original
        opt_ae = torch.optim.Adam(
            list(self.model.encoder.parameters()) +
            list(self.model.decoder.parameters()) +
            list(self.model.quant_conv.parameters()) +
            list(self.model.post_quant_conv.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )
        opt_disc = torch.optim.Adam(
            self.loss_fn.discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    @torch.no_grad()
    def log_images(self, batch: Dict[str, Any], only_inputs: bool = False) -> None:
        # example of logging images at epoch end
        x = self.model.get_input(batch).to(self.device)
        logs = {}
        logs["inputs"] = x
        if not only_inputs:
            recon, posterior = self.model(x)
            # optionally colorize here
            logs["reconstructions"] = recon
            logs["samples"] = self.model.decode(posterior.sample())
        for k, v in logs.items():
            # log image grids
            self.logger.experiment.add_images(
                f"{k}/{self.current_epoch}", v, self.current_epoch
            )
        return logs