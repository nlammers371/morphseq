import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from typing import Any, Callable, Optional
from torch.utils.data.sampler import SubsetRandomSampler

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
            loss, *metrics = self.model.compute_loss(x, out)
        else:
            loss, *metrics = self.loss_fn(out)

        # log the main loss
        self.log(f"{stage}/loss", loss, prog_bar=(stage=="train"))
        # if any extra metrics came back (e.g. reconst, kld), log them too
        for i, name in enumerate(("reconst", "kld")):
            if i < len(metrics):
                self.log(f"{stage}/{name}", metrics[i])

        return loss

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