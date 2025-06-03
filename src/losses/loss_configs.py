from typing    import Literal, Optional
from pydantic.dataclasses import dataclass
from dataclasses import field
from importlib import import_module
import torch
import numpy as np
from pydantic import ConfigDict
import math


@dataclass
class BasicLoss:
    target: Literal[
        "src.losses.loss_functions.VAELossBasic"
    ] = "src.losses.loss_functions.VAELossBasic"

    max_epochs: int = 25 # this will be overwritten by whatever is passed to the trainconfig

    # KLD
    kld_weight: float = 1.0
    schedule_kld: bool = True
    kld_warmup: int = 0

    # PIXEL
    reconstruction_loss: str = "L1"

    # PIPS
    pips_net: Literal["vgg", "alex", "squeeze"] = "vgg"
    pips_flag: bool = True
    pips_weight: float = 7.5
    schedule_pips: bool = True

    # ADVERSARIAL
    use_gan: bool = False
    gan_weight: float = 1.0
    gan_net: Literal["patch", "ms_patch", "style2"] = "patch"
    schedule_gan: bool = True

    # Extra PIPS to monitor recon quality
    use_pips_eval: bool = True
    eval_pips_net: Literal["vgg", "alex", "squeeze"] = "alex"

    # get scheduler info
    ramp_scale: float = 5.0
    hold_scale: float = 5.0
    tv_weight: float = 0 #1e-5

    @property
    def train_scale(self) -> float:
        return min([self.max_epochs / 30.0, 1.0])

    @property
    def kld_rampup(self) -> int:
        return math.floor(self.ramp_scale * self.train_scale)

    @property
    def pips_warmup(self) -> int:
        return self.kld_rampup + self.kld_warmup + math.floor(self.hold_scale * self.train_scale)

    @property
    def pips_rampup(self) -> int:
        return math.floor(self.ramp_scale * self.train_scale)

    @property
    def gan_warmup(self) -> int:
        if self.pips_weight > 0:
            return self.pips_rampup + self.pips_warmup + math.floor(self.hold_scale * self.train_scale)
        else:
            return self.kld_rampup + self.kld_warmup + math.floor(self.hold_scale * self.train_scale)

    @property
    def gan_rampup(self) -> int:
        return math.floor(self.ramp_scale * self.train_scale)

    @property
    def kld_cfg(self):
        return dict(n_warmup=self.kld_warmup, n_rampup=self.kld_rampup, w_min=0, w_max=self.kld_weight)

    @property
    def pips_cfg(self):
        return dict(n_warmup=self.pips_warmup, n_rampup=self.pips_rampup, w_min=0, w_max=self.pips_weight)

    @property
    def gan_cfg(self):
        return dict(n_warmup=self.gan_warmup, n_rampup=self.gan_rampup, w_min=0, w_max=self.gan_weight)

    def create_module(self):
        # dynamically import the module & class
        module_name, class_name = self.target.rsplit(".", 1)
        mod       = import_module(module_name)
        loss_cls  = getattr(mod, class_name)

        # instantiate with your validated kwargs
        return loss_cls(
            cfg=self,
        )
    def create_pips(self):

        # instantiate with your validated kwargs
        if self.use_pips_eval:
            from src.losses.loss_functions import EVALPIPSLOSS
            return EVALPIPSLOSS(
                cfg=self,
            )
        else:
            return None


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MetricLoss(BasicLoss):
    # override
    target: Literal["NT-Xent", "Triplet"] = "NT-Xent"

    # metric-specific
    schedule_metric: bool = True
    metric_warmup: int = 50
    metric_rampup: int = 20

    # model arch info
    frac_nuisance_latents: float = 0.2
    latent_dim: Optional[int] = None
    metric_array: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    # n_latents: Optional[int] = None

    # metric learning
    temperature: float = 0.1 # sets sharpness of loss 'gradient'
    metric_weight: float = 1.0  # tunes weight of contastive loss within the loss function
    margin: float = 1.0  #  sets tolerance/scale for metric loss.
    distance_metric: Literal["euclidean"] = "euclidean" # Could/should add cosine

    # params to structure interactions
    time_window: float = 1.5  # max permitted age difference between sequential pairs
    self_target_prob: float = 0.5  # fraction of time to load self-pair vs. alternative comparison

    # apply KLD reg to bio latents only?
    bio_only_kld: bool = False

    @property
    def metric_cfg(self):
        return dict(n_warmup=self.metric_warmup, n_rampup=self.metric_rampup, w_min=0, w_max=self.metric_weight)

    @property
    def pips_cfg(self):
        return dict(n_warmup=self.pips_warmup, n_rampup=self.pips_rampup, w_min=0, w_max=self.pips_weight)

    @property
    def kld_cfg(self):
        return dict(n_warmup=self.kld_warmup, n_rampup=self.kld_rampup, w_min=0, w_max=self.kld_weight)

    @property
    def latent_dim_bio(self) -> int:
        # at least 1, rounding up the nuisance count
        bio = self.latent_dim - math.ceil(self.frac_nuisance_latents * self.latent_dim)
        return max(bio, 1)

    @property
    def latent_dim_nuisance(self) -> int:
        return self.latent_dim - self.latent_dim_bio

    @property
    def biological_indices(self):
        return torch.arange(self.latent_dim_nuisance, self.latent_dim, dtype=torch.int64)

    @property
    def nuisance_indices(self):
        return torch.arange(0, self.latent_dim_nuisance, dtype=torch.int64)

    def create_module(self):
        # import as needed to avoid circularity
        from src.losses.loss_functions import NTXentLoss
        # map names→classes/functions
        loss_map = {
            "NT-Xent": NTXentLoss,
            # "Triplet": Triplet
            # "OtherDataset": OtherDataset,
        }

        loss_cls = loss_map[self.target]

        # instantiate your dataset with both fixed and configurable args
        return loss_cls(
            cfg=self
        )