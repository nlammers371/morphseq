from typing    import Literal, Optional
from pydantic.dataclasses import dataclass
from dataclasses import field
from importlib import import_module
import torch
import numpy as np
from pydantic import ConfigDict

@dataclass
class BasicLoss:
    target: Literal[
        "src.losses.legacy_loss_functions.VAELossBasic"
    ] = "src.losses.legacy_loss_functions.VAELossBasic"
    kld_weight: float = 1.0
    reconstruction_loss: str = "mse"

    def create_module(self):
        # dynamically import the module & class
        module_name, class_name = self.target.rsplit(".", 1)
        mod       = import_module(module_name)
        loss_cls  = getattr(mod, class_name)
        # instantiate with your validated kwargs
        return loss_cls(
            kld_weight=self.kld_weight,
            reconstruction_loss=self.reconstruction_loss,
        )


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MetricLoss:

    target: Literal["NT-Xent", "Triplet"] = "NT-Xent"

    # base characeristics for VAE
    kld_weight: float = 1.0
    reconstruction_loss: str = "mse"

    # model arch info
    latent_dim_bio: Optional[int] = None
    latent_dim_nuisance: Optional[int] = None
    metric_array: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    # n_latents: Optional[int] = None

    # metric learning
    temperature: float = 0.1 # sets sharpness of loss 'gradient'
    metric_weight: float = 1.0  # tunes weight of contastive loss within the loss function
    margin: float = 1.0  # (Triplet only) sets tolerance/scale for metric loss.
    distance_metric: Literal["euclidean"] = "euclidean" # Could/should add cosine

    # params to structure interactions
    time_window: float = 1.5  # max permitted age difference between sequential pairs
    self_target_prob: float = 0.5  # fraction of time to load self-pair vs. alternative comparison

    # apply KLD reg to bio latents only?
    bio_only_kld: bool = False

    @property
    def biological_indices(self):
        return torch.arange(self.latent_dim_nuisance, self.latent_dim, dtype=torch.int64)

    @property
    def nuisance_indices(self):
        return torch.arange(0, self.latent_dim_nuisance, dtype=torch.int64)


    def create_module(self):
        # import as needed to avoid circularity
        from src.losses.legacy_loss_functions import NTXentLoss
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