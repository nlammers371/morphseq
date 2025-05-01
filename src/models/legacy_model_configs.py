from dataclasses import  field, asdict
from pydantic import BaseModel
from pydantic.dataclasses import dataclass # as pydantic_dataclass
from typing import Any, Dict, Optional, Literal, Union, Tuple
from src.models.model_utils import deep_merge, prune_empty
from omegaconf import OmegaConf, DictConfig
# from src.losses.legacy_loss_functions import VAELossBasic
from src.losses.loss_configs import BasicLoss, MetricLoss
from src.data.dataset_configs import BaseDataConfig, NTXentDataConfig
from src.models.model_components.arch_configs import (LegacyArchitecture, SplitArchitecture,
                                                      ArchitectureAELDM, SplitArchitectureAELDM)
from src.lightning.train_config import LitTrainConfig


ARCH_REGISTRY: dict[str, type] = {"convVAE": LegacyArchitecture,
                         "convVAESplit": SplitArchitecture,
                         "ldmVAE": LegacyArchitecture,
                         "ldmVAESplit": SplitArchitecture,
                         }

def register_arch(name: str):
    def _inner(cls):
        ARCH_REGISTRY[name] = cls
        return cls
    return _inner

@dataclass
class VAEConfig:
    """VAE config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
    """
    ddconfig: LegacyArchitecture = field(default_factory=LegacyArchitecture)
    lossconfig: BasicLoss = field(default_factory=BasicLoss)
    dataconfig: BaseDataConfig = field(default_factory=BaseDataConfig)
    trainconfig: LitTrainConfig = field(default_factory=LitTrainConfig)

    name: Literal["VAE"] = "VAE"
    # objective: Literal['vae_loss_basic'] = 'vae_loss_basic'
    # base_learning_rate: float = 1e-4

    @classmethod
    def from_cfg(cls, cfg):
        # 1) pull in the raw user dict (OmegaConf or plain dict)
        user_model = cfg.pop("model", {})
        if isinstance(user_model, DictConfig):
            user_model = OmegaConf.to_container(user_model, resolve=True)
        user_model = prune_empty(user_model)

        # 2) build a default instance and dump it to a plain dict
        default_inst = cls()
        base = asdict(default_inst)

        # 3) deep‐merge user values on top of base
        merged = deep_merge(base, user_model)

        # 4) re‐instantiate & validate in one shot
        return cls(**merged)


@dataclass
class VAEFancyConfig:
    """VAE config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
    """
    ddconfig: ArchitectureAELDM = field(default_factory=ArchitectureAELDM)
    lossconfig: BasicLoss = field(default_factory=BasicLoss)
    dataconfig: BaseDataConfig = field(default_factory=BaseDataConfig)
    trainconfig: LitTrainConfig = field(default_factory=LitTrainConfig)

    name: Literal["VAEFancy"] = "VAEFancy"

    @classmethod
    def from_cfg(cls, cfg):
        # 1) pull in the raw user dict (OmegaConf or plain dict)
        user_model = cfg.pop("model", {})
        if isinstance(user_model, DictConfig):
            user_model = OmegaConf.to_container(user_model, resolve=True)
        user_model = prune_empty(user_model)

        # 2) build a default instance and dump it to a plain dict
        default_inst = cls()
        base = asdict(default_inst)

        # 3) deep‐merge user values on top of base
        merged = deep_merge(base, user_model)

        # 4) re‐instantiate & validate in one shot
        # add transform options
        inst = cls(**merged)
        inst.dataconfig.transform_kwargs["target_size"] = tuple([inst.ddconfig.resolution, inst.ddconfig.resolution])

        return inst


@dataclass
class morphVAEConfig:

    ddconfig: SplitArchitecture = field(default_factory=SplitArchitecture) # Done
    lossconfig: MetricLoss = field(default_factory=MetricLoss)
    dataconfig: NTXentDataConfig = field(default_factory=NTXentDataConfig)
    trainconfig: LitTrainConfig = field(default_factory=LitTrainConfig)

    name: Literal["morphVAE"] = "morphVAE"
    # objective: Literal['vae_loss_basic'] = 'vae_loss_basic'
    # base_learning_rate: float = 1e-4

    @classmethod
    def from_cfg(cls, cfg):
        # 1) pull in the raw user dict (OmegaConf or plain dict)
        user_model = cfg.pop("model", {})
        if isinstance(user_model, DictConfig):
            user_model = OmegaConf.to_container(user_model, resolve=True)
        user_model = prune_empty(user_model)

        # 2) build a default instance and dump it to a plain dict
        default_inst = cls()
        base = asdict(default_inst)

        # 3) deep‐merge user values on top of base
        merged = deep_merge(base, user_model)

        # 4) re‐instantiate & validate in one shot
        inst = cls(**merged)

        # 5) update loss with necessary config options
        inst.lossconfig.latent_dim = inst.ddconfig.latent_dim
        inst.lossconfig.latent_dim_bio = inst.ddconfig.latent_dim_bio
        inst.lossconfig.latent_dim_nuisance = inst.ddconfig.latent_dim_nuisance

        # loss -> data
        inst.dataconfig.self_target_prob = inst.lossconfig.self_target_prob
        inst.dataconfig.time_window = inst.lossconfig.time_window

        # data -> loss
        # inst.lossconfig.metric_array = inst.dataconfig.metric_array

        return inst

class morphVAEFancyConfig(morphVAEConfig):

    ddconfig: SplitArchitectureAELDM = field(default_factory=SplitArchitectureAELDM)

    name: Literal["morphVAEFancy"] = "morphVAEFancy"