from dataclasses import  field, asdict
from pydantic import BaseModel
from pydantic.dataclasses import dataclass # as pydantic_dataclass
from typing import Any, Dict, Optional, Literal, Union, Tuple
from src.models.model_utils import deep_merge, prune_empty
from omegaconf import OmegaConf, DictConfig
# from src.losses.legacy_loss_functions import VAELossBasic
from src.losses.loss_configs import BasicLoss, MetricLoss
from src.data.dataset_configs import BaseDataConfig, NTXentDataConfig
from src.models.model_components.arch_configs import (LegacyArchitecture, ArchitectureAELDM, TimmArchitecture)
from src.lightning.train_config import LitTrainConfig



ARCH_REGISTRY: dict[str, type] = {"convVAE": LegacyArchitecture,
                                  "ldmVAE": LegacyArchitecture,
                                    "Efficient-B0-RA": TimmArchitecture,   # E-B0-RA
                                    "Efficient-B4": TimmArchitecture,           # E-B4
                                    "ConvNeXt-Tiny": TimmArchitecture,
                                    "Swin-Tiny": TimmArchitecture,
                                    "MaxViT-Tiny": TimmArchitecture,
                                    "DeiT-Tiny": TimmArchitecture,  # alt
                                    "RegNet-Y": TimmArchitecture,
                                    "ViT-Tiny": TimmArchitecture,
                                    "Swin-Large": TimmArchitecture,
                                    "MaxViT-Small": TimmArchitecture,
                                    "Vit-Large": TimmArchitecture}


def resolve_arch(dd: dict):
    """
    dd is assumed to be a dict like {"type": "legacy", "latent_dim": 64, ...}
    """
    typ = dd.get("name")
    if typ not in ARCH_REGISTRY:
        raise KeyError(f"Unknown arch name {typ!r}; must be one of {list(ARCH_REGISTRY)}")
    cls = ARCH_REGISTRY[typ]
    kwargs = {k: v for k, v in dd.items() if k != "name"}
    return cls(**kwargs)

@dataclass
class VAEConfig:
    """VAE config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
    """
    ddconfig: Any = field(default_factory=LegacyArchitecture)
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

        # 2) instantiate ddconfig first
        dd_raw = user_model.pop("ddconfig", {})
        arch = resolve_arch(dd_raw) if dd_raw else cls().ddconfig

        # 3) build base dict and bake in the instantiated arch
        base = asdict(cls())
        base["ddconfig"] = arch

        # 3) deep‐merge user values on top of base
        merged = deep_merge(base, user_model)
        merged["lossconfig"]["max_epochs"] = merged["trainconfig"]["max_epochs"]
        # 4) re‐instantiate & validate in one shot
        inst = cls(**merged)
        inst.lossconfig.max_epochs = inst.trainconfig.max_epochs
        inst.lossconfig.input_dim = inst.ddconfig.input_dim

        return inst


@dataclass
class morphVAEConfig:

    ddconfig: Any = field(default_factory=LegacyArchitecture) # Done
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

        # 2) instantiate ddconfig first
        dd_raw = user_model.pop("ddconfig", {})
        arch = resolve_arch(dd_raw) if dd_raw else cls().ddconfig

        # 3) build base dict and bake in the instantiated arch
        base = asdict(cls())
        base["ddconfig"] = arch

        # 3) deep‐merge user values on top of base
        merged = deep_merge(base, user_model)

        # 4) re‐instantiate & validate in one shot
        inst = cls(**merged)

        # 5) update loss with necessary config options
        inst.lossconfig.max_epochs = inst.trainconfig.max_epochs
        inst.lossconfig.latent_dim = inst.ddconfig.latent_dim
        inst.lossconfig.input_dim = inst.ddconfig.input_dim

        # loss -> data
        inst.dataconfig.self_target_prob = inst.lossconfig.self_target_prob
        inst.dataconfig.time_window = inst.lossconfig.time_window

        # data -> loss
        # inst.lossconfig.metric_array = inst.dataconfig.metric_array

        return inst
    
    

# class morphVAEFancyConfig(morphVAEConfig):
#
#     ddconfig: SplitArchitectureAELDM = field(default_factory=SplitArchitectureAELDM)
#
#     name: Literal["morphVAEFancy"] = "morphVAEFancy"