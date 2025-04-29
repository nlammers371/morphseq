from dataclasses import  field, asdict
from pydantic import BaseModel
from pydantic.dataclasses import dataclass # as pydantic_dataclass
from typing import Any, Dict, Optional, Literal, Union, Tuple
from src.models.model_utils import deep_merge, prune_empty
from omegaconf import OmegaConf, DictConfig
# from src.losses.legacy_loss_functions import VAELossBasic
from src.losses.loss_configs import BasicLoss, MetricLoss
from src.data.dataset_configs import BaseDataConfig, NTXentDataConfig
from src.models.model_components.arch_configs import SplitArchitectureAELDM, ArchitectureAELDM
from src.lightning.train_config import LitTrainConfig


@dataclass
class ldmAEConfig: # VAE equivalent--no metric component

    ddconfig: ArchitectureAELDM = field(default_factory=SplitArchitectureAELDM) # Done
    lossconfig: BasicLoss = field(default_factory=BasicLoss)
    dataconfig: BaseDataConfig = field(default_factory=BaseDataConfig)
    trainconfig: LitTrainConfig = field(default_factory=LitTrainConfig)

    name: Literal["ldmAEkl"] = "ldmAEkl"
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


        # data -> loss
        # inst.lossconfig.metric_array = inst.dataconfig.metric_array

        return inst