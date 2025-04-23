from dataclasses import  field, asdict
from pydantic import BaseModel
from pydantic.dataclasses import dataclass # as pydantic_dataclass
from typing import Any, Dict, Optional, Literal, Union, Tuple
from src.run.run_utils import deep_merge, LossOptions
from omegaconf import OmegaConf, DictConfig
# from src.losses.legacy_loss_functions import VAELossBasic
from src.losses.loss_configs import BasicLoss
from src.data.dataset_configs import BaseDataConfig

@dataclass
class LegacyArchitecture:
    latent_dim: int = 64
    n_channels_out: int = 16
    n_conv_layers: int = 5
    input_dim: Tuple[int, int, int] = (1, 288, 128)


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
    name: str = "VAE"
    objective: Literal['vae_loss_basic'] = 'vae_loss_basic'
    base_learning_rate: float = 1e-4

    @classmethod
    def from_cfg(cls, cfg):
        # 1) pull in the raw user dict (OmegaConf or plain dict)
        user_model = cfg.pop("model", {})
        if isinstance(user_model, DictConfig):
            user_model = OmegaConf.to_container(user_model, resolve=True)
        # 2) build a default instance and dump it to a plain dict
        default_inst = cls()
        base = asdict(default_inst)

        # 3) deep‐merge user values on top of base
        merged = deep_merge(base, user_model)

        # 4) re‐instantiate & validate in one shot
        return cls(**merged)

class SeqVAEConfig(BaseModel):
    model_type:      Literal["SeqVAE"] = "SeqVAE"
    # encoder:         EncoderConfig
    # decoder:         DecoderConfig
    # sequence_length: int = Field(..., gt=1)
    objective:       Literal["seq_vae_loss"] = "seq_vae_loss"

