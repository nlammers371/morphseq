from dataclasses import  field, fields
# from ..base.base_config import BaseAEConfig
from pydantic import BaseModel
from pydantic.dataclasses import dataclass # as pydantic_dataclass
from typing import Any, Dict, Optional, Literal
from src.run.run_utils import deep_merge, LossOptions
from omegaconf import OmegaConf

@dataclass
class VAEConfig:
    """VAE config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
    """
    name: str = "VAE"
    ddconfig: Dict[str, Any] = field(default_factory=
                                lambda: { "latent_dim": 64,
                                          "input_dim": (1, 288, 128),
                                          "n_channels_out": 16,
                                          "n_conv_layers": 5,
                                })
    objective: Literal['vae_loss_basic'] = 'vae_loss_basic'
    base_learning_rate: float = 1e-4

    @classmethod
    def from_cfg(cls, cfg):

        # pull out model-specific params
        model_cfg = cfg.pop("model", OmegaConf.create())

        # filter only the fields we know about
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        clean = {k: v for k, v in model_cfg.items() if k in valid}

        # pull out loss-related params
        loss_cfg = cfg.pop("objective", OmegaConf.create())
        clean["objective"] = loss_cfg["target"]

        # get defaults
        defaults = {f.name: getattr(cls(), f.name) for f in fields(cls)}

        data = {}
        for k, default in defaults.items():
            if k in clean.keys():
                override = clean[k]
                # If it's a dict‐default, do a deep merge
                if isinstance(default, dict) and isinstance(override, dict):
                    merged = deep_merge(default, override)
                    data[k] = merged
                else:
                    data[k] = override
            else:
                data[k] = default

        # now validate Literals, etc.
        inst = cls(**data)

        return inst

