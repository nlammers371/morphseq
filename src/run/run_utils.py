import os
from dataclasses import dataclass, field
from typing import Any, Dict, Callable, Type
from src.data.DataTransforms import make_dynamic_rs_transform
from src.data.DatasetClasses import BasicDataset
import importlib
from torch.utils.data import Dataset

def deep_merge(default: dict, override: dict) -> dict:
    """
    Recursively merge override into default and return a new dict.
    Nested dicts get merged rather than overwritten wholesale.
    """
    out = default.copy()
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def import_from_string(path: str) -> Any:
    """
    Given a string like "mypackage.submod.ClassName",
    import the module and return the attribute.
    """
    module_path, name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, name)

@dataclass
class DataConfig:
    # code‐first defaults:
    batch_size:    int            = 64
    num_workers:   int            = 4
    shuffle:       bool           = True
    wrap:          bool           = True

    # default Dataset *class*
    target:   Type[Dataset]  = BasicDataset

    # default transform factory
    transform:     Callable[...,Any] = make_dynamic_rs_transform

    # catch‐all for extra args to dataset/__init__
    params: Dict[str,Any] = field(default_factory=dict)

    @classmethod
    def from_cfg(cls, data_cfg: Dict[str,Any]) -> "DataConfig":
        inst = cls()  # 1) apply code‐first defaults

        # 2) pull out user overrides
        for key, override in data_cfg.items():
            if not hasattr(inst, key):
                continue

            default = getattr(inst, key)

            # 3a) special‐case the class pointer
            if key == "target" and isinstance(override, str):
                inst.dataset_cls = import_from_string(override)

            # 3b) likewise for transform if you want strings there
            elif key == "transform" and isinstance(override, str):
                inst.transform = import_from_string(override)

            # 3c) deep‐merge any dict defaults
            elif isinstance(default, dict) and isinstance(override, dict):
                inst.dataset_kwargs = deep_merge(default, override)

            # 3d) everything else just overwrites
            else:
                setattr(inst, key, override)

        return inst

@dataclass
class ObjectiveConfig:
    loss_fn:       str    = "vae_basic"
    loss_kwargs:   Dict   = field(default_factory=dict)

@dataclass
class ModelConfig:
    architecture:  str                   = "VAE"
    model_kwargs:  Dict[str, Any]        = field(default_factory=dict)
    data:          DataConfig            = field(default_factory=DataConfig)
    objective:     ObjectiveConfig       = field(default_factory=ObjectiveConfig)


# 2) A function that patches in the architecture‑specific defaults
def parse_dataset_options(model_config, data_config):

    if model_config.name == "VAE": # use basic sampler for vanuilla vae
        # initialize default data config class
        data_cfg = DataConfig(
            # batch_size=128,
            # num_workers=8,
            target=BasicDataset,
            batch_size=128,
            num_workers=4,
            shuffle=True,
            wrap=True,
            transform=make_dynamic_rs_transform,
        )

        data_cfg.from_cfg(data_config)
    else:
        raise NotImplementedError

    return data_cfg

def apply_arch_defaults(cfg: ModelConfig) -> ModelConfig:
    if cfg.architecture == "VAE":
        # override only what differs from the generic defaults
        cfg.data = DataConfig(
            batch_size=128,
            num_workers=8,
            dataset_cls="BasicDataset",
            dataset_kwargs={"augment": True}
        )
        cfg.objective = ObjectiveConfig(
            loss_fn="vae_basic"
        )
        cfg.model_kwargs = {"latent_dim": 32, "n_conv_layers": 4}

    elif cfg.architecture == "Flow":
        cfg.data = DataConfig(
            batch_size=32,
            num_workers=2,
            dataset_cls="FlowDataset"
        )
        cfg.objective = ObjectiveConfig(
            loss_fn="nll",
            loss_kwargs={}
        )
        cfg.model_kwargs = {"n_flows": 16}
    # …add other architectures here…
    return cfg

def apply_arch_defaults(cfg: ModelConfig) -> ModelConfig:
    if cfg.architecture == "VAE":
        # override only what differs from the generic defaults
        cfg.data = DataConfig(
            batch_size=128,
            num_workers=8,
            dataset_cls="VaeDataset",
            dataset_kwargs={"augment": True}
        )
        cfg.objective = ObjectiveConfig(
            loss_fn="bce",
            loss_kwargs={"reduction": "sum"}
        )
        cfg.model_kwargs = {"latent_dim": 32, "n_conv_layers": 4}

    elif False: #cfg.architecture == "Flow":
        print("Do something else")

    # merge: arch_defaults <- user_overrides  (user wins)
    merged_data = {**arch_data, **user_data}
    merged_obj = {**arch_obj, **user_obj}
    merged_model_kw = {**arch_model_kw, **user_model_kw}

    # build new nested configs
    cfg.data = DataConfig(**merged_data)
    cfg.objective = ObjectiveConfig(**merged_obj)
    cfg.model_kwargs = merged_model_kw

    return cfg

    # …add other architectures here…
    return cfg

def handle_training_options(cfg):
    model_name = cfg["model"]["name"]

    # first, parse loss type options


    # next, set dataloader
    if model_name == "VAE":
        # Make datasets
        train_dataset = DatasetCached(root=os.path.join(train_dir, "images"),
                                      transform=data_transform,
                                      training_config=train_config,
                                      return_name=True)

    elif (model_type == "SeqVAE") & (model_config.metric_loss_type == "NT-Xent"):
        # Make datasets
        train_dataset = SeqPairDatasetCached(root=os.path.join(train_dir, "images"),
                                             model_config=model_config,
                                             train_config=train_config,
                                             transform=data_transform,
                                             return_name=True
                                             )

    elif (model_type == "SeqVAE") & (model_config.metric_loss_type == "triplet"):
        # Make datasets
        train_dataset = TripletDatasetCached(root=os.path.join(train_dir, "images"),
                                             model_config=model_config,
                                             train_config=train_config,
                                             transform=data_transform,
                                             return_name=True
                                             )