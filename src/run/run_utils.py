import importlib
from omegaconf import OmegaConf
from src.models.factories import build_from_config
from glob2 import glob
import os
import torch, warnings
import os
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from src.lightning.pl_wrappers import LitModel
import pytorch_lightning as pl
from src.lightning.callbacks import SaveRunMetadata
import torch


def train_vae(cfg, gpus: int | None = None):

    if isinstance(cfg, str):
        # load config file
        config = OmegaConf.load(cfg)
    elif isinstance(cfg, dict):
        config = cfg
    else:
        raise Exception("cfg argument dtype is not recognized")

    model, model_config, data_config, loss_fn, train_config = initialize_model(config)

    # 2) wrap it
    lit = LitModel(
        model=model,
        loss_fn=loss_fn,
        data_cfg=data_config,
        lr=train_config.learning_rate,
        batch_key="data",
    )

    # make output directory
    run_name = f"{model_config.name}_z{model_config.ddconfig.latent_dim:02}_e{train_config.max_epochs}_b{int(100*loss_fn.kld_weight)}_percep"
    save_dir = os.path.join(data_config.root, "output", "")

    # 3) create your logger with a human‐readable version label
    logger = TensorBoardLogger(
        save_dir=save_dir,  # top-level folder
        name=run_name,  # e.g. "VAE_ld64"
        # version=f"  # e.g. "e50"
    )

    device_kwargs = pick_devices(gpus)

    # 3) train with Lightning
    trainer = pl.Trainer(logger=logger,
                         max_epochs=train_config.max_epochs,
                         precision=16,
                         callbacks=[SaveRunMetadata(data_config)],
                         **device_kwargs)           # ← accelerator / devices injected here)
    trainer.fit(lit)

    return {}


def ramp_weight(
    step_curr: int,
    *,
    n_warmup: int,
    n_rampup: int,
    w_min: float = 0.0,
    w_max: float = 1.0,
) -> float:
    """
    Piece-wise schedule            (steps)
        • warm-up   : 0 … n_warmup-1      → w_min
        • ramp-up   : n_warmup … n_warmup+n_rampup-1
                      linear   w_min → w_max
        • plateau   : ≥ n_warmup+n_rampup → w_max
    """
    if step_curr < n_warmup:                     # ─── warm-up
        return w_min

    ramp_step = step_curr - n_warmup
    if ramp_step < n_rampup:                     # ─── linear ramp
        progress = ramp_step / max(n_rampup - 1, 1)
        return w_min + progress * (w_max - w_min)

    return w_max

# ----------------------------------------------------------------------
#  Utility: choose accelerator + devices
# ----------------------------------------------------------------------
def pick_devices(requested: int | None = None) -> dict:
    """
    Return a dict of kwargs for `Trainer(...)`:
        {"accelerator": "gpu", "devices": <n>}  or {"accelerator": "cpu"}
    *requested* overrides the auto-detected GPU count.
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if requested is None:
            n = gpu_count               # use *all* GPUs
        else:
            n = min(requested, gpu_count)
            if requested > gpu_count:
                warnings.warn(f"Asked for {requested} GPUs but only "
                              f"{gpu_count} available; using {gpu_count}.",
                              stacklevel=2)
        return {"accelerator": "gpu", "devices": n}

    # ----- CPU fallback -----
    if requested not in (None, 0):
        warnings.warn("No CUDA device found ‒ falling back to CPU.",
                      stacklevel=2)
    return {"accelerator": "cpu"}       # Lightning uses all CPU cores


def parse_model_paths(model_config, train_config, data_config, version, ckpt):
    run_name = f"{model_config.name}_z{model_config.ddconfig.latent_dim:02}_e{train_config.max_epochs}_b{int(100*model_config.lossconfig.kld_weight)}_percep"
    # get version
    if version is None:  # find most recent version
        all_versions = glob(os.path.join(data_config.root, "output", run_name, "*"))
        all_versions = sorted([v for v in all_versions if os.path.isdir(v)])
        model_dir = all_versions[-1]
    else:
        model_dir = os.path.join(data_config.root, "output", run_name, version, "")
    # get checkpoint
    ckpt_dir = os.path.join(model_dir, "checkpoints", "")
    # find all .ckpt files
    if ckpt is None:
        all_ckpts = glob(os.path.join(ckpt_dir, "*.ckpt"))
        # pick the one with the latest modification time
        latest_ckpt = max(all_ckpts, key=os.path.getmtime)
    else:
        latest_ckpt = ckpt_dir + ckpt + ".ckpt"

    return model_dir, latest_ckpt


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def initialize_model(config):
    # initialize the model
    config_full = config.copy()
    model_dict = config.pop("model", OmegaConf.create())
    target = model_dict["config_target"]
    model_config = get_obj_from_str(target)
    model_config = model_config.from_cfg(cfg=config_full)

    # parse dataset related options and merge with defaults as needed
    data_config = model_config.dataconfig
    data_config.make_metadata()

    # initialize model
    model = build_from_config(model_config)
    if hasattr(model_config.lossconfig, "metric_array"):
        model_config.lossconfig.metric_array = data_config.metric_array
    loss_fn = model_config.lossconfig.create_module()  # or model.compute_loss

    train_config = model_config.trainconfig

    return model, model_config, data_config, loss_fn, train_config


# def initialize_ldm_model(config):
#     # initialize the model
#     config_full = config.copy()
#     model_dict = config.pop("model", OmegaConf.create())
#     target = model_dict["config_target"]
#     model_config = get_obj_from_str(target)
#     model_config = model_config.from_cfg(cfg=config_full)
#
#     # parse dataset related options and merge with defaults as needed
#     # data_config = model_config.dataconfig
#     # # get train/test/eval indices
#     # data_config.make_metadata()
#
#     # initialize model
#     model = build_from_config(model_config)
#     if hasattr(model_config.lossconfig, "metric_array"):
#         model_config.lossconfig.metric_array = data_config.metric_array
#     loss_fn = model_config.lossconfig.create_module()  # or model.compute_loss
#
#     train_config = model_config.trainconfig
#
#     return model, model_config, data_config, loss_fn, train_config
