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
from hydra.core.hydra_config import HydraConfig
torch.set_float32_matmul_precision("medium")   # good default

# Option B: match by message regex (if you want to be extra precise)
warnings.filterwarnings(
    "ignore",
    message=r".*recommended to use `self\.log\('val/.*',.*sync_dist=True`.*"
)

def load_from_checkpoint(model, ckpt_path):

    if ckpt_path is not None:
        print("Loading checkpoint from {}".format(ckpt_path))

        ckpt = torch.load(
            ckpt_path,
            map_location="cpu")["state_dict"]

        # Fix encoder sizing
        w3 = ckpt["encoder.conv_in.weight"]  # shape [128, 3, 3, 3]
        w1 = w3.mean(dim=1, keepdim=True)  # now [128, 1, 3, 3]
        ckpt["encoder.conv_in.weight"] = w1

        # Fix decoder sizing
        w3 = ckpt["decoder.conv_out.weight"]
        w1 = w3.mean(axis=0, keepdim=True)
        ckpt["decoder.conv_out.weight"] = w1

        w3 = ckpt["decoder.conv_out.bias"]
        w1 = w3.mean(dim=0, keepdim=True)
        ckpt["decoder.conv_out.bias"] = w1

        enc_sd = {k.replace("encoder.", "enc."): v
                  for k, v in ckpt.items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc_sd, strict=False)

        dec_sd = {k: v
                  for k, v in ckpt.items() if k.startswith("decoder.")}
        model.decoder.load_state_dict(dec_sd, strict=False)
    else:
        print("No checkpoint found. Skipping ckpt load.")

    return model

def train_vae(cfg, gpus: int | None = None):

    if isinstance(cfg, str):
        # load config file
        config = OmegaConf.load(cfg)
    elif isinstance(cfg, dict):
        config = cfg
    else:
        raise Exception("cfg argument dtype is not recognized")

    model, model_config, data_config, loss_fn, train_config = initialize_model(config)

    if hasattr(model_config, "ckpt_path"):
        model = load_from_checkpoint(model=model, ckpt_path=model_config.ckpt_path)

    # 2) wrap it
    lit = LitModel(
        model=model,
        loss_fn=loss_fn,
        data_cfg=data_config,
        lr=train_config.learning_rate,
        batch_key="data",
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}   "
          f"({100 * trainable_params / total_params:.2f}% of total)")
    # make output directory
    # run_name = f"{model_config.name}_z{model_config.ddconfig.latent_dim:02}_e{train_config.max_epochs}_b{int(100*loss_fn.kld_weight)}_percep"
    # save_dir = os.path.join(data_config.root, "output", ""
    if HydraConfig.initialized():
        # we’re inside a Hydra job
        out_dir = HydraConfig.get().runtime.output_dir
    else:
        run_name = f"{model_config.name}_z{model_config.ddconfig.latent_dim:02}_e{train_config.max_epochs}_b{int(100 * loss_fn.kld_weight)}_percep"
        out_dir = os.path.join(data_config.root, "output", run_name, "")

    # 3) create your logger with a human‐readable version label
    logger = TensorBoardLogger(
        save_dir=out_dir,  # top-level folder
        # name=run_name,  # e.g. "VAE_ld64"
        # version=f"  # e.g. "e50"
    )

    # device_kwargs = pick_devices(gpus)
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = "auto"  # all GPUs
        strategy = "ddp"  # NCCL under the hood
    else:
        accelerator = "cpu"
        devices = 1
        strategy = "ddp_cpu"  # uses Gloo on CPU

    # 3) train with Lightning
    trainer = pl.Trainer(logger=logger,
                         max_epochs=train_config.max_epochs,
                         precision=16,
                         callbacks=[SaveRunMetadata(data_config)],
                         accelerator=accelerator,  # will pick 'gpu' if any GPUs are visible, else 'cpu'
                         devices=devices,  # will use all available GPUs or 1 CPU
                         strategy=strategy,)           # ← accelerator / devices injected here)
    trainer.fit(lit)

    return {}


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
