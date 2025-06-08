import importlib
from pytorch_lightning.strategies import DDPStrategy
from src.models.factories import build_from_config
from glob2 import glob
import wandb
import io
import torch, warnings
import os
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from src.lightning.pl_wrappers import LitModel
import pytorch_lightning as pl
from src.lightning.callbacks import SaveRunMetadata, EpochListCheckpoint
import torch
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import json
import yaml
import pandas as pd
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision("medium")
# good default

# Option B: match by message regex (if you want to be extra precise)
warnings.filterwarnings(
    "ignore",
    message=r".*recommended to use `self\.log\('val/.*',.*sync_dist=True`.*"
)

def update_rows(rows, mdl_path):
    cfg_path = os.path.join(mdl_path, ".hydra", "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            text = f.read()
        overrides = yaml.safe_load(text)
        # metrics = json.loads(metrics_path.read_text())
        # Use relative path from results_dir as run identifier
        run_id = os.path.basename(mdl_path)
        row = {"run": run_id}
        if "model" in overrides.keys():
            row.update(overrides["model"])
        else:
            row.update(overrides)

        row["mdl_path"] = mdl_path
        rows.append(row)

    return rows



def collect_results_recursive(
    results_dir: str,
    # output_csv: Union[str, Path],
    # column_order: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Recursively aggregates per-run overrides and metrics into a master CSV with enforced column order.

    Args:
        results_dir: Root directory containing nested run subfolders.
        output_csv: Path to write the aggregated CSV.
        column_order: List of column names to appear first in the DataFrame; any additional
                      columns will follow in alphabetical order.

    Returns:
        A pandas DataFrame of the aggregated results.
    """
    results_dir = Path(results_dir)
    # rows = []
    train_dir_list = sorted(glob(os.path.join(results_dir, "*")))
    train_dir_list = [td for td in train_dir_list if os.path.isdir(td)]

    rows = []
    # Search for all metrics.json files at any depth
    for mdl_path in train_dir_list:
        if os.path.isfile(os.path.join(mdl_path, "multirun.yaml")):
            sub_dir_list = sorted(glob(os.path.join(mdl_path, "*")))
            sub_dir_list = [td for td in sub_dir_list if os.path.isdir(td)]
            for sub_path in sub_dir_list:
                rows = update_rows(rows, sub_path)
        else:
            rows = update_rows(rows, mdl_path)

    df = pd.json_normalize(rows, sep="_")
    df.to_csv(os.path.join(results_dir, "job_summary_df.csv"), index=False)

    return df


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

def train_vae(cfg):

    if isinstance(cfg, str):
        # load config file
        config = OmegaConf.load(cfg)
    elif isinstance(cfg, dict):
        config = cfg
    else:
        raise Exception("cfg argument dtype is not recognized")
    full_config = config.copy()
    dummy_config = config.copy()
    model, model_config, data_config, loss_fn, pips_fn, train_config = initialize_model(config)
    dummy_model, _, _, _, _, _= initialize_model(dummy_config)

    if hasattr(model_config, "ckpt_path"):
        model = load_from_checkpoint(model=model, ckpt_path=model_config.ckpt_path)

    # 2) wrap it
    lit = LitModel(
        model=model,
        loss_fn=loss_fn,
        eval_gpu_flag=train_config.eval_gpu_flag,
        data_cfg=data_config,
        train_cfg=train_config,
        batch_key="data",
    )
    # make output directory
    if HydraConfig.initialized():
        # we’re inside a Hydra job
        out_dir = HydraConfig.get().runtime.output_dir
    else:
        run_name = f"{model_config.name}_z{model_config.ddconfig.latent_dim:02}_e{train_config.max_epochs}_b{int(100 * loss_fn.kld_weight)}_percep"
        out_dir = os.path.join(data_config.root, "output", run_name, "")

    # 3) create your logger with a human‐readable version label
    # — now swap in W&B:

    tb_logger = TensorBoardLogger(
        save_dir=out_dir,
        name="tensorboard"
    )

    wb_conf = config["wandb"]
    wandb_logger = WandbLogger(
        project=wb_conf["project"],
        entity=wb_conf.get("entity"),
        name=wb_conf.get("run_name"),
        config=full_config,
        offline=wb_conf.get("offline", False),
        save_dir=out_dir,
        sync_tensorboard=True
    )

    ckpt_path = os.path.join(wandb_logger.save_dir, "checkpoints")
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_path,  # same top‑level folder as your logger
        filename="epoch{epoch:02d}",  # e.g. epoch=05.ckpt
        # save_top_k=-1,  # save all checkpoints, not just the best
        # every_n_epochs=model_config.trainconfig.save_every_n,
        save_weights_only=True,
        save_last=True,  # also keep 'last.ckpt'
    )

    if hasattr(model_config.trainconfig, "save_epochs"):
        spec_ckpt_cb = [EpochListCheckpoint(
            epochs=model_config.trainconfig.save_epochs,
            dirpath=os.path.join(ckpt_path, "special")
        )]
    else:
        spec_ckpt_cb = []

    # device_kwargs = pick_devices(gpus)
    # if torch.cuda.is_available():
    #     accelerator = "gpu"
    #     devices = "auto"  # all GPUs
    #     strategy = "ddp"  # NCCL under the hood
    # else:
    #     accelerator = "cpu"
    #     devices = 1
    #     strategy = "ddp_cpu"  # uses Gloo on CPU

    # 3) train with Lightning
    trainer = pl.Trainer(logger=[wandb_logger, tb_logger],
                         max_epochs=train_config.max_epochs,
                         precision=16,
                         callbacks=[SaveRunMetadata(data_config), checkpoint_cb] + spec_ckpt_cb,
                         accelerator="gpu",
                         log_every_n_steps=10,
                         strategy=DDPStrategy(find_unused_parameters=True),
                         devices="auto",
                         )

    # wandb_run = trainer.logger.experiment
    # wandb_logger.experiment.watch(
    #     lit.model, log="all"
    # )
    # dummy_input = torch.zeros((1, *model_config.ddconfig.input_dim))

    # torch.onnx.export(
    #     model,  # underlying PyTorch model
    #     dummy_input,  # representative input tensor
    #     "model.onnx",  # output file
    #     export_params=True,
    #     opset_version=12,
    #     input_names=["input"],
    #     output_names=["output"],
    #     dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    # )

    # artifact = wandb.Artifact("model-architecture", type="model")
    # artifact.add_file("model.onnx")
    # wandb_logger.experiment.log_artifact(artifact
    # run it!
    try:
        trainer.fit(lit)
        # tell logger to close
        wandb_logger.experiment.finish()
    except:
        wandb.finish()
        print("Erorr encountered during training. Skipping.")

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
    pips_fn = model_config.lossconfig.create_pips()

    train_config = model_config.trainconfig

    return model, model_config, data_config, loss_fn, pips_fn, train_config


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
