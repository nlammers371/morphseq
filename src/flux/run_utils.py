import importlib
from pytorch_lightning.strategies import DDPStrategy
from src.models.factories import build_from_config
from glob2 import glob
import wandb
import io
import torch, warnings
import os
from pytorch_lightning.loggers import TensorBoardLogger
# import hydra
from src.flux.lightning import ClockNVF
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision("medium")
# good default

# Option B: match by message regex (if you want to be extra precise)
warnings.filterwarnings(
    "ignore",
    message=r".*recommended to use `self\.log\('val/.*',.*sync_dist=True`.*"
)



def train_vector_nn(root: Path,
                    model_class: str = "legacy",
                    model_name: str = "20241107_ds_sweep01_optimum",
                    batch_size: int = 8192,):

    # if isinstance(cfg, str):
    #     # load config file
    #     config = OmegaConf.load(cfg)
    # elif isinstance(cfg, dict):
    #     config = cfg
    # else:
    #     raise Exception("cfg argument dtype is not recognized")


    # 1) Intialize dataloader 
    embryo_df = load_embryo_df(root, model_class, model_name)
    ds = build_traing_data(df=embryo_df,)

    loader = torch.utils.data.DataLoader(ds,
               batch_size=batch_size,
               shuffle=True,
               pin_memory=True)
    
    # calculate parameters from dataset
    dim = ds.z0.shape[1]  # latent dimension
    num_exp = ds.exp_idx.max().item() + 1  # number of experiments
    num_embryo = ds.emb_idx.max().item() + 1  # number of embryos

    # 2) intialize model
    lit = ClockNVF(
                dim=dim,
                num_exp=num_exp,
                num_embryo=num_embryo,
                infer_embryo_clock=infer_embryo_clock,
                hidden=mlp_structure)
   
    # make output directory
    # if HydraConfig.initialized():
    #     # we’re inside a Hydra job
    #     out_dir = HydraConfig.get().runtime.output_dir
    # else:
    #     run_name = f"{model_config.name}_z{model_config.ddconfig.latent_dim:02}_e{train_config.max_epochs}_b{int(100 * loss_fn.kld_weight)}_percep"
    #     out_dir = os.path.join(data_config.root, "output", run_name, "")

    # 3) create your logger with a human‐readable version label
    # — now swap in W&B:

    # wb_conf = config["wandb"]
    # wandb_logger = WandbLogger(
    #     project=wb_conf["project"],
    #     entity=wb_conf.get("entity"),
    #     name=wb_conf.get("run_name"),
    #     # config=full_config,
    #     offline=wb_conf.get("offline", False),
    #     save_dir=out_dir,
    #     sync_tensorboard=True
    # )

    ckpt_path = Path(wandb_logger.save_dir) / "checkpoints"
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_path,  # same top‑level folder as your logger
        filename="epoch{epoch:02d}",  # e.g. epoch=05.ckpt
        # save_top_k=-1,  # save all checkpoints, not just the best
        # every_n_epochs=model_config.trainconfig.save_every_n,
        save_weights_only=True,
        save_last=True,  # also keep 'last.ckpt'
    )

    # 3) train with Lightning
    trainer = pl.Trainer(logger=[wandb_logger, tb_logger],
                         max_epochs=max_epochs,
                         precision=16,
                         accelerator="gpu",
                         log_every_n_steps=10,
                         strategy=DDPStrategy(find_unused_parameters=True),
                         devices="auto",
                         )

    trainer.fit(lit)
    # tell logger to close
    wandb_logger.experiment.finish()

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




