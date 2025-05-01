from src.functions.dataset_utils import make_dynamic_rs_transform, ContrastiveLearningDataset, SeqPairDatasetCached,  TripletDatasetCached, DatasetCached
import os
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from src.lightning.pl_wrappers import LitModel
import pytorch_lightning as pl
from src.run.run_utils import initialize_model, initialize_ldm_model ,pick_devices
from src.lightning.callbacks import SaveRunMetadata
import torch

torch.set_float32_matmul_precision("medium")   # good default

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

@hydra.main(version_base="1.1",
            config_path="/home/nick/projects/morphseq/src/hydra_configs",
            config_name="vae_base")

def main(cfg: DictConfig):
    # immediately turn it into a plain dict:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    train_vae(cfg_dict)


if __name__ == "__main__":
    main()



