from src.functions.dataset_utils import make_dynamic_rs_transform, ContrastiveLearningDataset, SeqPairDatasetCached,  TripletDatasetCached, DatasetCached
import os
from glob2 import glob
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
import importlib
from src.models.factories import build_from_config
from src.lightning.pl_wrappers import LitModel
import pytorch_lightning as pl
from src.lightning.train_config import LitTrainConfig


#########################
# define helper functions
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

def train_vae(train_data_path, cfg):

    if isinstance(cfg, str):
        # load config file
        config = OmegaConf.load(cfg)
    elif isinstance(cfg, dict):
        config = cfg
    else:
        raise Exception("cfg argument dtype is not recognized")

    config_full = config.copy()
    # get different components
    model_cfg = config.pop("model", OmegaConf.create())

    # # initialize training config
    # train_config = LitTrainConfig()
    # train_config = train_config.from_cfg(cfg=config_full)

    # instantiate config for specified model type
    target = model_cfg["config_target"]
    model_config = get_obj_from_str(target)
    model_config = model_config.from_cfg(cfg=config_full)

    # parse dataset related options and merge with defaults as needed
    data_config = model_config.dataconfig
    # get train/test/eval indices
    data_config.split_train_test()

    # initialize model
    model = build_from_config(model_config)
    loss_fn = model_config.lossconfig.create_module() # or model.compute_loss

    train_config = model_config.trainconfig
    # 2) wrap it
    lit = LitModel(
        model=model,
        loss_fn=loss_fn,
        data_cfg=data_config,
        lr=train_config.learning_rate,
        batch_key="data",
    )

    # make output directory
    run_name = f"{model_config.name}_z{model_config.ddconfig.latent_dim:02}"
    save_dir = os.path.join(data_config.root, "output", "")

    # 3) create your logger with a human‐readable version label
    logger = TensorBoardLogger(
        save_dir=save_dir,  # top-level folder
        name=run_name,  # e.g. "VAE_ld64"
        version=f"e{train_config.max_epochs}"  # e.g. "e50"
    )

    # 3) train with Lightning
    trainer = pl.Trainer(logger=logger, accelerator="gpu", devices=1, max_epochs=train_config.max_epochs)
    trainer.fit(lit)

    return {}


if __name__ == "__main__":
    data_path = "/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/training_data/20241107_ds/"
    # cfg = "/home/nick/projects/morphseq/src/vae/vae_base_config.yaml"
    cfg = "/home/nick/projects/morphseq/src/config_files/vae_test_run.yaml"
    train_vae(train_data_path=data_path, cfg=cfg)

    # config = OmegaConf.load(cfg)
    #
    # lightning_config = config.pop("lightning", OmegaConf.create())
    # # merge trainer cli with config
    # trainer_config = lightning_config.get("trainer", OmegaConf.create())
    #
    #
    # model = instantiate_from_config(config.model)
    #
    # root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    # train_folder = "20241021_ds"
    #
    # latent_dim = 100
    # n_conv_layers = 5
    # batch_size = 256
    # n_epochs = 25
    # # input_dim = (1, 576, 256)
    # input_dim = (1, 288, 128)
    # learning_rate = 1e-4
    # cache_data = True
    # model_type = "SeqVAE"
    #
    # #####################
    # # (3B & 4B) Seq-VAE (triplet)
    # #####################
    # train_suffix = "seq_test"
    # margin = 15  # , 50]
    # metric_weight = 10  # 50, 2, 1000] #[0.0001, 0.001, 0.01, 0.1, 1]
    # distance_metric = "euclidean"
    # self_target_prob = 0.5
    # metric_loss_type = "triplet"
    # n_workers = 2
    #
    # train_vae(root, train_folder, train_suffix=train_suffix, model_type=model_type,
    #           latent_dim=latent_dim, batch_size=batch_size, input_dim=input_dim,
    #           n_preload_workers=n_workers, n_load_workers=n_workers, metric_loss_type=metric_loss_type,
    #           distance_metric=distance_metric, n_epochs=n_epochs, margin=margin, metric_weight=metric_weight,
    #           learning_rate=learning_rate, n_conv_layers=n_conv_layers, self_target_prob=self_target_prob,
    #           cache_data=cache_data)



