from src.functions.dataset_utils import make_dynamic_rs_transform, ContrastiveLearningDataset, SeqPairDatasetCached,  TripletDatasetCached, DatasetCached
import os
from glob2 import glob
from omegaconf import OmegaConf
from run_utils import parse_dataset_options
import importlib


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

    # get different components
    lightning_cfg = config.pop("lightning", OmegaConf.create())
    trainer_cfg = lightning_cfg.get("trainer", OmegaConf.create())
    model_cfg = config.pop("model", OmegaConf.create())
    data_cfg = config.pop("data", OmegaConf.create())

    # instantiate model config and model
    target = model_cfg["config_target"]
    model_config = get_obj_from_str(target)
    model_config = model_config.from_cfg(model_cfg=model_cfg)

    # parse dataset related options and merge with defaults as needed
    # initialize dataset
    data_config = parse_dataset_options(model_config, data_cfg)
    DataCls = data_config.target
    Dataset = DataCls(root=os.path.join(train_data_path, "images"), transform=data_config.transform)
    # ds = DatasetCls(**model_config.dataset_kwargs)

    # initialize loss function

    # initialize model
    print("check")
    # train_dir = os.path.join(root, "training_data", train_folder)
    #
    # # make output directory to save training results
    # if train_suffix == '':
    #     model_name = model_type + f'_z{model_config.latent_dim:02}_' + f'ne{n_epochs:03}'  # + f'gamma{int(model_config.gamme):04}_' + f'temp{int(model_config.temperature):04}'
    # else:
    #     model_name = model_type + f'_z{model_config.latent_dim:02}_' + f'ne{n_epochs:03}_' + train_suffix  # + f'gamma{int(model_config.gamma):04}_' + f'temp{int(model_config.temperature):04}'  + '_'
    # output_dir = os.path.join(train_dir, model_name)

    # initialize training configuration
    # train_config = BaseTrainerConfig(
    #     output_dir=output_dir,
    #     num_epochs=n_epochs,
    #     **training_args
    # )

    # get train and test indices
    train_idx = model_config.train_indices
    eval_idx = model_config.eval_indices

    train_config.train_indices = train_idx
    train_config.eval_indices = eval_idx





    # Initialize encoder and decoder
    encoder = Encoder_Conv_VAE(model_config)  # these are custom classes I wrote for this use case
    decoder = Decoder_Conv_VAE(encoder)

    # initialize model
    # if model_type == "MetricVAE":
    #     model = MetricVAE(
    #         model_config=model_config,
    #         encoder=encoder,
    #         decoder=decoder
    #     )
    if model_type == "VAE":
        model = VAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder
        )
    elif model_type == "SeqVAE":
        model = SeqVAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder
        )
    elif model_type == "MorphIAFVAE":
        model = MorphIAFVAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder
        )
    else:
        raise Exception("Unrecognized model type: " + model_type)

    # Initialize training pipeliene
    pipeline = TrainingPipeline(
        training_config=train_config,
        model=model
    )

    # inputs = next(iter(test_loader))

    # Run pipeline
    pipeline(
        train_data=train_dataset,  # here we use the custom train dataset
        eval_data=train_dataset  # same dataset, but we will use different image indices
    )

    return output_dir


if __name__ == "__main__":
    data_path = "/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/training_data/20241107_ds/"
    cfg = "/home/nick/projects/morphseq/src/vae/vae_base_config.yaml"
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



