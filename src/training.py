from src.functions.dataset_utils import make_dynamic_rs_transform, ContrastiveLearningDataset, SeqPairDatasetCached,  TripletDatasetCached, DatasetCached
import os
from glob2 import glob
from omegaconf import OmegaConf
from src.diffusion.ldm.util import instantiate_from_config


def train_vae(root, cfg):

    if isinstance(cfg, str):
        # load config file
        config = OmegaConf.load(cfg)
    elif isinstance(cfg, dict):
        config = cfg
    else:
        raise Exception("cfg argument dtype is not recognized")

    # get different components
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    model_config = config.pop("model", OmegaConf.create())
    data_config = config.pop("data", OmegaConf.create())

    # training_keys = ["batch_size", "learning_rate", "n_load_workers"]  # optional training config kywords
    # training_args = dict({})
    # model_args = dict({})
    # for key, value in kwargs.items():
    #     if key in training_keys:
    #         if key == "batch_size":
    #             training_args["per_device_train_batch_size"] = value
    #             training_args["per_device_eval_batch_size"] = value
    #         elif key == "n_load_workers":
    #             training_args["train_dataloader_num_workers"] = value
    #             training_args["eval_dataloader_num_workers"] = value
    #         elif key == "n_preload_workers":
    #             training_args["preload_dataloader_num_workers"] = value
    #         else:
    #             training_args[key] = value
    #     else:
    #         model_args[key] = value

    # if input_dim == None:
    #     input_dim = (1, 288, 128)

    train_dir = os.path.join(root, "training_data", train_folder)

    if model_type == "VAE":
        # load standard VAE config
        model_config = VAEConfig(
            input_dim=input_dim,
            data_root=root,
            train_folder=train_folder,
            **model_args
        )
        # generate train/test sample indices
        model_config.split_train_test()

        # Standard data transform
        if cache_data:
            data_transform = None
        else:
            data_transform = make_dynamic_rs_transform()

    elif model_type == "SeqVAE":
        # raise Error("Need to update dataloader architecture for seqVAE")
        # initialize model configuration
        model_config = SeqVAEConfig(
            input_dim=input_dim,
            data_root=root,
            train_folder=train_folder,
            **model_args
        )

        # initialize reference dataset
        print("Making lookup dictionary for sequential pairs...")
        model_config.make_dataset()

        # initialize contrastive data loader
        # if cache_data:
        #     data_transform = ContrastiveLearningDataset.get_contrastive_transform_cache()
        # else:
        data_transform = ContrastiveLearningDataset.get_simclr_pipeline_transform()

    elif model_type == "MorphIAFVAE":
        # raise Error("Need to update dataloader architecture for seqVAE")
        # initialize model configuration
        model_config = MorphIAFVAEConfig(
            input_dim=input_dim,
            data_root=root,
            train_folder=train_folder,
            **model_args
        )

        # initialize reference dataset
        print("Making lookup dictionary for sequential pairs...")
        model_config.make_dataset()

        # initialize contrastive data loader
        # if cache_data:
        #     data_transform = ContrastiveLearningDataset.get_contrastive_transform_cache()
        # else:
        data_transform = ContrastiveLearningDataset.get_simclr_pipeline_transform()

    else:
        raise Exception("Unrecognized model type: " + model_type)

    # make output directory to save training results
    if train_suffix == '':
        model_name = model_type + f'_z{model_config.latent_dim:02}_' + f'ne{n_epochs:03}'  # + f'gamma{int(model_config.gamme):04}_' + f'temp{int(model_config.temperature):04}'
    else:
        model_name = model_type + f'_z{model_config.latent_dim:02}_' + f'ne{n_epochs:03}_' + train_suffix  # + f'gamma{int(model_config.gamma):04}_' + f'temp{int(model_config.temperature):04}'  + '_'
    output_dir = os.path.join(train_dir, model_name)

    # initialize training configuration
    train_config = BaseTrainerConfig(
        output_dir=output_dir,
        num_epochs=n_epochs,
        **training_args
    )

    # get train and test indices
    train_idx = model_config.train_indices
    eval_idx = model_config.eval_indices

    train_config.train_indices = train_idx
    train_config.eval_indices = eval_idx

    if model_type == "VAE":
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

    elif (model_type == "MorphIAFVAE") & (model_config.metric_loss_type == "triplet"):
        # Make datasets
        train_dataset = TripletDatasetCached(root=os.path.join(train_dir, "images"),
                                             model_config=model_config,
                                             train_config=train_config,
                                             cache_data=cache_data,
                                             transform=data_transform,
                                             return_name=True
                                             )

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

    cfg = "/Users/nick/Projects/morphseq/src/vae/test_config.yaml"
    train_vae(root="", cfg=cfg)
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



