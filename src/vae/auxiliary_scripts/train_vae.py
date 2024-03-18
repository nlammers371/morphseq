import sys
sys.path.append("/functions")
sys.path.append("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")
sys.path.append("E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\")
# sys.path.append("../src/")

from src.functions.dataset_utils import make_dynamic_rs_transform, MyCustomDataset, ContrastiveLearningDataset, \
    ContrastiveLearningViewGenerator, SeqPairDataset, SeqPairDatasetCached, TripletPairDataset, TripletDatasetCached, DatasetCached, grayscale_transform
import os
from src.vae.models import VAE, VAEConfig, MetricVAE, MetricVAEConfig, SeqVAEConfig, SeqVAE
from src.functions.custom_networks import Encoder_Conv_VAE, Decoder_Conv_VAE
from src.vae.trainers import BaseTrainerConfig
from src.vae.pipelines.training import TrainingPipeline
from torch.utils.data.sampler import SubsetRandomSampler

def train_vae(root, train_folder, n_epochs, model_type, input_dim=None, cache_data=True, train_suffix='', **kwargs):

    training_keys = ["batch_size", "learning_rate", "n_load_workers"] # optional training config kywords
    # model_keys = ["n_latent", "n_out_channels", "zn_frac", "depth", "nt_xent_temperature"]
    training_args = dict({})
    model_args = dict({})
    for key, value in kwargs.items():
        if key in training_keys:
            if key == "batch_size":
                training_args["per_device_train_batch_size"] = value
                training_args["per_device_eval_batch_size"] = value
            elif key == "n_load_workers":
                training_args["train_dataloader_num_workers"] = value
                training_args["eval_dataloader_num_workers"] = value
            elif key == "n_preload_workers":
                training_args["preload_dataloader_num_workers"] = value
            else:
                training_args[key] = value
        else:
            model_args[key] = value

    if input_dim == None:
        input_dim = (1, 288, 128)

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
        if cache_data:
            data_transform = ContrastiveLearningDataset.get_contrastive_transform_cache()
        else:
            data_transform = ContrastiveLearningDataset.get_simclr_pipeline_transform()

    else:
        raise Exception("Unrecognized model type: " + model_type)


    # make output directory to save training results
    if train_suffix == '':
        model_name = model_type + f'_z{model_config.latent_dim:02}_' + f'ne{n_epochs:03}' #+ f'gamma{int(model_config.gamme):04}_' + f'temp{int(model_config.temperature):04}'
    else:
        model_name = model_type + f'_z{model_config.latent_dim:02}_' + f'ne{n_epochs:03}_' + train_suffix #+ f'gamma{int(model_config.gamma):04}_' + f'temp{int(model_config.temperature):04}'  + '_'
    output_dir = os.path.join(train_dir, model_name)

    # initialize training configuration
    train_config = BaseTrainerConfig(
        output_dir=output_dir,
        num_epochs=n_epochs,
        cache_data=cache_data,
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
    else:
        raise Exception("Unrecognized model type: " + model_type)


    pipeline = TrainingPipeline(
        training_config=train_config,
        model=model
    )

    # # test data loader
    # from pythae.data.datasets import BaseDataset, collate_dataset_output
    # from torch.utils.data import DataLoader, Dataset
    # from torch.utils.data.distributed import DistributedSampler
    # from torch.utils.data.sampler import SubsetRandomSampler

    # test_loader = DataLoader(
    #         dataset=train_dataset,
    #         batch_size=train_config.per_device_train_batch_size,
    #         num_workers=train_config.train_dataloader_num_workers,
    #         shuffle=True,
    #         collate_fn=collate_dataset_output,
    #     )

    # inputs = next(iter(test_loader))

    pipeline(
        train_data=train_dataset,  # here we use the custom train dataset
        eval_data=train_dataset  # here we use the custom eval dataset
    )

    return output_dir


if __name__ == "__main__":
    # from functions.pythae_utils import *

    #####################
    # Required arguments
    # root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
    train_folder = "20231106_ds"
    train_dir = os.path.join(root, "training_data", train_folder)
    model_type = "SeqVAE"

    #####################
    # Optional arguments
    train_suffix = "speed_test"
    temperature = 0.0001
    batch_size = 64
    n_epochs = 2
    latent_dim = 100
    n_conv_layers = 5
    distance_metric = "euclidean"
    input_dim = (1, 288, 128)

    output_dir = train_vae(root, train_folder, train_suffix=train_suffix, model_type=model_type,
                           latent_dim=latent_dim, batch_size=batch_size,
                           n_epochs=n_epochs, temperature=temperature, learning_rate=1e-4, n_conv_layers=n_conv_layers)




