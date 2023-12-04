import sys
sys.path.append("/functions")
sys.path.append("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")
sys.path.append("E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\")
# sys.path.append("../src/")

from src.functions.dataset_utils import make_dynamic_rs_transform, MyCustomDataset, ContrastiveLearningDataset, ContrastiveLearningViewGenerator
import os
from src.vae.models import VAE, VAEConfig, MetricVAE, MetricVAEConfig
from src.functions.custom_networks import Encoder_Conv_VAE, Decoder_Conv_VAE
from src.vae.trainers import BaseTrainerConfig
from src.vae.pipelines.training import TrainingPipeline

def train_vae(root, train_folder, n_epochs, model_type, input_dim=None, train_suffix='', **kwargs):

    training_keys = ["batch_size", "learning_rate"] # optional training config kywords
    # model_keys = ["n_latent", "n_out_channels", "zn_frac", "depth", "nt_xent_temperature"]
    training_args = dict({})
    model_args = dict({})
    for key, value in kwargs.items():
        if key in training_keys:
            if key == "batch_size":
                training_args["per_device_train_batch_size"] = value
                training_args["per_device_eval_batch_size"] = value
            else:
                training_args[key] = value
        else:
            model_args[key] = value

    if input_dim == None:
        input_dim = (1, 288, 128)

    train_dir = os.path.join(root, "training_data", train_folder)
    # metadata_path = os.path.join(root, "metadata", '')

    if model_type == "MetricVAE":
        # initialize model configuration
        model_config = MetricVAEConfig(
            input_dim=input_dim,
            **model_args
        )
        # initialize contrastive data loader
        data_transform = ContrastiveLearningViewGenerator(
                                                 ContrastiveLearningDataset.get_simclr_pipeline_transform(), 2)
    elif model_type == "VAE":
        # load standard VAE config
        model_config = VAEConfig(
            input_dim=input_dim,
            **model_args
        )
        # Standard data transform
        data_transform = make_dynamic_rs_transform()
    else:
        raise Exception("Unrecognized model type: " + model_type)

    # Make datasets
    train_dataset = MyCustomDataset(root=os.path.join(train_dir, "train"),
                                    transform=data_transform,
                                    return_name=True
                                    )

    eval_dataset = MyCustomDataset(root=os.path.join(train_dir, "eval"),
                                    transform=data_transform,
                                    return_name=True
                                    )


    # make output directory to save training results
    if train_suffix == '':
        model_name = model_type + f'_z{model_config.latent_dim:02}_' + f'ne{n_epochs:03}'
    else:
        model_name = model_type + f'_z{model_config.latent_dim:02}_' + f'ne{n_epochs:03}' + '_' + train_suffix
    output_dir = os.path.join(train_dir, model_name)

    # initialize training configuration
    config = BaseTrainerConfig(
        output_dir=output_dir,
        num_epochs=n_epochs,
        **training_args
    )

    # Initialize encoder and decoder
    encoder = Encoder_Conv_VAE(model_config)  # these are custom classes I wrote for this use case
    decoder = Decoder_Conv_VAE(encoder)

    # initialize model
    if model_type == "MetricVAE":
        model = MetricVAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder
        )
    elif model_type == "VAE":
        model = VAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder
        )
    else:
        raise Exception("Unrecognized model type: " + model_type)

    pipeline = TrainingPipeline(
        training_config=config,
        model=model
    )

    pipeline(
        train_data=train_dataset,  # here we use the custom train dataset
        eval_data=eval_dataset  # here we use the custom eval dataset
    )

    return output_dir


if __name__ == "__main__":
    # from functions.pythae_utils import *

    #####################
    # Required arguments
    root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
    train_folder = "20231120_ds_small"
    train_dir = os.path.join(root, "training_data", train_folder)
    model_type = "MetricVAE"

    #####################
    # Optional arguments
    train_suffix = "refactor_test"
    temperature = 0.0001
    batch_size = 64
    n_epochs = 100
    latent_dim = 100
    n_conv_layers = 5
    distance_metric = "euclidean"
    input_dim = (1, 288, 128)

    # for z in z_dim_vec:
    #     for d in depth_vec:
    #         for t in temperature_vec:

    output_dir = train_vae(root, train_folder, train_suffix=train_suffix, model_type=model_type,
                           latent_dim=latent_dim, batch_size=batch_size,
                           n_epochs=n_epochs, temperature=temperature, learning_rate=1e-4, n_conv_layers=n_conv_layers)




