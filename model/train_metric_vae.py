import sys
sys.path.append("/functions")
# print("hello")
# import functions
from functions.pythae_utils import make_dynamic_rs_transform, data_transform, MyCustomDataset
import os
from pythae.models import VAE, VAEConfig, BetaTCVAE, BetaTCVAEConfig
from custom_classes.metric_vae_model import METRICVAE
from functions.custom_networks import Encoder_Conv_VAE, Decoder_Conv_VAE
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from functions.ContrastiveLearningDataset import ContrastiveLearningDataset
from functions.view_generator import ContrastiveLearningViewGenerator
from custom_classes.base_trainer_metric import BaseTrainerMetric

# import matplotlib.pyplot as plt
# from pythae.samplers import NormalSampler
import argparse

def train_metric_vae(train_dir, n_latent=50, batch_size=32, n_epochs=100, learning_rate=1e-3, n_out_channels=16,
                      input_dim=None, depth=5, contrastive_flag=False):

    if input_dim == None:
        input_dim = (1, 576, 256)
        transform = data_transform
    else:
        transform = make_dynamic_rs_transform(input_dim[1:])

    model_name = f'z{n_latent:02}_' + f'bs{batch_size:03}_' + f'ne{n_epochs:03}_' + f'depth{depth:02}_' + f'out{n_out_channels:02}'


    output_dir = os.path.join(train_dir, model_name)

    if contrastive_flag:
        # train_generator = ContrastiveLearningDataset(os.path.join(train_dir, "train"))
        # train_dataset = train_generator.get_dataset('custom', 2)
        train_dataset = MyCustomDataset(root=os.path.join(train_dir, "train"),
                                        transform=ContrastiveLearningViewGenerator(
                                                 ContrastiveLearningDataset.get_simclr_pipeline_transform(),#(96),
                                                 2)
                                        )

        # eval_generator = ContrastiveLearningDataset(os.path.join(train_dir, "eval"))
        # eval_dataset = eval_generator.get_dataset('custom', 2)

        eval_dataset = MyCustomDataset(root=os.path.join(train_dir, "eval"),
                                        transform=ContrastiveLearningViewGenerator(
                                            ContrastiveLearningDataset.get_simclr_pipeline_transform(),  # (96),
                                            2)
                                        )

    else:
        train_dataset = MyCustomDataset(
            root=os.path.join(train_dir, "train"),
            transform=transform
        )

        eval_dataset = MyCustomDataset(
            root=os.path.join(train_dir, "eval"),
            transform=transform
        )

    config = BaseTrainerConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_epochs=n_epochs,  # Change this to train the model a bit more
    )

    model_config = VAEConfig(
        input_dim=input_dim,
        latent_dim=n_latent
    )


    encoder = Encoder_Conv_VAE(model_config, n_conv_layers=depth, n_out_channels=n_out_channels) # these are custom classes I wrote for this use case
    # if matched_decoder_flag:
    decoder = Decoder_Conv_VAE(encoder)
    # else:
    #     decoder = Decoder_Conv_AE_FLEX(encoder)

    model = METRICVAE(
        model_config=model_config,
        encoder=encoder,
        decoder=decoder
    )

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

    # parser = argparse.ArgumentParser(description="Function to call VAE training in batch",
    #                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    # parser.add_argument("-rd", "--root", default="E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\", help="string path to folder contining 'training_data' subfolder")
    # parser.add_argument("-nz", "--n_latent", default=25, help="Integer specifying number of latent dimensions")
    # parser.add_argument("-ne", "--n_epochs", default=100, help="Integer specifying number of training epochs")
    # parser.add_argument("-bs", "--batch_size", default=32, help="Integer specifying number of images per batch")
    # parser.add_argument("-md", "--depth", default=5, help="Integer specifying number of convolutional layers")
    # parser.add_argument("-tp", "--train_folder", default="20230815_vae", help="Folder containing training data to use")
    # parser.add_argument("-lt", "--learning_rate", default=1e-4, help="float <<1 specifying learning rate for model training")
    #
    # args = vars(parser.parse_args())
    #
    # train_dir = os.path.join(args["root"], "training_data", args["train_folder"])

    root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
    train_folder = "20230804_vae_test"
    contrastive_flag = True
    train_dir = os.path.join(root, "training_data", train_folder)
    batch_size = 32
    n_epochs = 10
    z_dim_vec = [50]
    depth_vec = [5]
    n_out_channel_vec = [16]
    max_tries = 3
    # output_dir = train_vanilla_vae(train_dir, n_latent=10, batch_size=batch_size, n_epochs=n_epochs,
    #                                learning_rate=1e-4, depth=7)

    for z in z_dim_vec:
        for d in depth_vec:
            for n in n_out_channel_vec:
                iter_flag = 0
                while iter_flag < max_tries:
                    try:
                        print(f"Depth: {d}")
                        print(f"Latent dim: {z}")
                        print(f"Out channels: {n}")
                        # train model
                        output_dir = train_metric_vae(train_dir, n_latent=z, batch_size=batch_size, n_epochs=n_epochs,
                                                        n_out_channels=n, learning_rate=1e-4, depth=d, contrastive_flag=contrastive_flag)
                        iter_flag = max_tries
                    except:
                        iter_flag += 1
                        print(iter_flag)



