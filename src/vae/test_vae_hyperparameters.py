import sys
sys.path.append("/functions")
# print("hello")
# import functions
from src.functions.dataset_utils import make_dynamic_rs_transform, data_transform, MyCustomDataset
import os
from pythae.models import VAE, VAEConfig, BetaTCVAE, BetaTCVAEConfig
from src.functions.custom_networks import Encoder_Conv_VAE, Decoder_Conv_VAE
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
# from pythae.models import AutoModel
# import matplotlib.pyplot as plt
# from pythae.samplers import NormalSampler

def train_vanilla_vae(train_dir, n_latent=50, batch_size=32, n_epochs=100, learning_rate=1e-3, n_out_channels=16,
                      input_dim=None, depth=5, tc_flag=False, beta_factor=1):

    # input_dim = (1, 128, 128)
    if input_dim == None:
        input_dim = (1, 576, 256)
        transform = data_transform
    else:
        transform = make_dynamic_rs_transform(input_dim[1:])
    if not tc_flag:
        model_name = f'z{n_latent:02}_' + f'bs{batch_size:03}_' + f'ne{n_epochs:03}_' + f'depth{depth:02}_' + f'out{n_out_channels:02}'
    else:
        model_name = f'z{n_latent:02}_' + f'bs{batch_size:03}_' + f'ne{n_epochs:03}_' + f'depth{depth:02}' + f'_beta{beta_factor:02}'

    output_dir = os.path.join(train_dir, model_name)

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

    if not tc_flag:
        model_config = VAEConfig(
            input_dim=input_dim,
            latent_dim=n_latent
        )
    else:
        model_config = BetaTCVAEConfig(
            input_dim=input_dim,
            latent_dim=n_latent,
            beta=beta_factor
        )

    encoder = Encoder_Conv_VAE(model_config, n_conv_layers=depth, n_out_channels=n_out_channels) # these are custom classes I wrote for this use case
    # if matched_decoder_flag:
    decoder = Decoder_Conv_VAE(encoder)
    # else:
    #     decoder = Decoder_Conv_AE_FLEX(encoder)
    if not tc_flag:
        model = VAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder
        )
    else:
        model = BetaTCVAE(
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
    train_folder = "20230915_vae"
    tc_flag = False  # indicates whether we wish to use standard or disentangled TC vae
    train_dir = os.path.join(root, "training_data", train_folder)
    batch_size = 32
    n_epochs = 100
    z_dim_vec = [50, 100, 250]
    depth_vec = [5]
    beta_vec = []
    n_out_channel_vec = [16, 32, 64]
    max_tries = 3
    # output_dir = train_vanilla_vae(train_dir, n_latent=10, batch_size=batch_size, n_epochs=n_epochs,
    #                                learning_rate=1e-4, depth=7)

    if not tc_flag:
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
                            output_dir = train_vanilla_vae(train_dir, n_latent=z, batch_size=batch_size, n_epochs=n_epochs,
                                                            n_out_channels=n, learning_rate=1e-4, depth=d, tc_flag=tc_flag)
                            iter_flag = max_tries
                        except:
                            iter_flag += 1
                            print(iter_flag)

    else:
        for z in z_dim_vec:
            for d in depth_vec:
                for b in beta_vec:
                    iter_flag = 0
                    while iter_flag < max_tries:
                        try:
                            print(f"Depth: {d}")
                            print(f"Latent dim: {z}")
                            print(f"Beta weight: {b}")
                            # train model
                            output_dir = train_vanilla_vae(train_dir, n_latent=z, batch_size=batch_size,
                                                           n_epochs=n_epochs,
                                                           learning_rate=1e-4, depth=d, tc_flag=tc_flag,
                                                           beta_factor=b)
                            iter_flag = max_tries
                        except:
                            iter_flag += 1
                            print(iter_flag)

