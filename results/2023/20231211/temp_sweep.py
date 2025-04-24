import sys
sys.path.append("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")
sys.path.append("E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\")

import os
from src._Archive.vae.auxiliary_scripts.train_vae import train_vae


if __name__ == "__main__":
    # from functions.pythae_utils import *

    #####################
    # Required arguments
    # root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    train_folder = "20231106_ds"
    train_dir = os.path.join(root, "training_data", train_folder)
    model_type = "SeqVAE"

    #####################
    # Optional arguments
    train_suffix = "temperature_sweep_v2"
    temperature_vec = [0.0001, 0.001, 0.01]
    batch_size = 64
    n_epochs = 250
    beta_vec = [1] #[0.01, 0.1, 1]
    latent_dim = 100
    n_conv_layers = 5
    distance_metric = "euclidean"
    input_dim = (1, 288, 128)

    for beta in beta_vec:
        for temperature in temperature_vec:

            output_dir = train_vae(root, train_folder, train_suffix=train_suffix, model_type=model_type,
                                   latent_dim=latent_dim, batch_size=batch_size, beta=beta,
                                   n_epochs=n_epochs, temperature=temperature, learning_rate=1e-4, n_conv_layers=n_conv_layers)