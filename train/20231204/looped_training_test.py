import sys
sys.path.append("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")
sys.path.append("E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\")

import os
from src.vae.auxiliary_scripts.train_vae import train_vae


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
    temperature_vec = [0.0001, 0.01, 1]
    batch_size = 64
    n_epochs = 3
    latent_dim_vec = [10, 50, 100]
    n_conv_layers = 5
    distance_metric = "euclidean"
    input_dim = (1, 288, 128)

    for latent_dim in latent_dim_vec:
        for temperature in temperature_vec:


            output_dir = train_vae(root, train_folder, train_suffix=train_suffix, model_type=model_type,
                                   latent_dim=latent_dim, batch_size=batch_size,
                                   n_epochs=n_epochs, temperature=temperature, learning_rate=1e-4, n_conv_layers=n_conv_layers)