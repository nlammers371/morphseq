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
    model_type_vec = ["SeqVAE"]

    #####################
    # Optional arguments
    train_suffix = "multiclass_ntxent_beta_temp_sweep"
    temperature_vec = [0.0001, 0.001, 0.01, 0.1, 1]
    batch_size = 64
    n_epochs = 250
    beta_vec = [0.1, 1, 10, 100]
    latent_dim = 100
    n_conv_layers = 5
    distance_metric = "euclidean"
    model_type = "SeqVAE"
    age_key_path = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/20231106_ds/VAE_z100_ne250_vanilla/VAE_training_2023-12-18_12-37-02/figures/age_key_df.csv"
    input_dim = (1, 288, 128)

    for beta in beta_vec:
        for temperature in temperature_vec:
            output_dir = train_vae(root, train_folder, train_suffix=train_suffix, model_type=model_type,
                                latent_dim=latent_dim, batch_size=batch_size, beta=beta,  n_load_workers=4, age_key_path=age_key_path,
                                n_epochs=n_epochs, temperature=temperature, learning_rate=1e-4, n_conv_layers=n_conv_layers, distance_metric=distance_metric)