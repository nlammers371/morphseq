import os
from src.vae.auxiliary_scripts.train_vae import train_vae
import multiprocessing

#######################
# Script to test whether recent tweaks to model architecture are impacting performance

def main():

    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    train_folder = "20241021_ds"

    latent_dim = 100
    n_conv_layers = 5
    batch_size = 256
    n_epochs = 25
    # input_dim = (1, 576, 256)
    input_dim = (1, 288, 128)
    learning_rate = 1e-4
    cache_data = True
    model_type = "SeqVAE"

    #####################
    # (3B & 4B) Seq-VAE (triplet)
    #####################
    train_suffix = "seq_test"
    margin = 15 # , 50]
    metric_weight = 10  # 50, 2, 1000] #[0.0001, 0.001, 0.01, 0.1, 1]
    distance_metric = "euclidean"
    self_target_prob = 0.5
    metric_loss_type = "triplet" 
    n_workers = 2

    train_vae(root, train_folder, train_suffix=train_suffix, model_type=model_type, 
                latent_dim=latent_dim, batch_size=batch_size, input_dim=input_dim,
                n_preload_workers=n_workers, n_load_workers=n_workers, metric_loss_type=metric_loss_type,
                distance_metric=distance_metric, n_epochs=n_epochs, margin=margin, metric_weight=metric_weight, 
                learning_rate=learning_rate, n_conv_layers=n_conv_layers, self_target_prob=self_target_prob,
                cache_data=cache_data)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()