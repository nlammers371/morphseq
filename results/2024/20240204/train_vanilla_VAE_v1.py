# This script trains a generic VAE on old and new sets. Will be used primarily to sync developmental staging
import os
from src.vae import train_vae

root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
train_folder = "20230204_ds_v1"
train_dir = os.path.join(root, "training_data", train_folder)
model_type = "VAE"

train_suffix = "vanilla_VAE"
batch_size = 64
n_epochs = 250
beta = 1 #[0.01, 0.1, 1]
latent_dim = 100
n_conv_layers = 5
input_dim = (1, 288, 128)

output_dir = train_vae(root, train_folder, train_suffix=train_suffix, model_type=model_type,
                    latent_dim=latent_dim, batch_size=batch_size, beta=beta, n_load_workers=4, n_epochs=n_epochs, 
                    learning_rate=1e-4, n_conv_layers=n_conv_layers)