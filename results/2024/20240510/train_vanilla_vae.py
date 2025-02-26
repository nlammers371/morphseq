import os
from src.vae.auxiliary_scripts.train_vae import train_vae
import multiprocessing

#######################
# Script to test updates to training pipeline

def main():
    # root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\morphseq\\"
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    train_folder = "20240204_ds_v1"
    # train_dir = os.path.join(root, "training_data", train_folder)

    latent_dim = 100
    n_conv_layers = 5
    batch_size = 128
    n_epochs = 250
    input_dim = (1, 288, 128)
    learning_rate = 1e-4
    # model_type = "SeqVAE"
    cache_data = False
    model_type = "VAE"

    #####################
    # (3B & 4B) Seq-VAE (triplet)
    #####################
    train_suffix = "base_model_v2"
    
    # age_key_path = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\morphseq\\metadata\\age_key_df.csv"
    # age_key_path = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/metadata/age_key_df.csv"
    pert_time_key_path = "" # "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/metadata/curation/perturbation_train_key_gdf3_36.csv"
    age_key_path = "" #"/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/metadata/age_key_df.csv"
    output_dir = train_vae(root, train_folder, train_suffix=train_suffix, model_type=model_type, 
                                latent_dim=latent_dim, batch_size=batch_size, input_dim=input_dim,
                                pert_time_key_path=pert_time_key_path, age_key_path=age_key_path,
                                n_preload_workers=4, n_load_workers=4, n_epochs=n_epochs,
                                learning_rate=learning_rate, n_conv_layers=n_conv_layers, cache_data=cache_data)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()