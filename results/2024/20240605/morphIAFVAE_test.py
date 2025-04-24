from src._Archive.vae.auxiliary_scripts.train_vae import train_vae
import multiprocessing

#######################
# Script to test updates to training pipeline

def main():

    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
    train_folder = "20240607"

    latent_dim = 100
    n_conv_layers = 5
    batch_size = 64
    n_epochs = 250
    input_dim = (1, 576, 256)
    learning_rate = 1e-4
    cache_data = False
    model_type = "MorphIAFVAE"

    #####################
    # (3B & 4B) Seq-VAE (triplet)
    #####################
    train_suffix = "iaf_test"
    temperature = 15 # , 50]
    gamma = 10  # 50, 2, 1000] #[0.0001, 0.001, 0.01, 0.1, 1]
    distance_metric = "euclidean"
    self_target_prob = 0.5
    metric_loss_type = "triplet" #"NT-Xent" 

    age_key_path = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/metadata/age_key_df.csv"
    metric_key_path = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/metadata/curation/perturbation_metric_key_basic.csv"

    output_dir = train_vae(root, train_folder, train_suffix=train_suffix, model_type=model_type, 
                                latent_dim=latent_dim, batch_size=batch_size, input_dim=input_dim,
                                n_preload_workers=4, n_load_workers=4, metric_loss_type=metric_loss_type,
                                distance_metric=distance_metric, n_epochs=n_epochs, temperature=temperature, gamma=gamma, 
                                learning_rate=learning_rate, n_conv_layers=n_conv_layers, self_target_prob=self_target_prob,
                                age_key_path=age_key_path, metric_key_path=metric_key_path, cache_data=cache_data)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()