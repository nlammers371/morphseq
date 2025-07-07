from src._Archive.vae import train_vae
import multiprocessing

#######################
# Script to test updates to training pipeline

def main():

    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
    train_folder = "20240509_ds"

    latent_dim = 100
    n_conv_layers = 5
    batch_size = 128
    n_epochs = 250
    input_dim = (1, 288, 128)
    learning_rate = 1e-4
    cache_data = False
    model_type = "SeqVAE"

    #####################
    # (3B & 4B) Seq-VAE (triplet)
    #####################
    train_suffix_vec = ["all", "gdf3_lmx"]
    temperature_vec = [15, 22] # , 50]
    gamma = 10  # 50, 2, 1000] #[0.0001, 0.001, 0.01, 0.1, 1]
    distance_metric = "euclidean"
    self_target_prob = 0.5
    metric_loss_type = "triplet" #"NT-Xent" 

    age_key_path = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/metadata/age_key_df.csv"
    metric_key_path = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/metadata/curation/perturbation_metric_key_20240508.csv"
    pert_time_key_path_vec = ["", "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/metadata/curation/perturbation_train_key_gdf3_30_no_lmx1b.csv"]

    for t, train_suffix in enumerate(train_suffix_vec):
        for temperature in temperature_vec:
            pert_time_key_path = pert_time_key_path_vec[t]
            train_suffix += f"_temp{temperature:02}"
            output_dir = train_vae(root, train_folder, train_suffix=train_suffix, model_type=model_type,
                                latent_dim=latent_dim, batch_size=batch_size, input_dim=input_dim,
                                n_preload_workers=4, n_load_workers=4, metric_loss_type=metric_loss_type,
                                distance_metric=distance_metric, n_epochs=n_epochs, temperature=temperature, gamma=gamma, 
                                learning_rate=learning_rate, n_conv_layers=n_conv_layers, self_target_prob=self_target_prob,
                                age_key_path=age_key_path, metric_key_path=metric_key_path, cache_data=cache_data, pert_time_key_path=pert_time_key_path)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()