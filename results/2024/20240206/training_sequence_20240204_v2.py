from src.vae import train_vae
import multiprocessing

#######################
# Set general arguments used by all models
# root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"

def main():
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\morphseq\\"
    # root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    train_folder = "20240204_ds_v2"
    # train_dir = os.path.join(root, "training_data", train_folder)

    latent_dim = 100
    n_conv_layers = 5
    batch_size = 64
    n_epochs = 250
    input_dim = (1, 288, 128)
    learning_rate = 1e-4
    beta = 1
    model_type = "SeqVAE"

    #####################
    # (3B & 4B) Seq-VAE (triplet)
    #####################
    train_suffix = "triplet_loss_test_SELF_and_OTHER"
    temperature_vec = [10]# , 50]
    gamma_vec = [7.5]  # 50, 2, 1000] #[0.0001, 0.001, 0.01, 0.1, 1]
    distance_metric = "euclidean"
    self_target_prob = 0.5
    metric_loss_type = "triplet"
    age_key_path = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\morphseq\\metadata\\age_key_df.csv"
    # age_key_path = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/metadata/age_key_df.csv"

    for gamma in gamma_vec:
        for temperature in temperature_vec:
            output_dir = train_vae(root, train_folder, train_suffix=train_suffix, model_type=model_type,
                                latent_dim=latent_dim, batch_size=batch_size, beta=beta, n_load_workers=4, metric_loss_type=metric_loss_type,
                                distance_metric=distance_metric, n_epochs=n_epochs, temperature=temperature, gamma=gamma,
                                learning_rate=learning_rate, n_conv_layers=n_conv_layers, self_target_prob=self_target_prob,
                                age_key_path=age_key_path)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()