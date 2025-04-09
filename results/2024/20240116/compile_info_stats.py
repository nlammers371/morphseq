from src.vae.auxiliary_scripts._Archive.calculate_latent_info_stats import calculate_latent_info_stats

# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
# root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
train_name = "20231106_ds"
# model_name = "SeqVAE_z100_ne250_triplet_loss_test_self_and_other" #"SeqVAE_z100_ne250_gamma_temp_SELF_ONLY"
architecture_name_vec = ["SeqVAE_z100_ne250_triplet_loss_test_self_and_other", "SeqVAE_z100_ne250_triplet_loss_SELF_ONLY",
                        "SeqVAE_z100_ne250_gamma_temp_self_and_other", "SeqVAE_z100_ne250_gamma_temp_SELF_ONLY",
                        "VAE_z100_ne250_vanilla", "MetricVAE_z100_ne250_temperature_sweep_v2"]
# mode_vec = ["train", "eval", "test"]

models_to_assess = None #["SeqVAE_training_2023-12-12_23-56-02"]

for architecture_name in architecture_name_vec:
    calculate_latent_info_stats(root, train_name, architecture_name)

