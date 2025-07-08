from src.vae import calculate_latent_info_stats

# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
# root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
train_name = "20231106_ds"
architecture_name_vec = ["VAE_z100_ne250_vanilla"]
# mode_vec = ["train", "eval", "test"]

models_to_assess = None #["SeqVAE_training_2023-12-12_23-56-02"]

for architecture_name in architecture_name_vec:
    calculate_latent_info_stats(root, train_name, architecture_name)

