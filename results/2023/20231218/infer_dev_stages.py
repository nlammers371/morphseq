from src.vae import infer_developmental_age

root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
train_name = "20231106_ds"
architecture_name = "VAE_z100_ne250_vanilla"
model_name = "VAE_training_2023-12-18_12-37-02"

age_key_df = infer_developmental_age(root, train_name, architecture_name, model_name)
