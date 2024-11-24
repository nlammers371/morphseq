from src.build.infer_developmental_age import infer_developmental_age

root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
train_name = "20241008"
architecture_name = "VAE_z100_ne250_base_model"
model_name = "VAE_training_2024-10-08_21-40-44"
infer_developmental_age(root, train_name, architecture_name, model_name)