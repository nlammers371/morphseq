from src.build.infer_developmental_age import infer_developmental_age

root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
train_name = "20240204_ds_v2"
architecture_name = "VAE_z100_ne250_vanilla_VAE"
model_name = "VAE_training_2024-02-04_13-54-24"
reference_datasets=["20231110", "20231206", "20231218"]
infer_developmental_age(root, train_name, architecture_name, model_name, reference_datasets=reference_datasets)