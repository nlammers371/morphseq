from src.build.infer_developmental_age import infer_developmental_age

root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
train_name = "20240509_ds"
architecture_name = "VAE_z100_ne250_base_model_v2"
model_name = "VAE_training_2024-05-11_14-20-17"
reference_datasets = ["20231110", "20240307", "20231206", "20240509"] #"20231206", , "20240306", "20240307", , "20240411", "20240418"] #,
infer_developmental_age(root, train_name, architecture_name, model_name, reference_datasets=reference_datasets)