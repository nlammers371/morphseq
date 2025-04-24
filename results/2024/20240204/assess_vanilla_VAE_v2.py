from src._Archive.vae.auxiliary_scripts.assess_vae_results import assess_vae_results

root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
# root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\morphseq\\"
batch_size = 64  # batch size to use generating latent encodings and image reconstructions
overwrite_flag = True
n_image_figures = 100  # make qualitative side-by-side reconstruction figures
# n_contrastive_samples = 1000  # number of images to reconstruct for loss calc
skip_figures_flag = True
train_name = "20240204_ds_v2"
architecture_name_vec = ["VAE_z100_ne250_vanilla_VAE"]
# mode_vec = ["train", "eval", "test"]
# /net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/20240204_ds_v2/VAE_z100_ne250_vanilla_VAE
models_to_assess = None #["SeqVAE_training_2023-12-21_00-18-39", "SeqVAE_training_2023-12-21_04-44-03"]

for architecture_name in architecture_name_vec:
    assess_vae_results(root, train_name, architecture_name, n_image_figures=n_image_figures,
                    overwrite_flag=overwrite_flag, batch_size=64, models_to_assess=models_to_assess)

