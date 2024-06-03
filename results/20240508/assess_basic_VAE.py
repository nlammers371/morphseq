from src.vae.auxiliary_scripts.assess_vae_results import assess_vae_results

root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"

batch_size = 64  # batch size to use generating latent encodings and image reconstructions
overwrite_flag = True
n_image_figures = 10000  # make qualitative side-by-side reconstruction figures
skip_figures_flag = True
train_name = "20240507"
architecture_name_vec = ["VAE_z100_ne250_base_model"]

models_to_assess = None

for architecture_name in architecture_name_vec:
    assess_vae_results(root, train_name, architecture_name, n_image_figures=n_image_figures,
                    overwrite_flag=overwrite_flag, batch_size=64, models_to_assess=models_to_assess)
