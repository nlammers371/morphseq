from src._Archive.vae import assess_vae_results

root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"

batch_size = 64  # batch size to use generating latent encodings and image reconstructions
overwrite_flag = True
n_image_figures = 100  # make qualitative side-by-side reconstruction figures
skip_figures_flag = True
train_name = "20240607"
# architecture_name_vec = ["SeqVAE_z100_ne250_all_temp15", "SeqVAE_z100_ne250_gdf3_lmx_temp15",
                         # "SeqVAE_z100_ne250_all_temp22", "SeqVAE_z100_ne250_gdf3_lmx_temp22",]

architecture_name_vec = ["MorphIAFVAE_z100_ne250_iaf_test_v2", "SeqVAE_z100_ne250_seq_test"]
models_to_assess = None

for architecture_name in architecture_name_vec:
    assess_vae_results(root, train_name, architecture_name, n_image_figures=n_image_figures,
                    overwrite_flag=overwrite_flag, batch_size=64, models_to_assess=models_to_assess)
