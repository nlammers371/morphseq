from src._Archive.vae.auxiliary_scripts.assess_vae_results import assess_vae_results

# root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\morphseq\\"
batch_size = 64  # batch size to use generating latent encodings and image reconstructions
overwrite_flag = True
n_image_figures = 100  # make qualitative side-by-side reconstruction figures
# n_contrastive_samples = 1000  # number of images to reconstruct for loss calc
skip_figures_flag = False
train_name = "20231106_ds"
architecture_name = "MetricVAE_z100_ne250_temperature_sweep_v2"
# mode_vec = ["train", "eval", "test"]

models_to_assess = ["MetricVAE_training_2023-12-13_09-09-34"]

assess_vae_results(root, train_name, architecture_name, n_image_figures=n_image_figures,
                   overwrite_flag=True, batch_size=64, models_to_assess=models_to_assess)

