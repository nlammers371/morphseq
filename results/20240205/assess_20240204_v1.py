from src.vae.auxiliary_scripts.assess_vae_results import assess_vae_results

# root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\morphseq\\"
batch_size = 64  # batch size to use generating latent encodings and image reconstructions
overwrite_flag = False
n_image_figures = 100  # make qualitative side-by-side reconstruction figures
# n_contrastive_samples = 1000  # number of images to reconstruct for loss calc
skip_figures_flag = False
train_name = "20240204_ds_v1"
architecture_name_vec = ["SeqVAE_z100_ne250_triplet_loss_test_SELF_and_OTHER"]
# mode_vec = ["train", "eval", "test"]

models_to_assess = None #["SeqVAE_training_2024-02-06_09-11-54"]

for architecture_name in architecture_name_vec:
    assess_vae_results(root, train_name, architecture_name, n_image_figures=n_image_figures,
                    overwrite_flag=overwrite_flag, batch_size=64, models_to_assess=models_to_assess)