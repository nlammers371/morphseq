from src.vae import generate_contrastive_dataframe

root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
# root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\morphseq\\"
batch_size = 64  # batch size to use generating latent encodings and image reconstructions
overwrite_flag = True
n_image_figures = 100  # make qualitative side-by-side reconstruction figures
# n_contrastive_samples = 1000  # number of images to reconstruct for loss calc
skip_figures_flag = False
train_name = "20231106_ds"
architecture_name_vec = ["SeqVAE_z100_ne250_gamma_temp_self_and_other", "SeqVAE_z100_ne250_gamma_temp_SELF_ONLY", "SeqVAE_z100_ne250_triplet_loss_test_self_and_other"]
# mode_vec = ["train", "eval", "test"]

models_to_assess = None #["SeqVAE_training_2023-12-12_23-56-02"]

for architecture_name in architecture_name_vec:
    generate_contrastive_dataframe(root, train_name, architecture_name, overwrite_flag=overwrite_flag, batch_size=64, models_to_assess=models_to_assess)

