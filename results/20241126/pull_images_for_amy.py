import pandas as pd
import os
import numpy as np
from tqdm import tqdm 
import skimage.io as io
import glob2 as glob
import skimage

# set key path parameters
root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/" # path to top of the data directory
train_folder = "20241107_ds" # name of 'master' training folder that contains all runs

# load metadata DF
embryo_df = pd.read_csv(os.path.join(root, "training_data", train_folder, "embryo_metadata_df_train.csv"))

# get list of SeaHUB genotypes
genotype_list = np.asarray(['smad5', 'smyda:smydb', 'sox10', 'sox10:tbx5a', 'tbx2b', 'tbx2b:tbx2a', 'tbx5a', 'tbx6','meox1', 'meox1:sox1', 'meox1:tbx5a', 'nfatc1', 'nfatc:gata2a'])

# make output directory
out_dir = os.path.join(root, "shared_data", "seahub_crispant_images")
image_dir = os.path.join(root, "shared_data", "seahub_crispant_images", "images")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# find rows in metadata df
genotype_vec = embryo_df["genotype"].to_numpy()
seahub_flags = np.isin(genotype_vec, genotype_list)

# filter
keep_cols = ["snip_id", "experiment_date", "genotype", "has_sci_data", "microscope", "pert_type", "phenotype", "use_flag"]
seahub_df = embryo_df.loc[seahub_flags, keep_cols].reset_index(drop=True)

# save the metadata
seahub_df.to_csv(os.path.join(out_dir, "image_metadata.csv"), index=False)

# now lets save the images themselves
for i in range(seahub_df.shape[0]):

    # get key info
    date = seahub_df.loc[i, "experiment_date"]
    genotype = seahub_df.loc[i, "genotype"].replace(":", "-")
    snip = seahub_df.loc[i, "snip_id"]

    # load
    im_mask = io.imread(os.path.join(root, "training_data", "bf_embryo_masks", "emb" + "_" + snip + ".jpg"))
    im_cropped = io.imread(os.path.join(root, "training_data", "bf_embryo_snips", date, snip + ".jpg"))
    im_uncropped = io.imread(os.path.join(root, "training_data", "bf_embryo_snips_uncropped", date, snip + ".jpg"))

    # invert
    im_cropped_inv = 255 - im_uncropped.copy()
    im_mask[im_mask > 1] = 1
    # try distance-based taper 
    noise_array = np.zeros_like(im_cropped) + 40
    mask_cropped_gauss = skimage.filters.gaussian(im_mask.astype(float), sigma= 175/6.5)
    im_cropped_gauss = np.multiply(im_cropped_inv.astype(float), mask_cropped_gauss) + np.multiply(noise_array, 1-mask_cropped_gauss)

    im_cropped_gauss = im_cropped_gauss.astype(np.uint8)

    # save 
    save_stub = genotype + "_" + snip[:-10]
    io.imsave(os.path.join(image_dir, save_stub + "_cropped.jpg"), im_cropped_gauss, check_contrast=False)
    io.imsave(os.path.join(image_dir, save_stub + "_uncropped.jpg"), 255 - im_uncropped, check_contrast=False)



