# We will save the data in folders to mimic the desired example
import os
import torch
import numpy as np
import imageio
import glob2 as glob
import pandas as pd
from functions.utilities import path_leaf

train_eval_test = None
test_ids = []
# set path to data
root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
metadata_path = os.path.join(root, "metadata", '')
data_path = os.path.join(root, "training_data", "bf_embryo_snips", '')

# read in metadata database
embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df_final.csv"), index_col=0)

# get list of training files
image_list = glob.glob(data_path + "*.tif")
snip_id_list = [path_leaf(path) for path in image_list]
emb_id_list = [eid[0:16] for eid in snip_id_list]

# randomly partition into train, eval, and test
# this needs to be done at the level of embryos, not images
if train_eval_test == None:
    train_eval_test = [0.6, 0.2, 0.1]

n_frames_total = np.sum(embryo_metadata_df["use_embryo_flag"].values == True)
snip_id_vec = embryo_metadata_df["snip_id"].values
good_snip_indices = np.where(embryo_metadata_df["use_embryo_flag"].values == True)[0]
embryo_id_vec = np.unique(embryo_metadata_df["embryo_id"].values)
embryo_id_vec_shuffle = np.random.choice(embryo_id_vec, len(embryo_id_vec), replace=False)
image_list_shuffle = []
for eid in embryo_id_vec_shuffle:
    # extract embryos that match current eid
    e_list = [image_list[e] for e in range(len(image_list)) if (emb_id_list[e] == eid) and (e in good_snip_indices)]
    # remove frames that did not meet QC standards
    image_list_shuffle += e_list

if len(test_ids) > 0:
    test_paths = [image_list[i] for i in range(len(image_list)) if emb_id_list[i] in test_ids]
    train_eval_test[-1] = 0
    train_eval_test = train_eval_test / np.sum(train_eval_test)
    embryo_id_vec_shuffle = [e for e in embryo_id_vec_shuffle if e not in test_ids]
else:
    test_paths = []

# assign to groups. Note that a few frames from the same embryo may end up split between categories. I think that this is fine
n_train = np.round(train_eval_test[0]*n_frames_total).astype(int)
train_paths = image_list_shuffle[:n_train]
n_eval = np.round(train_eval_test[1]*n_frames_total).astype(int)
eval_paths = image_list_shuffle[n_train:n_train+n_eval]
test_paths += image_list_shuffle[n_train+n_eval:]


# if not os.path.exists("data_folders"):
#     os.mkdir("data_folders")
# if not os.path.exists("data_folders/train"):
#     os.mkdir("data_folders/train")
# if not os.path.exists("data_folders/eval"):
#     os.mkdir("data_folders/eval")
#
# for i in range(len(train_dataset)):
#     img = 255.0*train_dataset[i][0].unsqueeze(-1)
#     img_folder = os.path.join("data_folders", "train", f"{train_targets[i]}")
#     if not os.path.exists(img_folder):
#         os.mkdir(img_folder)
#     imageio.imwrite(os.path.join(img_folder, "%08d.jpg" % i), np.repeat(img, repeats=3, axis=-1).type(torch.uint8))
#
# for i in range(len(eval_dataset)):
#     img = 255.0*eval_dataset[i][0].unsqueeze(-1)
#     img_folder = os.path.join("data_folders", "eval", f"{eval_targets[i]}")
#     if not os.path.exists(img_folder):
#         os.mkdir(img_folder)
#     imageio.imwrite(os.path.join(img_folder, "%08d.jpg" % i), np.repeat(img, repeats=3, axis=-1).type(torch.uint8))