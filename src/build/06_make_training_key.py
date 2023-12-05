# We will save the data in folders to mimic the desired example
import os
import torch
import numpy as np
import glob as glob
import pandas as pd
from src.functions.utilities import path_leaf
import imageio
from tqdm import tqdm
from skimage.transform import rescale

def make_seq_key(root, train_name, time_window=3, self_weight=0.5):

    metadata_path = os.path.join(root, "metadata", '')
    training_path = os.path.join(root, "training_data", train_name, '')
    mode_vec = ["train", "eval", "test"]

    # read in metadata database
    embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df_final.csv"), index_col=0)
    seq_key = embryo_metadata_df.loc[:, ["snip_id", "experiment_id", "predicted_stage_hpf", "master_perturbation"]]

    # The above dataset comprises all available images. Some training folders will only use a subset of these. Check
    # which images are in the present one (specified by "train name")
    image_list = []
    mode_list = []
    for mode in mode_vec:
        lb_folder_list = glob.glob(os.path.join(training_path, mode, '') + "*")
        for subdir in lb_folder_list:
            image_list_temp = glob.glob(os.path.join(subdir, '') + "*.jpg")
            image_names = [path_leaf(path)[:-4] for path in image_list_temp]
            image_list += image_names
            mode_list += [mode]*len(image_names)
    # make temporary dataframe for merge purposes
    temp_df = pd.DataFrame(np.asarray(image_list), columns=["snip_id"])
    temp_df["train_cat"] = mode_list
    seq_key = seq_key.merge(temp_df, how="inner", on="snip_id")

    # join on perturbation ID variables for convenience
    pert_u = np.unique(seq_key["master_perturbation"].to_numpy())
    pert_index = np.arange(len(pert_u))
    pert_df = pd.DataFrame(pert_u, columns=["master_perturbation"])
    pert_df["perturbation_id"] = pert_index
    seq_key = seq_key.merge(pert_df, how="left", on="master_perturbation")

    emb_id_list = [snip_id[:-6] for snip_id in list(seq_key["snip_id"])]
    seq_key["embryo_id"] = emb_id_list
    # now, for each image compile a list of valid comparisons and corresponding delta T's
    # to be a valid comparison, an image needs to be of the same perturbation type and within 3 hpf
    for i in range(seq_key.shape[0]):
        self_entry = pd.DataFrame(seq_key.iloc[i, :]).transpose()
        snip_id = seq_key.loc[i, "snip_id"]
        age_hpf = seq_key.loc[i, "predicted_stage_hpf"]
        pert_id = seq_key.loc[i, "perturbation_id"]
        train_cat_id = seq_key.loc[i, "train_cat"]

        seq_key_self = seq_key.merge(self_entry["embryo_id"], how="inner", on=["embryo_id"])
        self_age_deltas = np.abs(seq_key_self["predicted_stage_hpf"].to_numpy() - age_hpf)
        valid_self_ids = seq_key_self.loc[np.where(self_age_deltas <= time_window)[0], "snip_id"].to_numpy()

        seq_key_other = seq_key.merge(self_entry[["perturbation_id", "train_cat"]], how="inner", on=["perturbation_id", "train_cat"])
        other_age_deltas = np.abs(seq_key_other["predicted_stage_hpf"].to_numpy() - age_hpf)
        valid_other_ids = seq_key_other.loc[np.where(other_age_deltas <= time_window)[0], "snip_id"].to_numpy()
    seq_key.to_csv(os.path.join(metadata_path, "seq_key.csv"))


if __name__ == "__main__":
    # set path to data
    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\morphseq\\"

    train_name = "20231106_ds"
    label_var = "experiment_date"

    make_seq_key(root, train_name)