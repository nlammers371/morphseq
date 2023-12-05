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

def make_seq_key(root, train_name):

    metadata_path = os.path.join(root, "metadata", '')
    
    # read in metadata database
    embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df_final.csv"), index_col=0)
    seq_key = embryo_metadata_df.loc[:, ["snip_id", "experiment_id", "predicted_stage_hpf", "master_perturbation"]]
    pert_u = np.unique(seq_key["master_perturbation"].to_numpy())
    pert_index = np.arange(len(pert_u))
    pert_df = pd.DataFrame(pert_u, columns=["master_perturbation"])
    pert_df["perturbation_id"] = pert_index

    seq_key = seq_key.merge(pert_df, how="left", on="master_perturbation")
        
    seq_key.to_csv(os.path.join(metadata_path, "seq_key.csv"))


if __name__ == "__main__":
    # set path to data
    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"

    train_name = "20231106_ds_test"
    label_var = "experiment_date"

    make_training_key(root, train_name)