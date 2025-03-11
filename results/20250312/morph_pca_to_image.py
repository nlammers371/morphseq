import os
import sys

code_root = "/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq"
sys.path.insert(0, code_root)

import glob as glob
from src.functions.dataset_utils import *
import os
from src.vae.models.auto_model import AutoModel
import umap.umap_ as umap
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from sklearn.neural_network import MLPRegressor
# from pythae.trainers.base_trainer_verbose import base_trainer_verbose
from pythae.data.datasets import collate_dataset_output
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import joblib 



if __name__ == "__main__":

    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"

    # path to save data
    read_path = os.path.join(root, "results", "20250312", "morph_latent_space", "")

    # load in PCA embeddings and PCA model
    hf_pca_df = pd.read_csv(os.path.join(read_path, "hf_morph_df.csv"))
    ref_pca_df = pd.read_csv(os.path.join(read_path, "ab_ref_morph_df.csv"))
    spline_df = pd.read_csv(os.path.join(read_path, "spline_morph_df_full.csv"))

    # Save the fitted PCA instance to a file
    morph_pca = joblib.load(os.path.join(read_path, 'morp_pca_model.pkl'))