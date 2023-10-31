import glob as glob
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from functions.pythae_utils import *
import os
from pythae.models import AutoModel
import matplotlib.pyplot as plt
import umap
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch

def calculate_UMAPs(embryo_df, n_components=3):

    zmb_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_b" in embryo_df.columns[i]]
    zmn_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_n" in embryo_df.columns[i]]

    embryo_df = embryo_df.reset_index()

    MetricFlag = len(zmb_indices) > 0

    if MetricFlag:
        umap_df = embryo_df[
            ["snip_id", "experiment_date", "medium", "master_perturbation", "predicted_stage_hpf", "train_cat", "recon_mse",
             "UMAP_00", "UMAP_01", "UMAP_00_bio", "UMAP_01_bio", "UMAP_00_n", "UMAP_01_n"]].copy()
    else:
        umap_df = embryo_df[
            ["snip_id", "experiment_date", "medium", "master_perturbation", "predicted_stage_hpf", "train_cat", "recon_mse",
             "UMAP_00", "UMAP_01"]].copy()

    mu_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_" in embryo_df.columns[i]]

    z_mu_array = embryo_df.iloc[:, mu_indices].to_numpy()
    if MetricFlag:
        z_mu_array_b = embryo_df.iloc[:, zmb_indices].to_numpy()
        z_mu_array_n = embryo_df.iloc[:, zmn_indices].to_numpy()

    print(f"Calculating UMAP...")
    dim_str = str(n_components)
    # calculate 2D morphology UMAPS
    reducer = umap.UMAP(n_components=n_components)
    scaled_z_mu = StandardScaler().fit_transform(z_mu_array)
    embedding = reducer.fit_transform(scaled_z_mu)
    for n in range(n_components):
        umap_df.loc[:, f"UMAP_{n:02}_" + dim_str] = embedding[:, n]

    if MetricFlag:
        reducer_bio = umap.UMAP(n_components=n_components)
        scaled_z_mu_bio = StandardScaler().fit_transform(z_mu_array_b)
        embedding_bio = reducer_bio.fit_transform(scaled_z_mu_bio)
        for n in range(n_components):
            umap_df.loc[:, f"UMAP_{n:02}_bio_" + dim_str] = embedding_bio[:, n]

        reducer_n = umap.UMAP(n_components=n_components)
        scaled_z_mu_n = StandardScaler().fit_transform(z_mu_array_n)
        embedding_n = reducer_n.fit_transform(scaled_z_mu_n)
        for n in range(n_components):
            umap_df.loc[:, f"UMAP_{n:02}_n_" + dim_str] = embedding_n[:, n]

    return umap_df


def initialize_assessment(train_dir, output_dir, mode_vec=None):

    if mode_vec is None:
        mode_vec = ["train", "eval", "test"]

    continue_flag = False

    data_transform = make_dynamic_rs_transform(main_dims)
    data_sampler_vec = []
    for mode in mode_vec:
        ds_temp = MyCustomDataset(
            root=os.path.join(train_dir, mode),
            transform=data_transform,
            return_name=True
        )
        data_sampler_vec.append(ds_temp)

    try:
        trained_model = AutoModel.load_from_folder(os.path.join(output_dir, 'final_model'))

    except:
        try:
            trained_model_list = glob.glob(os.path.join(output_dir, "*epoch*"))
            underscore_list = [s.rfind("_") for s in trained_model_list]
            epoch_num_list = [int(trained_model_list[s][underscore_list[s] + 1:]) for s in range(len(underscore_list))]
            last_ind = np.argmax(epoch_num_list)

            # last_training = path_leaf(trained_model_list[last_ind])
            trained_model = AutoModel.load_from_folder(trained_model_list[last_ind])

            print("No final model found for " + output_dir + ". Using most recent saved training instance.")
        except:
            print("No final model for " + output_dir + ". Still training?")
            # continue_flag = True
            trained_model = []


    figure_path = os.path.join(output_dir, "figures")
    if not os.path.isdir(figure_path):
        raise Exception("No figure directory found. Have you run assess_metric_vae_results?")

    # meta_df = pd.read_csv(os.path.join(figure_path, "meta_summary_df.csv"))
    vae_df = pd.read_csv(os.path.join(figure_path, "embryo_stats_df.csv"))

    return trained_model, vae_df, figure_path, data_sampler_vec, continue_flag

if __name__ == "__main__":

    root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    # root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
    # batch_size = 128  # batch size to use generating latent encodings and image reconstructions
    overwrite_flag = True
    # main_dims = (576, 256)
    n_image_figures = 100  # make qualitative side-by-side figures
    # n_contrastive_samples = 1000  # number of images to reconstruct for loss calc
    # test_contrastive_pairs = True

    mode_vec = ["train", "eval", "test"]

    # load metadata
    metadata_path = os.path.join(root, 'metadata', '')

    train_name = "20230915_vae" #"20230915_vae"
    architecture_name = "z100_bs032_ne250_depth05_out16_temperature_sweep2"
    # architecture_name = "z50_bs032_ne010_depth05_out16_metric_test"
    train_dir = os.path.join(root, "training_data", train_name, '')

    # get list of models in this folder
    models_to_assess = ["MetricVAE_training_2023-10-27_09-29-34"]

    if models_to_assess is None:
        models_to_assess = sorted(glob.glob(os.path.join(train_dir, architecture_name, '*metric_test*')))

    for m_iter, model_name in enumerate(models_to_assess):

        print("Evaluating model " + model_name + f'({m_iter + 1:02} of ' + str(len(models_to_assess)) + ')')

        output_dir = os.path.join(train_dir, architecture_name, model_name) #"/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/training_data/20230807_vae_test/"

        _, vae_df, figure_path, _, continue_flag = initialize_assessment(train_dir, output_dir)
        # if continue_flag:
        #     continue

        prev_run_flag = os.path.isfile(os.path.join(figure_path, "vae_df.csv"))

        if prev_run_flag and overwrite_flag == False:
            print("Results already exist for: " + figure_path + ". Skipping.")
            continue

        # np.random.seed(123)

        umap_df = calculate_UMAPs(vae_df)

        umap_df2 = calculate_UMAPs(vae_df, n_components=2)

        umap_df["UMAP_00_2"] = umap_df2["UMAP_00_2"]
        umap_df["UMAP_01_2"] = umap_df2["UMAP_01_2"]
        umap_df["UMAP_00_bio_2"] = umap_df2["UMAP_00_bio_2"]
        umap_df["UMAP_01_bio_2"] = umap_df2["UMAP_01_bio_2"]
        umap_df["UMAP_00_n_2"] = umap_df2["UMAP_00_n_2"]
        umap_df["UMAP_01_n_2"] = umap_df2["UMAP_01_n_2"]

        print(f"Saving data...")
        umap_df.to_csv(os.path.join(figure_path, "umap_df.csv"))
        print("Done.")


