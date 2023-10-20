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
import torch.nn.functional as F
import pandas as pd
from _archive.functions_folder.utilities import path_leaf
from tqdm import tqdm


if __name__ == "__main__":

    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
    batch_size = 128
    overwrite_flag = False
    # load metadata
    metadata_path = os.path.join(root, 'metadata', '')

    train_name = "20230815_vae"
    train_dir = os.path.join(root, "training_data", train_name, '')
    # model_name = "20230804_vae_full_conv_z25_bs032_ne100_depth05"
    # get list of models in this folder
    model_dir = os.path.join(root, "training_data", train_name, "z50_bs032_ne100_depth05")
    last_training = sorted(os.listdir(model_dir))[-1]
    model_fig_dir = os.path.join(model_dir, last_training, "figures")
    embryo_df_rev = pd.read_csv(os.path.join(model_fig_dir, "embryo_stats_df_rev1.csv"), index_col=0)

    main_dims = (576, 256)
    data_transform = make_dynamic_rs_transform(main_dims)

    mode_vec = ["train", "eval", "test"]
    data_sampler_vec = []
    for mode in mode_vec:
        ds_temp = MyCustomDataset(
            root=os.path.join(train_dir, mode),
            transform=data_transform,
            return_name=True
        )
        data_sampler_vec.append(ds_temp)

    try:
        trained_model = AutoModel.load_from_folder(
            os.path.join(model_dir,last_training, 'final_model'))
    except:
        raise Exception("No final model for " + model_dir + ". Still training?")

    for n in range(trained_model.latent_dim):
        embryo_df_rev[f"z_mu_rev{n:02}"] = np.nan
        embryo_df_rev[f"z_sigma_rev{n:02}"] = np.nan

    snip_id_vec = embryo_df_rev["snip_id"]

    print("Calculating latent embeddings...")
    for m, mode in enumerate(mode_vec):

        data_sampler = data_sampler_vec[m]
        n_images = len(data_sampler)

        sample_indices = range(n_images)
        batch_id_vec = []
        n_batches = np.ceil(len(sample_indices) / batch_size).astype(int)
        for n in range(n_batches):
            ind1 = n * batch_size
            ind2 = (n + 1) * batch_size
            batch_id_vec.append(sample_indices[ind1:ind2])

        # get latent space representations for all test images
        print(f"Calculating {mode} latent spaces...")
        for n in tqdm(range(n_batches)):
            batch_ids = batch_id_vec[n]
            im_stack = np.empty((len(batch_ids), main_dims[0], main_dims[1])).astype(np.float32)
            snip_index_vec = []
            snip_name_vec = []
            for b in range(len(batch_ids)):
                im_raw = np.asarray(data_sampler[batch_ids[b]][0]).tolist()[0]
                path_data = data_sampler[batch_ids[b]][1]
                snip_name = path_leaf(path_data[0]).replace(".jpg", "")
                snip_name_vec.append(snip_name)

                snip_ind = np.where(snip_name == snip_id_vec)[0][0]
                snip_index_vec.append(snip_ind)

                if embryo_df_rev.loc[snip_ind, "revision_labels"]<=0:
                    im_stack[b, :, :] = im_raw
                elif embryo_df_rev.loc[snip_ind, "revision_labels"]==1:
                    im_stack[b, :, :] = np.fliplr(im_raw)

            im_test = torch.reshape(torch.from_numpy(im_stack), (len(batch_ids), 1, main_dims[0], main_dims[1]))
            encoder_out = trained_model.encoder(im_test)
            zm_vec = np.asarray(encoder_out[0].detach())
            zs_vec = np.asarray(encoder_out[1].detach())
            snip_ind_array = np.asarray(snip_index_vec)
            for z in range(trained_model.latent_dim):
                embryo_df_rev.loc[snip_ind_array, f"z_mu_rev{z:02}"] = zm_vec[:, z]
                embryo_df_rev.loc[snip_ind_array, f"z_sigma_rev{z:02}"] = zs_vec[:, z]

            # z_mu_array[n, :] = np.asarray(encoder_out[0].detach())
            # z_sigma_array[n, :] = np.asarray(np.exp(encoder_out[1].detach()/2))


    zm_indices = [i for i in range(len(embryo_df_rev.columns)) if "z_mu_rev" in embryo_df_rev.columns[i]]
    z_mu_array = embryo_df_rev.iloc[:, zm_indices].to_numpy()

    # remove D/V embryos
    embryo_df_rev = embryo_df_rev.iloc[np.where(embryo_df_rev["revision_labels"]>=0)[0]]

    print(f"Calculating UMAP...")
    # calculate 2D morphology UMAPS
    reducer = umap.UMAP()
    scaled_z_mu = StandardScaler().fit_transform(z_mu_array)
    embedding2d = reducer.fit_transform(scaled_z_mu)
    embryo_df.loc[:, "UMAP_00"] = embedding2d[:, 0]
    embryo_df.loc[:, "UMAP_01"] = embedding2d[:, 1]

    print(f"Saving data...")
        #save latent arrays and UMAP
        embryo_df = embryo_df.iloc[:, 1:]
        embryo_df.to_csv(os.path.join(figure_path, "embryo_stats_df.csv"))