import glob as glob
from src.functions.dataset_utils import *
import os
import skimage.io as io 
from src.vae.models.auto_model import AutoModel
import matplotlib.pyplot as plt
import umap.umap_ as umap
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
# from pythae.trainers.base_trainer_verbose import base_trainer_verbose
#from src.functions.dataset_utils import ContrastiveLearningDataset, ContrastiveLearningViewGenerator
from pythae.data.datasets import collate_dataset_output
from torch.utils.data import DataLoader
import json
from typing import Any, Dict, List, Optional, Union
import ntpath

# Script to generate image reconstructions and latent space projections for a designated set of embryo images
def assess_image_set(image_path, metadata_path, trained_model_path, out_path, image_prefix_list="", rs_factor=1.0, batch_size=64):

    # load raw metadata DF
    embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df_final.csv"), index_col=0)

    # load the model
    trained_model = AutoModel.load_from_folder(os.path.join(trained_model_path, 'final_model'))

    # load images 
    image_path_list = []
    for image_prefix in image_prefix_list:
        im_list = sorted(glob.glob(os.path.join(image_path, image_prefix + "*")))
        image_prefix_list += im_list

    image_snip_names = [path_leaf(im)[:-4] for im in image_path_list]
    keep_indices = np.asarray([i for i in range(snip_id_vec) if snip_id_vec[i] in image_snip_names])

    # pair down data entries
    embryo_df = embryo_metadata_df[
            ["snip_id", "experiment_date", "medium", "master_perturbation", "predicted_stage_hpf"]].iloc[keep_indices].copy()
    embryo_df = embryo_df.reset_index()

    # initialize new columns
    embryo_df["recon_mse"] = np.nan
    snip_id_vec = list(embryo_df["snip_id"])

    # load and store the images
    for i in tqdm(range(len(image_path_list))):
        img_raw = io.imread(mage_path_list[i])
        if rs_flag:
            img = torch.from_numpy(rescale(img_raw.astype(np.float16), rs_factor, anti_aliasing=True))
        else:
            img = torch.from_numpy(img_raw)

        if i == 0:
            # generate tensor to hold images (I'm assuming they will be few enough to hold in memory)
            rs_dims = img.shape
            image_tensor = torch.zeros((len(image_path_names), 1, rs_dims[0], rs_dims[1]))
        
    # initialize latent variable columns
    new_cols = []
    for n in range(trained_model.latent_dim):
        if (trained_model.model_name == "MetricVAE") or (trained_model.model_name == "SeqVAE"):
            if n in trained_model.nuisance_indices:
                new_cols.append(f"z_mu_n_{n:02}")
                new_cols.append(f"z_sigma_n_{n:02}")
            else:
                new_cols.append(f"z_mu_b_{n:02}")
                new_cols.append(f"z_sigma_b_{n:02}")
        else:
            new_cols.append(f"z_mu_{n:02}")
            new_cols.append(f"z_sigma_{n:02}")
    embryo_df.loc[:, new_cols] = np.nan

    print("Extracting latent space vectors and testing image reconstructions...")
    # make subdir for images
    image_path = os.path.join(out_path, "images_reconstructions")
    if not os.path.isdir(image_path):
        os.makedirs(image_path)

    # make subdir for comparison figures
    recon_fig_path = os.path.join(out_path, "recon_figs")
    if not os.path.isdir(recon_fig_path):
        os.makedirs(recon_fig_path)

    # get the dataloader
    n_images = image_tensor.shape[0]
    n_batches = int(np.ceil(n_images/batch_size))
    start_i = 0
    for n in tqdm(range(n_batches)):
        stop_i = np.min([n_images, start_i+batch_size])
        inputs = image_tensor[start_i:stop_i]
        x = set_inputs_to_device(device, inputs)

        encoder_output = trained_model.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)

        z_out, eps = trained_model._sample_gauss(mu, std)
        recon_x_out = trained_model.decoder(z_out)["reconstruction"]
        # .detach().cpu()

        recon_loss = F.mse_loss(
            recon_x_out.reshape(recon_x_out.shape[0], -1),
            x.reshape(x.shape[0], -1),
            reduction="none",
        ).sum(dim=-1).detach().cpu()
        x = x.detach().cpu()
        recon_x_out = recon_x_out.detach().cpu()
        # encoder_output = encoder_output.detach().cpu()
        ###
        # Add recon loss and latent encodings to the dataframe
        df_ind_vec = np.asarray([snip_id_vec.index(snip_id) for snip_id in y])
        embryo_df.loc[df_ind_vec, "train_cat"] = mode
        embryo_df.loc[df_ind_vec, "recon_mse"] = np.asarray(recon_loss)

        # add latent encodings
        zm_array = np.asarray(encoder_output[0].detach().cpu())
        zs_array = np.asarray(encoder_output[1].detach().cpu())
        for z in range(trained_model.latent_dim):
            if (trained_model.model_name == "MetricVAE") or (trained_model.model_name == "SeqVAE"):
                if z in trained_model.nuisance_indices:
                    embryo_df.loc[df_ind_vec, f"z_mu_n_{z:02}"] = zm_array[:, z]
                    embryo_df.loc[df_ind_vec, f"z_sigma_n_{z:02}"] = zs_array[:, z]
                else:
                    embryo_df.loc[df_ind_vec, f"z_mu_b_{z:02}"] = zm_array[:, z]
                    embryo_df.loc[df_ind_vec, f"z_sigma_b_{z:02}"] = zs_array[:, z]
            else:
                embryo_df.loc[df_ind_vec, f"z_mu_{z:02}"] = zm_array[:, z]
                embryo_df.loc[df_ind_vec, f"z_sigma_{z:02}"] = zs_array[:, z]

        # recon_loss_array[i] = recon_loss
        for b in range(len(y)):

            if not skip_figures:
                if fig_counter < n_image_figs:
                    # show results with normal sampler
                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

                    axes[0].imshow(np.squeeze(np.squeeze(x[b, 0, :, :])), cmap='gray')
                    axes[0].axis('off')

                    axes[1].imshow(np.squeeze(np.squeeze(recon_x_out[b, 0, :, :])), cmap='gray')
                    axes[1].axis('off')

                    plt.tight_layout(pad=0.)

                    plt.savefig(
                        os.path.join(image_path, y[b] + f'_loss{int(np.round(recon_loss[b], 0)):05}.jpg'))
                    plt.close()

                    fig_counter += 1


if __name__ == "__main__":

    image_path = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/bf_embryo_snips"
    metadata_path = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/metadata"
    out_path = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/analysis/lmx1b"
    prefix_list = ["20230830", "20230831", "20231207", "20231208"]
    rs_factor = 0.5
    batch_size = 64  # batch size to use generating latent encodings and image reconstructions

    # get path to model
    train_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/"
    train_name = "20231106_ds"
    model_name = "SeqVAE_z100_ne250_triplet_loss_test_self_and_other"
    training_instance = "SeqVAE_training_2024-01-06_03-55-23"
    model_dir = os.path.join(train_root, train_name, model_name, training_instance) 


    assess_image_set(image_path=image_path, 
                     metadata_path=metadata_path, 
                    trained_model_path=model_dir, 
                    out_path=out_path, 
                    image_prefix_list=prefix_list, 
                    rs_factor=rs_factor, 
                    batch_size=batch_size)