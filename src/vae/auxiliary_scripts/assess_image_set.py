import glob as glob
from src.functions.dataset_utils import *
import os
import skimage.io as io
from skimage.transform import rescale
from src.vae.models.auto_model import AutoModel
import matplotlib.pyplot as plt
import umap.umap_ as umap
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import pandas as pd
from pythae.data.datasets import collate_dataset_output
from tqdm import tqdm
# from pythae.trainers.base_trainer_verbose import base_trainer_verbose
#from src.functions.dataset_utils import ContrastiveLearningDataset, ContrastiveLearningViewGenerator
from src.functions.utilities import path_leaf
from torch.utils.data import DataLoader
import json
from typing import Any, Dict, List, Optional, Union
import ntpath
from src.vae.auxiliary_scripts.assess_vae_results import calculate_UMAPs


def set_inputs_to_device(device, inputs: Dict[str, Any]):
    inputs_on_device = inputs

    if device == "cuda":
        cuda_inputs = dict.fromkeys(inputs)

        for key in inputs.keys():
            if torch.is_tensor(inputs[key]):
                cuda_inputs[key] = inputs[key].cuda()

            else:
                cuda_inputs[key] = inputs[key]
        inputs_on_device = cuda_inputs

    return inputs_on_device

# Script to generate image reconstructions and latent space projections for a designated set of embryo images
def assess_image_set(image_path, metadata_path, trained_model_path, out_path, image_prefix_list="", rs_factor=1.0, batch_size=64):

    rs_flag = rs_factor != 1
    # check for GPU
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # load raw metadata DF
    print("Loading metadata...")
    embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df_final.csv"), index_col=0)

    # load the model
    print("Loading model...")
    trained_model = AutoModel.load_from_folder(os.path.join(trained_model_path, 'final_model'))

    # load images 
    image_path_list = []
    for image_prefix in image_prefix_list:
        im_list = sorted(glob.glob(os.path.join(image_path, image_prefix + "*")))
        image_path_list += im_list

    image_snip_names = [path_leaf(im)[:-4] for im in image_path_list]


    # pair down data entries
    embryo_df = embryo_metadata_df.loc[:,
            ["snip_id", "experiment_date", "medium", "master_perturbation", "predicted_stage_hpf"]].copy()

    # initialize new columns
    embryo_df["recon_mse"] = np.nan
    snip_id_vec = list(embryo_df["snip_id"])
    keep_indices = np.asarray([i for i in range(len(snip_id_vec)) if snip_id_vec[i] in image_snip_names])

    embryo_df = embryo_df.loc[keep_indices, :].copy()
    embryo_df.reset_index(inplace=True)

    # load and store the images
    print("Loading images...")
    input_image_path = os.path.join(out_path, "input_images", "class0")
    if not os.path.isdir(input_image_path):
        os.makedirs(input_image_path)

    for i in tqdm(range(len(image_path_list))):
        img_raw = io.imread(image_path_list[i])
        if len(img_raw.shape) == 3:
            img_raw = img_raw[:, :, 0]
        
        if rs_flag:
            img = rescale(img_raw.astype(np.float16), rs_factor, preserve_range=True, anti_aliasing=True).astype(np.uint8)
        else:
            img = img_raw
        im_name = path_leaf(image_path_list[i])
        # save
        io.imsave(fname=os.path.join(input_image_path, im_name), arr=img)

    data_transform = make_dynamic_rs_transform()
    dataset = MyCustomDataset(
            root=os.path.join(out_path, "input_images"),
            transform=data_transform,
            return_name=True
        )
    dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_dataset_output,
                )
    # initialize latent variable columns
    new_cols = []
    for n in range(trained_model.latent_dim):
        if (trained_model.model_name == "MetricVAE") or (trained_model.model_name == "SeqVAE"):
            if n in trained_model.nuisance_indices:
                new_cols.append(f"z_mu_n_{n:02}")
                # new_cols.append(f"z_sigma_n_{n:02}")
            else:
                new_cols.append(f"z_mu_b_{n:02}")
                # new_cols.append(f"z_sigma_b_{n:02}")
        else:
            new_cols.append(f"z_mu_{n:02}")
            # new_cols.append(f"z_sigma_{n:02}")
    embryo_df.loc[:, new_cols] = np.nan

    # make subdir for images
    image_path = os.path.join(out_path, "images_reconstructions")
    if not os.path.isdir(image_path):
        os.makedirs(image_path)

    # make subdir for comparison figures
    recon_fig_path = os.path.join(out_path, "recon_figs")
    if not os.path.isdir(recon_fig_path):
        os.makedirs(recon_fig_path)

    print("Extracting latent space vectors and testing image reconstructions...")
    trained_model = trained_model.to(device)
    # get the dataloader
    start_i = 0
    for n, inputs in enumerate(tqdm(dataloader)):

        inputs = set_inputs_to_device(device, inputs)
        x = inputs["data"]

        encoder_output = trained_model.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)

        z_out, eps = trained_model._sample_gauss(mu, std)
        recon_x_out = trained_model.decoder(z_out)["reconstruction"]

        recon_loss = F.mse_loss(
            recon_x_out.reshape(recon_x_out.shape[0], -1),
            x.reshape(x.shape[0], -1),
            reduction="none",
        ).sum(dim=-1).detach().cpu()
        # x = x.detach().cpu()
        recon_x_out = recon_x_out.detach().cpu()

        # encoder_output = encoder_output.detach().cpu()
        # ###
        # # Add recon loss and latent encodings to the dataframe
        stop_i = np.min([start_i + batch_size, len(dataset)])
        iter_snips = image_snip_names[start_i:stop_i]
        df_ind_vec = np.asarray([np.where(embryo_df.loc[:, "snip_id"] == snip)[0][0] for snip in iter_snips])
        embryo_df.loc[df_ind_vec, "recon_mse"] = np.asarray(recon_loss)

        start_i += batch_size
        # add latent encodings
        zm_array = np.asarray(encoder_output[0].detach().cpu())

        for z in range(trained_model.latent_dim):
            if (trained_model.model_name == "MetricVAE") or (trained_model.model_name == "SeqVAE"):
                if z in trained_model.nuisance_indices:
                    embryo_df.loc[df_ind_vec, f"z_mu_n_{z:02}"] = zm_array[:, z]
                else:
                    embryo_df.loc[df_ind_vec, f"z_mu_b_{z:02}"] = zm_array[:, z]
            else:
                embryo_df.loc[df_ind_vec, f"z_mu_{z:02}"] = zm_array[:, z]

        for b in range(x.shape[0]):

            # show results
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

            axes[0].imshow(np.squeeze(np.squeeze(x.detach().cpu()[b, 0, :, :])), cmap='gray')
            axes[0].axis('off')

            axes[1].imshow(np.squeeze(recon_x_out[b, 0, :, :]), cmap='gray')
            axes[1].axis('off')

            plt.tight_layout(pad=0.)

            plt.savefig(
                os.path.join(recon_fig_path, image_snip_names[b] + '_loss.jpg'))
            plt.close()

            # save just the recon on its own
            int_recon_out = (np.squeeze(np.asarray(recon_x_out[b, 0, :, :]))*255).astype(np.uint8)
            io.imsave(fname=os.path.join(image_path, image_snip_names[b] + '_loss.jpg'), arr=int_recon_out)


    # now fit 2 and 3-dimensional UMAP models 
    print("Calculating UMAPS...")
    embryo_df = calculate_UMAPs(embryo_df)

    print("Saving...")
    embryo_df.to_csv(os.path.join(out_path, "embryo_df.csv"))
       

if __name__ == "__main__":

    # root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\morphseq"
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
    image_path = os.path.join(root, "training_data", "bf_embryo_snips")
    # image_path = os.path.join(root, "training_data", "20231106_ds", "train", "20230525")
    metadata_path = os.path.join(root, "metadata")
    out_path = os.path.join(root, "analysis", "lmx1b")
    prefix_list = ["20230830", "20230831", "20231207", "20231208"]
    rs_factor = 0.5
    batch_size = 64  # batch size to use generating latent encodings and image reconstructions

    # get path to model
    train_root = os.path.join(root, "training_data")
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