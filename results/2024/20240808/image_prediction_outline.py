import os
from src._Archive.vae import AutoModel
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict


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


# Script to generate image reconstructions from latent space vectors
def assess_image_set(image_path, metadata_path, trained_model_path, out_path, image_prefix_list="", rs_factor=1.0,
                     batch_size=64):

    rs_flag = rs_factor != 1
    # check for GPU
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # load your data here
    print("Loading metadata...")
    embryo_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df_final.csv"), index_col=0)

    # load the model
    print("Loading model...")
    trained_model = AutoModel.load_from_folder(os.path.join(trained_model_path, 'final_model'))

    # make subdir for images
    image_path = os.path.join(out_path, "images_reconstructions")
    if not os.path.isdir(image_path):
        os.makedirs(image_path)

    # make subdir for side-by-side comparison figures
    recon_fig_path = os.path.join(out_path, "recon_figs")
    if not os.path.isdir(recon_fig_path):
        os.makedirs(recon_fig_path)

    print("Extracting latent space vectors and testing image reconstructions...")
    trained_model = trained_model.to(device)

    # iterate through df...you may wnat to batch this somehow so you can pass multiple mu vecs at once
    for n in tqdm(range(embryo_df.shape[0])):

        mu, log_var = ... # EXTRACT MU HERE

        recon_x_out = trained_model.decoder(mu)["reconstruction"]

        recon_loss = F.mse_loss(
            recon_x_out.reshape(recon_x_out.shape[0], -1),
            x.reshape(x.shape[0], -1),
            reduction="none",
        ).sum(dim=-1).detach().cpu()
        # x = x.detach().cpu()
        recon_x_out = recon_x_out.detach().cpu()