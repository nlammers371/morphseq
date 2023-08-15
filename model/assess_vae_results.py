from _archive.functions_folder.pythae_utils import *
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

    # load metadata
    metadata_path = os.path.join(root, 'metadata', '')
    embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df_final.csv"), index_col=0)
    embryo_df = embryo_metadata_df[
        ["snip_id", "experiment_date", "medium", "master_perturbation", "predicted_stage_hpf", "surface_area_um",
         "length_um", "width_um"]].iloc[np.where(embryo_metadata_df["use_embryo_flag"] == 1)].copy()
    embryo_df = embryo_df.reset_index()
    snip_id_vec = embryo_df["snip_id"]

    train_name = "20230804_vae_full"
    # model_name = "20230804_vae_full_conv_z25_bs032_ne100_depth05"
    model_name = "20230804_vae_full_conv_z25_bs032_ne100_depth05_matchdec01"
    train_dir = os.path.join(root, "training_data", train_name)

    output_dir = os.path.join(train_dir, model_name) #"/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/training_data/20230807_vae_test/"

    main_dims = (576, 256)
    n_image_figures = 100  # make qualitative side-by-side figures
    n_images_to_sample = 1000  # number of images to reconstruct for loss calc
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


    last_training = sorted(os.listdir(output_dir))[-1]
    trained_model = AutoModel.load_from_folder(
        os.path.join(output_dir, last_training, 'final_model'))

    ############
    # Question 1: how well does it reproduce train, eval, and test images?
    ############

    figure_path = os.path.join(output_dir, last_training, "figures")
    if not os.path.isdir(figure_path):
        os.makedirs(figure_path)
    print("Saving to: " + figure_path)

    np.random.seed(123)

    # initialize new columns
    embryo_df["train_cat"] = ''
    embryo_df["recon_mse"] = np.nan

    print("Making image figures...")
    for m, mode in enumerate(mode_vec):

        # make subdir for images
        image_path = os.path.join(figure_path, mode + "_images")
        if not os.path.isdir(image_path):
            os.makedirs(image_path)

        data_sampler = data_sampler_vec[m]
        n_images = len(data_sampler)
        n_image_figs = np.min([n_images, n_image_figures])
        n_recon_samples = n_images #np.min([n_images, n_images_to_sample])

        # draw random samples
        sample_indices = np.random.choice(range(n_images), n_recon_samples, replace=False)
        # recon_loss_array = np.empty((n_recon_samples,))

        print("Scoring image reconstructions for " + mode + " images...")
        for i, i_test in enumerate(tqdm(sample_indices)):

            im_raw = np.asarray(data_sampler[i_test][0]).tolist()[0]
            path_data = data_sampler[i_test][1]
            snip_name = path_leaf(path_data[0]).replace(".jpg", "")
            snip_df_index = np.where(snip_name == snip_id_vec)[0][0]

            im_test = torch.reshape(im_raw, (1, 1, main_dims[0], main_dims[1]))
            im_recon = trained_model.reconstruct(im_test).detach().cpu()

            recon_loss = F.mse_loss(
                im_recon.reshape(im_test.shape[0], -1),
                im_test.reshape(im_test.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
            # recon_loss_array[i] = recon_loss

            embryo_df.loc[snip_df_index, "train_cat"] = mode
            embryo_df.loc[snip_df_index, "recon_mse"] = np.asarray(recon_loss)[0]

            if i <= n_image_figs:
                # show results with normal sampler
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

                axes[0].imshow(np.squeeze(im_test), cmap='gray')
                axes[0].axis('off')

                axes[1].imshow(np.squeeze(im_recon), cmap='gray')
                axes[1].axis('off')

                plt.tight_layout(pad=0.)

                plt.savefig(os.path.join(image_path, f'im_{i_test:04}_loss{int(np.round(recon_loss,0)):05}.tiff'))
                plt.close()



        # save
        # np.save(os.path.join(figure_path, mode + "_set_recon_loss.npy"), recon_loss_array)
    if np.any(np.isnan(embryo_df.loc[:, "recon_mse"].to_numpy())):
        print("Uh-Oh")
    ############
    # Question 2: what does latent space look like?
    ############
    for n in range(trained_model.latent_dim):
        embryo_df[f"z_mu_{n:02}"] = np.nan
        embryo_df[f"z_sigma_{n:02}"] = np.nan

    print("Calculating latent embeddings...")
    for m, mode in enumerate(mode_vec):

        data_sampler = data_sampler_vec[m]
        n_images = len(data_sampler)

        # get latent space representations for all test images
        # z_mu_array = np.empty((n_images, trained_model.latent_dim))
        # z_sigma_array = np.empty((n_images, trained_model.latent_dim))
        print(f"Calculating {mode} latent spaces...")
        for n in tqdm(range(n_images)):
            im_raw = np.asarray(data_sampler[n][0]).tolist()[0]
            path_data = data_sampler[n][1]
            snip_name = path_leaf(path_data[0]).replace(".jpg", "")
            snip_df_index = np.where(snip_name == snip_id_vec)[0][0]

            im_test = torch.reshape(im_raw, (1, 1, main_dims[0], main_dims[1]))
            encoder_out = trained_model.encoder(im_test)
            zm_vec = np.asarray(encoder_out[0].detach())
            zs_vec = np.asarray(encoder_out[1].detach())
            for z in range(trained_model.latent_dim):
                embryo_df.loc[snip_df_index, f"z_mu_{z:02}"] = zm_vec[0][z]
                embryo_df.loc[snip_df_index, f"z_sigma_{z:02}"] = zs_vec[0][z]

            # z_mu_array[n, :] = np.asarray(encoder_out[0].detach())
            # z_sigma_array[n, :] = np.asarray(np.exp(encoder_out[1].detach()/2))


    zm_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_" in embryo_df.columns[i]]
    z_mu_array = embryo_df.iloc[:, zm_indices].to_numpy()

    print(f"Calculating UMAP...")
    # calculate 2D morphology UMAPS
    reducer = umap.UMAP()
    scaled_z_mu = StandardScaler().fit_transform(z_mu_array)
    embedding2d = reducer.fit_transform(scaled_z_mu)
    embryo_df["UMAP_00"] = embedding2d[:, 0]
    embryo_df["UMAP_01"] = embedding2d[:, 1]

    print(f"Saving data...")
    #save latent arrays and UMAP
    embryo_df = embryo_df.iloc[:, 1:]
    embryo_df.to_csv(os.path.join(figure_path, "embryo_stats_df.csv"))

    print("Done.")


