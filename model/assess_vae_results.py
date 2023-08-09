from functions.pythae_utils import *
import os
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from pythae.models import AutoModel
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import umap
import numpy as np
from sklearn.preprocessing import StandardScaler
from pythae.samplers import NormalSampler
import torch.nn.functional as F

if __name__ == "__main__":

    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
    train_name = "20230804_vae_full"
    model_name = "20230804_vae_full_conv_z25_bs032_ne100_depth05"

    train_dir = os.path.join(root, "training_data", train_name)

    output_dir = os.path.join(train_dir, model_name) #"/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/training_data/20230807_vae_test/"

    main_dims = (576, 256)
    n_image_figures = 100  # make qualitative side-by-side figures
    n_images_to_sample = 1000 # number of images to reconstruct for loss calc
    data_transform = make_dynamic_rs_transform(main_dims)

    mode_vec = ["train", "eval", "test"]
    data_sampler_vec = []
    for mode in mode_vec:
        ds_temp = MyCustomDataset(
            root=os.path.join(train_dir, mode),
            transform=data_transform
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

    print("Making image figures...")
    for m, mode in enumerate(mode_vec):

        # make subdir for images
        image_path = os.path.join(figure_path, mode + "_images")
        if not os.path.isdir(image_path):
            os.makedirs(image_path)

        data_sampler = data_sampler_vec[m]
        n_images = len(data_sampler)
        n_image_figs = np.min([n_images, n_image_figures])
        n_recon_samples = np.min([n_images, n_images_to_sample])

        # draw random samples
        sample_indices = np.random.choice(range(n_images), n_images_to_sample, replace=False)
        recon_loss_array = np.empty((n_recon_samples,))

        for i, i_test in enumerate(sample_indices):

            im_test = torch.reshape(np.asarray(data_sampler[i_test]).tolist()[0], (1, 1, main_dims[0], main_dims[1]))
            im_recon = trained_model.reconstruct(im_test).detach().cpu()

            recon_loss = F.mse_loss(
                im_recon.reshape(im_test.shape[0], -1),
                im_test.reshape(im_test.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
            recon_loss_array[i] = recon_loss

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
        np.save(os.path.join(figure_path, mode + "_set_recon_loss.npy"), recon_loss_array)

    # ############
    # # Question 2: what does latent space look like?
    # ############
    #
    # print("Calculating latent embeddings...")
    # for m, mode in enumerate(mode_vec):
    #
    #     data_sampler = data_sampler_vec[m]
    #     n_images = len(data_sampler)
    #
    #     # get latent space representations for all test images
    #     z_mu_array = np.empty((n_images, trained_model.latent_dim))
    #     z_sigma_array = np.empty((n_images, trained_model.latent_dim))
    #     print(f"Calculating {mode} latent spaces...")
    #     for n in range(n_images):
    #         im_test = torch.reshape(np.asarray(data_sampler[n]).tolist()[0], (1, 1, main_dims[0], main_dims[1]))
    #         encoder_out = trained_model.encoder(im_test)
    #
    #         z_mu_array[n, :] = np.asarray(encoder_out[0].detach())
    #         z_sigma_array[n, :] = np.asarray(np.exp(encoder_out[1].detach()/2))
    #
    #     print(f"Calculating {mode} UMAPs...")
    #     # calculate 2D morphology UMAPS
    #     reducer = umap.UMAP()
    #     scaled_z_mu = StandardScaler().fit_transform(z_mu_array)
    #     embedding2d = reducer.fit_transform(scaled_z_mu)
    #
    #     print(f"Saving {mode} data...")
    #     #save latent arrays and UMAP
    #     np.save(os.path.join(figure_path, mode + "_set_z_mu_scores.npy"), z_mu_array)
    #     np.save(os.path.join(figure_path, mode + "_set_z_sigma_scores.npy"), z_sigma_array)
    #
    #     np.save(os.path.join(figure_path, mode + "_set_z_mu_UMAP_scores.npy"), embedding2d)
    #
    # print("Done.")


