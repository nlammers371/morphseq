import glob as glob
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from src.functions.dataset_utils import *
import os
from src.vae.models.auto_model import AutoModel
import matplotlib.pyplot as plt
import umap.umap_ as umap
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
# from pythae.trainers.base_trainer_verbose import base_trainer_verbose
from src.functions.dataset_utils import ContrastiveLearningDataset, ContrastiveLearningViewGenerator
from pythae.data.datasets import collate_dataset_output
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import json
from typing import Any, Dict, List, Optional, Union
import ntpath

def clean_path_names(path_list):
    path_list_out = []
    for path in path_list:
        head, tail = ntpath.split(path)
        path_list_out.append(tail[:-4])

    return path_list_out
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

def get_embryo_age_predictions(embryo_df, mu_indices):

    train_indices = np.where((embryo_df["train_cat"] == "train") | (embryo_df["train_cat"] == "eval"))[0]
    test_indices = np.where(embryo_df["train_cat"] == "test")[0]

    # extract target vector
    y_train = embryo_df["predicted_stage_hpf"].iloc[train_indices].to_numpy().astype(float)
    y_test = embryo_df["predicted_stage_hpf"].iloc[test_indices].to_numpy().astype(float)

    # extract predictor variables
    X_train = embryo_df.iloc[train_indices, mu_indices].to_numpy().astype(float)
    X_test = embryo_df.iloc[test_indices, mu_indices].to_numpy().astype(float)

    ###################
    # run MLP regressor
    clf_age_nonlin = MLPRegressor(random_state=1, max_iter=5000).fit(X_train, y_train)

    ###################
    # Run multivariate linear regressor
    clf_age_lin = linear_model.LinearRegression().fit(X_train, y_train)

    # initialize pandas dataframe to store results
    X_full = embryo_df.iloc[:, mu_indices].to_numpy().astype(float)

    y_pd_nonlin = clf_age_nonlin.predict(X_full)
    y_score_nonlin = clf_age_nonlin.score(X_test, y_test)

    y_pd_lin = clf_age_lin.predict(X_full)
    y_score_lin = clf_age_lin.score(X_test, y_test)

    return y_pd_lin, y_score_lin, y_pd_nonlin, y_score_nonlin

def get_pert_class_predictions(embryo_df, mu_indices):

    train_indices = np.where((embryo_df["train_cat"] == "train"))[0]
    test_indices = np.where((embryo_df["train_cat"] == "test") | (embryo_df["train_cat"] == "eval"))[0]

    pert_df = embryo_df.loc[:, ["snip_id", "predicted_stage_hpf", "train_cat", "master_perturbation"]].copy()

    ########################
    # How well does latent space predict perturbation type?
    pert_class_train = np.asarray(embryo_df["master_perturbation"].iloc[train_indices])
    pert_class_test = np.asarray(embryo_df["master_perturbation"].iloc[test_indices])


    # extract predictor variables
    X_train = embryo_df.iloc[train_indices, mu_indices].to_numpy().astype(float)
    X_test = embryo_df.iloc[test_indices, mu_indices].to_numpy().astype(float)

    ###################
    # run MLP classifier
    ###################
    clf = MLPClassifier(random_state=1, max_iter=5000).fit(X_train, pert_class_train)
    accuracy_nonlin = clf.score(X_test, pert_class_test)

    ###################
    # Run multivariate logistic classifier
    ###################
    clf_lin = LogisticRegression(random_state=0).fit(X_train, pert_class_train)
    accuracy_lin = clf_lin.score(X_test, pert_class_test)

    class_pd_nonlin_train = clf.predict(X_train)
    class_pd_nonlin_test = clf.predict(X_test)

    class_pd_lin_train = clf_lin.predict(X_train)
    class_pd_lin_test = clf_lin.predict(X_test)

    # pert_df_train = pert_df.iloc[train_indices]
    # pert_df_test = pert_df.iloc[test_indices]

    pert_df.loc[train_indices, "class_nonlinear_pd"] = class_pd_nonlin_train
    pert_df.loc[train_indices, "class_linear_pd"] = class_pd_lin_train

    pert_df.loc[test_indices, "class_nonlinear_pd"] = class_pd_nonlin_test
    pert_df.loc[test_indices, "class_linear_pd"] = class_pd_lin_test


    return accuracy_nonlin, accuracy_lin, pert_df


def assess_image_reconstructions(embryo_df, trained_model, figure_path, data_sampler_vec,
                                 n_image_figures, device, mode_vec=None, skip_figures=False):

    if mode_vec is None:
        mode_vec = ["train", "eval", "test"]

    # initialize new columns
    embryo_df["train_cat"] = ''
    embryo_df["recon_mse"] = np.nan
    snip_id_vec = list(embryo_df["snip_id"])

    # initialize latent variable columns
    new_cols = []
    for n in range(trained_model.latent_dim):

        if trained_model.model_name == "MetricVAE":
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

    print("Making image figures...")
    for m, mode in enumerate(mode_vec):

        # make subdir for images
        image_path = os.path.join(figure_path, mode + "_images")
        if not os.path.isdir(image_path):
            os.makedirs(image_path)

        # get the dataloader
        data_loader = data_sampler_vec[m]
        n_images = len(data_loader.dataset)
        n_image_figs = np.min([n_images, n_image_figures])

        # recon_loss_array = np.empty((n_recon_samples,))
        fig_counter = 0
        print("Scoring image reconstructions for " + mode + " images...")
        for n, inputs in enumerate(tqdm(data_loader)):

            inputs = set_inputs_to_device(device, inputs)
            x = inputs["data"]
            y = list(inputs["label"][0])
            y = clean_path_names(y)

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
                if trained_model.model_name == "MetricVAE":
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
                            os.path.join(image_path, y[b] + f'_loss{int(np.round(recon_loss[b], 0)):05}.tiff'))
                        plt.close()

                        fig_counter += 1

    embryo_df = embryo_df.dropna(ignore_index=True)

    return embryo_df

def calculate_UMAPs(embryo_df):

    print(f"Calculating UMAPs...")
    zmb_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_b" in embryo_df.columns[i]]
    zmn_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_n" in embryo_df.columns[i]]
    mu_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_" in embryo_df.columns[i]]
    # embryo_df = embryo_df.reset_index()

    MetricFlag = len(zmb_indices) > 0

    z_mu_array = embryo_df.iloc[:, mu_indices].to_numpy()
    if MetricFlag:
        z_mu_array_b = embryo_df.iloc[:, zmb_indices].to_numpy()
        z_mu_array_n = embryo_df.iloc[:, zmn_indices].to_numpy()

    for n_components in [2, 3]:
        dim_str = str(n_components)
        # calculate 2D morphology UMAPS
        reducer = umap.UMAP(n_components=n_components)
        scaled_z_mu = StandardScaler().fit_transform(z_mu_array)
        embedding = reducer.fit_transform(scaled_z_mu)
        for n in range(n_components):
            embryo_df.loc[:, f"UMAP_{n:02}_" + dim_str] = embedding[:, n]

        if MetricFlag:
            reducer_bio = umap.UMAP(n_components=n_components)
            scaled_z_mu_bio = StandardScaler().fit_transform(z_mu_array_b)
            embedding_bio = reducer_bio.fit_transform(scaled_z_mu_bio)
            for n in range(n_components):
                embryo_df.loc[:, f"UMAP_{n:02}_bio_" + dim_str] = embedding_bio[:, n]

            reducer_n = umap.UMAP(n_components=n_components)
            scaled_z_mu_n = StandardScaler().fit_transform(z_mu_array_n)
            embedding_n = reducer_n.fit_transform(scaled_z_mu_n)
            for n in range(n_components):
                embryo_df.loc[:, f"UMAP_{n:02}_n_" + dim_str] = embedding_n[:, n]

    return embryo_df

def calculate_contrastive_distances(embryo_df, meta_df, trained_model, train_dir, device, mode_vec=None):

    if mode_vec is None:
        mode_vec = ["train", "eval", "test"]

    c_data_loader_vec = []
    # n_total_samples = 0
    for mode in mode_vec:
        temp_dataset = MyCustomDataset(root=os.path.join(train_dir, mode),
                                       transform=ContrastiveLearningViewGenerator(
                                           ContrastiveLearningDataset.get_simclr_pipeline_transform(),  # (96),
                                           2)
                                       )
        data_loader = DataLoader(
            dataset=temp_dataset,
            batch_size=batch_size,
            collate_fn=collate_dataset_output,
        )

        c_data_loader_vec.append(data_loader)
        # n_total_samples += np.min([n_contrastive_samples, len(data_loader)])

    metric_df_list = []
    zm_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_" in embryo_df.columns[i]]
    z_col_list = embryo_df.columns[zm_indices].to_list()
    sample_iter = 0

    # embryo_df = embryo_df.reset_index()
    for m, mode in enumerate(mode_vec):
        data_loader = c_data_loader_vec[m]

        print(f"Calculating {mode} contrastive differences...")
        for i, inputs in enumerate(tqdm(data_loader)):
            inputs = set_inputs_to_device(device, inputs)
            x = inputs["data"]
            bs = x.shape[0]

            # initialize temporary DF
            metric_df_temp = pd.DataFrame(np.empty((x.shape[0] * 2, trained_model.latent_dim + 3 + 2)),
                                          columns=["sample_id", "contrast_id", "train_cat", "euc_all",
                                                   "cos_all"] + z_col_list)

            # latent_encodings = trained_model.encoder(inputs)
            x0 = torch.reshape(x[:, 0, :, :, :],
                               (x.shape[0], x.shape[2], x.shape[3], x.shape[4]))  # first set of images
            x1 = torch.reshape(x[:, 1, :, :, :], (
                x.shape[0], x.shape[2], x.shape[3], x.shape[4]))  # second set with matched contrastive pairs

            encoder_output0 = trained_model.encoder(x0)
            encoder_output1 = trained_model.encoder(x1)

            mu0, log_var0 = encoder_output0.embedding, encoder_output0.log_covariance
            mu1, log_var1 = encoder_output1.embedding, encoder_output1.log_covariance

            mu0 = mu0.detach().cpu()
            mu1 = mu1.detach().cpu()

            # draw vector to randomly shuffle
            shuffle_array = np.random.choice(range(x.shape[0]), x.shape[0], replace=False)

            # store
            metric_df_temp.loc[:x.shape[0] - 1, "contrast_id"] = 0
            metric_df_temp.loc[:x.shape[0] - 1, "sample_id"] = range(sample_iter, sample_iter + bs)

            metric_df_temp.loc[x.shape[0]:, "contrast_id"] = 1
            metric_df_temp.loc[x.shape[0]:, "sample_id"] = range(sample_iter, sample_iter + bs)

            metric_df_temp.loc[:, "train_cat"] = mode

            # calculate distance metrics
            latent_norm_factor = np.sqrt(mu0.shape[1])
            cos_d_all = np.diag(cosine_similarity(mu0, mu1))
            metric_df_temp.loc[:x.shape[0] - 1, "cos_all"] = cos_d_all
            metric_df_temp.loc[x.shape[0]:, "cos_all"] = cos_d_all

            euc_d_all = np.diag(euclidean_distances(mu0, mu1))
            metric_df_temp.loc[:x.shape[0] - 1, "euc_all"] = euc_d_all / latent_norm_factor
            metric_df_temp.loc[x.shape[0]:, "euc_all"] = euc_d_all / latent_norm_factor

            # compare to shuffled pairwise distances
            cos_d_all_sh = np.diag(cosine_similarity(mu0, mu1[shuffle_array, :]))
            metric_df_temp.loc[:x.shape[0] - 1, "cos_all_rand"] = cos_d_all_sh
            metric_df_temp.loc[x.shape[0]:, "cos_all_rand"] = cos_d_all_sh

            euc_d_all_sh = np.diag(euclidean_distances(mu0, mu1[shuffle_array, :]))
            metric_df_temp.loc[:x.shape[0] - 1, "euc_all)rand"] = euc_d_all_sh / latent_norm_factor
            metric_df_temp.loc[x.shape[0]:, "euc_all_rand"] = euc_d_all_sh / latent_norm_factor

            bio_indices = np.asarray([i for i in range(len(z_col_list)) if "z_mu_b_" in z_col_list[i]])
            nbio_indices = np.asarray([i for i in range(len(z_col_list)) if "z_mu_n_" in z_col_list[i]])
            if len(bio_indices) > 0:
                n_bio = np.sqrt(len(bio_indices))
                n_nbio = np.sqrt(len(nbio_indices))
                # biological partition
                cos_d_bio = np.diag(cosine_similarity(mu0[:, bio_indices], mu1[:, bio_indices]))
                metric_df_temp.loc[:x.shape[0] - 1, "cos_bio"] = cos_d_bio
                metric_df_temp.loc[x.shape[0]:, "cos_bio"] = cos_d_bio

                euc_d_bio = np.diag(euclidean_distances(mu0[:, bio_indices], mu1[:, bio_indices]))
                metric_df_temp.loc[:x.shape[0] - 1, "euc_bio"] = euc_d_bio / n_bio
                metric_df_temp.loc[x.shape[0]:, "euc_bio"] = euc_d_bio / n_bio

                cos_d_bio_sh = np.diag(cosine_similarity(mu0[:, bio_indices], mu1[:, bio_indices][shuffle_array, :]))
                metric_df_temp.loc[:x.shape[0] - 1, "cos_bio_rand"] = cos_d_bio_sh
                metric_df_temp.loc[x.shape[0]:, "cos_bio_rand"] = cos_d_bio_sh

                euc_d_bio_sh = np.diag(euclidean_distances(mu0[:, bio_indices], mu1[:, bio_indices][shuffle_array, :]))
                metric_df_temp.loc[:x.shape[0] - 1, "euc_bio_rand"] = euc_d_bio_sh / n_bio
                metric_df_temp.loc[x.shape[0]:, "euc_bio_rand"] = euc_d_bio_sh / n_bio

                # nuisance partition
                cos_d_nbio = np.diag(cosine_similarity(mu0[:, nbio_indices], mu1[:, nbio_indices]))
                metric_df_temp.loc[:x.shape[0] - 1, "cos_nbio"] = cos_d_nbio
                metric_df_temp.loc[x.shape[0]:, "cos_nbio"] = cos_d_nbio

                euc_d_nbio = np.diag(euclidean_distances(mu0[:, nbio_indices], mu1[:, nbio_indices]))
                metric_df_temp.loc[:x.shape[0] - 1, "euc_nbio"] = euc_d_nbio / n_nbio
                metric_df_temp.loc[x.shape[0]:, "euc_nbio"] = euc_d_nbio / n_nbio

                cos_d_nbio_sh = np.diag(cosine_similarity(mu0[:, nbio_indices], mu1[:, nbio_indices][shuffle_array, :]))
                metric_df_temp.loc[:x.shape[0] - 1, "cos_nbio_rand"] = cos_d_nbio_sh
                metric_df_temp.loc[x.shape[0]:, "cos_nbio_rand"] = cos_d_nbio_sh

                euc_d_nbio_sh = np.diag(euclidean_distances(mu0[:, nbio_indices], mu1[:, nbio_indices][shuffle_array, :]))
                metric_df_temp.loc[:x.shape[0] - 1, "euc_nbio_rand"] = euc_d_nbio_sh / n_nbio
                metric_df_temp.loc[x.shape[0]:, "euc_nbio_rand"] = euc_d_nbio_sh / n_nbio

            metric_df_temp.loc[:x.shape[0] - 1, z_col_list] = np.asarray(mu0)
            metric_df_temp.loc[x.shape[0]:, z_col_list] = np.asarray(mu1)

            metric_df_list.append(metric_df_temp)
            sample_iter += bs

    metric_df_out = pd.concat(metric_df_list, axis=0, ignore_index=True)

    meta_df["cos_all_mean"] = np.mean(metric_df_out["cos_all"])
    meta_df["euc_all_mean"] = np.mean(metric_df_out["euc_all"])
    meta_df["cos_all_mean_rand"] = np.mean(metric_df_out["cos_all_rand"])
    meta_df["euc_all_mean_rand"] = np.mean(metric_df_out["euc_all_rand"])

    if trained_model.model_name == "MetricVAE":
        meta_df["cos_bio_mean"] = np.mean(metric_df_out["cos_bio"])
        meta_df["euc_bio_mean"] = np.mean(metric_df_out["euc_bio"])
        meta_df["cos_bio_mean_rand"] = np.mean(metric_df_out["cos_bio_rand"])
        meta_df["euc_bio_mean_rand"] = np.mean(metric_df_out["euc_bio_rand"])
        meta_df["cos_nbio_mean"] = np.mean(metric_df_out["cos_nbio"])
        meta_df["euc_nbio_mean"] = np.mean(metric_df_out["euc_nbio"])
        meta_df["cos_nbio_mean_rand"] = np.mean(metric_df_out["cos_nbio_rand"])
        meta_df["euc_nbio_mean_rand"] = np.mean(metric_df_out["euc_nbio_rand"])

    return metric_df_out, meta_df

def bio_prediction_wrapper(embryo_df, meta_df):

    print("Training basic classifiers to test latent space information content...")
    mu_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_" in embryo_df.columns[i]]

    age_df = embryo_df.loc[:, ["snip_id", "predicted_stage_hpf", "train_cat", "master_perturbation"]].copy()

    y_pd_lin, y_score_lin, y_pd_nonlin, y_score_nonlin = get_embryo_age_predictions(embryo_df, mu_indices)

    age_df["stage_nonlinear_pd"] = y_pd_nonlin
    age_df["stage_linear_pd"] = y_pd_lin

    meta_df["stage_R2_nonlin_all"] = y_score_nonlin
    meta_df["stage_R2_lin_all"] = y_score_lin

    zmb_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_b" in embryo_df.columns[i]]
    zmn_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_n" in embryo_df.columns[i]]
    if len(zmb_indices) > 0:
        y_pd_lin_n, y_score_lin_n, y_pd_nonlin_n, y_score_nonlin_n = get_embryo_age_predictions(embryo_df,
                                                                                                zmn_indices)
        y_pd_lin_b, y_score_lin_b, y_pd_nonlin_b, y_score_nonlin_b = get_embryo_age_predictions(embryo_df,
                                                                                                zmb_indices)

        age_df["stage_nonlinear_pd_n"] = y_pd_nonlin_n
        age_df["stage_linear_pd_n"] = y_pd_lin_n
        age_df["stage_nonlinear_pd_b"] = y_pd_nonlin_b
        age_df["stage_linear_pd_b"] = y_pd_lin_b

        meta_df["stage_R2_nonlin_bio"] = y_score_nonlin_b
        meta_df["stage_R2_lin_bio"] = y_score_lin_b
        meta_df["stage_R2_nonlin_nbio"] = y_score_nonlin_n
        meta_df["stage_R2_lin_nbio"] = y_score_lin_n

    accuracy_nonlin, accuracy_lin, perturbation_df = get_pert_class_predictions(embryo_df, mu_indices)
    meta_df["perturbation_acc_nonlin_all"] = accuracy_nonlin
    meta_df["perturbation_acc_lin_all"] = accuracy_lin

    if len(zmb_indices) > 0:
        accuracy_nonlin_n, accuracy_lin_n, perturbation_df_n = get_pert_class_predictions(embryo_df, zmn_indices)
        accuracy_nonlin_b, accuracy_lin_b, perturbation_df_b = get_pert_class_predictions(embryo_df, zmb_indices)

        # subset
        perturbation_df_n = perturbation_df_n.loc[:, ["snip_id", "class_linear_pd", "class_nonlinear_pd"]]
        perturbation_df_n = perturbation_df_n.rename(
            columns={"class_linear_pd": "class_linear_pd_n", "class_nonlinear_pd": "class_nonlinear_pd_n"})

        perturbation_df_b = perturbation_df_b.loc[:, ["snip_id", "class_linear_pd", "class_nonlinear_pd"]]
        perturbation_df_b = perturbation_df_b.rename(
            columns={"class_linear_pd": "class_linear_pd_b", "class_nonlinear_pd": "class_nonlinear_pd_b"})

        perturbation_df = perturbation_df.merge(perturbation_df_n, how="left", on="snip_id")
        perturbation_df = perturbation_df.merge(perturbation_df_b, how="left", on="snip_id")

        meta_df["perturbation_acc_nonlin_bio"] = accuracy_nonlin_b
        meta_df["perturbation_acc_lin_bio"] = accuracy_lin_b

        meta_df["perturbation_acc_nonlin_nbio"] = accuracy_nonlin_n
        meta_df["perturbation_acc_lin_nbio"] = accuracy_lin_n

    return age_df, perturbation_df, meta_df

def initialize_assessment(train_dir, output_dir, mode_vec=None):

    if mode_vec is None:
        mode_vec = ["train", "eval", "test"]

    continue_flag = False

    # check device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
        )

    data_transform = make_dynamic_rs_transform() # use standard dataloader
    data_sampler_vec = []
    for mode in mode_vec:
        ds_temp = MyCustomDataset(
            root=os.path.join(train_dir, mode),
            transform=data_transform,
            return_name=True
        )
        temp_loader = DataLoader(
                        dataset=ds_temp,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=collate_dataset_output,
                    )
        data_sampler_vec.append(temp_loader)

    try:
        trained_model = AutoModel.load_from_folder(os.path.join(output_dir, 'final_model'))

        train_config_file = open(os.path.join(output_dir, 'final_model', 'training_config.json'))
        train_config = json.load(train_config_file)

        model_config_file = open(os.path.join(output_dir, 'final_model', 'model_config.json'))
        model_config = json.load(model_config_file)

    except:
        try:
            trained_model_list = glob.glob(os.path.join(output_dir, "*epoch*"))
            underscore_list = [s.rfind("_") for s in trained_model_list]
            epoch_num_list = [int(trained_model_list[s][underscore_list[s] + 1:]) for s in range(len(underscore_list))]
            last_ind = np.argmax(epoch_num_list)

            # last_training = path_leaf(trained_model_list[last_ind])
            trained_model = AutoModel.load_from_folder(trained_model_list[last_ind])

            train_config_file = open(os.path.join(trained_model_list[last_ind], 'training_config.json'))
            train_config = json.load(train_config_file)

            model_config_file = open(os.path.join(trained_model_list[last_ind], 'model_config.json'))
            model_config = json.load(model_config_file)

            print("No final model found for " + output_dir + ". Using most recent saved training instance.")
        except:
            print("No final model loaded for " + output_dir + ". \nEither there are no saved model directories, or an error occurred during loading")
            continue_flag = True
            trained_model = []

    meta_df = []
    if not continue_flag:

        # pass model to device
        trained_model = trained_model.to(device)

        training_keys_to_keep = ['name', 'output_dir', 'per_device_train_batch_size', 'per_device_eval_batch_size',
                                 'num_epochs', 'learning_rate']
        training_key_list = list(train_config.keys())
        meta_df = pd.DataFrame(np.empty((1, len(training_key_list))), columns=training_key_list)
        for k in training_key_list:
            if k in training_keys_to_keep:
                meta_df[k] = train_config[k]

        model_key_list = list(model_config.keys())
        model_keys_to_keep = ['input_dim', 'latent_dim', 'orth_flag', 'n_conv_layers', 'n_out_channels',
                              'reconstruction_loss', 'temperature', 'zn_frac', 'distance_metric', 'beta']
        for k in model_key_list:
            if k in model_keys_to_keep:
                entry = model_config[k]
                try:
                    meta_df[k] = entry
                except:
                    meta_df[k] = [entry]

    ############
    # Question 1: how well does it reproduce train, eval, and test images?
    ############

    figure_path = os.path.join(output_dir, "figures")

    # if "contrastive_learning_flag" in dir(trained_model):
    #     trained_model.contrastive_flag = False
    if "time_ignorance_flag" in dir(trained_model):
        trained_model.time_ignorance_flag = False
    if "class_ignorance_flag" in dir(trained_model):
        trained_model.class_ignorance_flag = False
    
    return trained_model, meta_df, figure_path, data_sampler_vec, continue_flag, device

if __name__ == "__main__":

    # root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\morphseq\\"

    batch_size = 64  # batch size to use generating latent encodings and image reconstructions
    overwrite_flag = True
    n_image_figures = 100  # make qualitative side-by-side reconstruction figures
    n_contrastive_samples = 1000  # number of images to reconstruct for loss calc
    skip_figures_flag = False
    train_name = "20231120_ds_small"
    architecture_name = "MetricVAE_z100_ne003_refactor_test"
    mode_vec = ["train", "eval", "test"]

    # set paths
    metadata_path = os.path.join(root, 'metadata', '')
    train_dir = os.path.join(root, "training_data", train_name, '')

    # get list of models in this folder
    models_to_assess = None  #["MetricVAE_training_2023-10-27_09-29-34"]

    if models_to_assess is None:
        models_to_assess = sorted(glob.glob(os.path.join(train_dir, architecture_name, '*VAE*')))

    for m_iter, model_name in enumerate(models_to_assess):

        embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df_final.csv"), index_col=0)
        # strip down the full dataset
        embryo_df = embryo_metadata_df[
            ["snip_id", "experiment_date", "medium", "master_perturbation", "predicted_stage_hpf", "surface_area_um",
             "length_um", "width_um"]].iloc[np.where(embryo_metadata_df["use_embryo_flag"] == 1)].copy()
        embryo_df = embryo_df.reset_index()

        # set path to output dir
        output_dir = os.path.join(train_dir, architecture_name, model_name)

        # initialize model assessment
        trained_model, meta_df, figure_path, data_sampler_vec, continue_flag, device = initialize_assessment(train_dir, output_dir)

        ########
        #  Skip if no model data or a previous assessment output exists and overwrite_flag==False
        if continue_flag:
            continue

        if not os.path.isdir(figure_path):
            os.makedirs(figure_path)

        prev_run_flag = os.path.isfile(os.path.join(figure_path, "embryo_stats_df.csv"))

        if prev_run_flag and overwrite_flag is False:
            print("Results already exist for: " + figure_path + ". Skipping.")
            continue

        print("Evaluating model " + model_name + f'({m_iter+1:02} of ' + str(len(models_to_assess)) + ')')

        np.random.seed(123)

        embryo_df = assess_image_reconstructions(embryo_df=embryo_df, trained_model=trained_model, figure_path=figure_path,
                                                 data_sampler_vec=data_sampler_vec, n_image_figures=n_image_figures,
                                                 device=device, skip_figures=skip_figures_flag)


        # Calculate UMAPs
        embryo_df = calculate_UMAPs(embryo_df)
        print(f"Saving data...")
        #save latent arrays and UMAP
        embryo_df = embryo_df.iloc[:, 1:]
        embryo_df.to_csv(os.path.join(figure_path, "embryo_stats_df.csv"))

        # make a narrower DF with just the UMAP cols and key metadata
        emb_cols = embryo_df.columns
        umap_cols = [col for col in emb_cols if "UMAP" in col]
        umap_df = embryo_df[
            ["snip_id", "experiment_date", "medium", "master_perturbation", "predicted_stage_hpf", "train_cat",
             "recon_mse"] + umap_cols].copy()
        umap_df.to_csv(os.path.join(figure_path, "umap_df.csv"))

        ############################################
        # Compare latent encodings of contrastive pairs
        if "z_mu_n_00" in embryo_df.columns:
            metric_df, meta_df = calculate_contrastive_distances(embryo_df, meta_df, trained_model, train_dir,
                                                                 device=device)#,
                                                                 # n_contrastive_samples=n_contrastive_samples)
            metric_df.to_csv(os.path.join(figure_path, "metric_df.csv"))

        # #########################################
        # Test how predictive latent space is of developmental age
        age_df, perturbation_df, meta_df = bio_prediction_wrapper(embryo_df, meta_df)

        age_df.to_csv(os.path.join(figure_path, "age_pd_df.csv"))
        perturbation_df.to_csv(os.path.join(figure_path, "perturbation_pd_df.csv"))

        meta_df["model_name"] = model_name
        meta_df.to_csv(os.path.join(figure_path, "meta_summary_df.csv"))

        print("Done.")


