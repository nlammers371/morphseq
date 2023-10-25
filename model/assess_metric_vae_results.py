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
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from pythae.trainers.base_trainer_verbose import base_trainer_verbose
from functions.ContrastiveLearningDataset import ContrastiveLearningDataset
from functions.view_generator import ContrastiveLearningViewGenerator
from pythae.data.datasets import collate_dataset_output
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

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

def get_gdf3_class_predictions(embryo_df, mu_indices):

    train_indices = np.where((embryo_df["train_cat"] == "train") | (embryo_df["train_cat"] == "eval"))[0]
    test_indices = np.where(embryo_df["train_cat"] == "test")[0]

    gdf3_df = embryo_df.loc[:, ["snip_id", "predicted_stage_hpf", "train_cat", "master_perturbation"]].copy()

    ########################
    # How well does latent space predict perturbation type?
    pert_class_train = np.asarray(embryo_df["master_perturbation"].iloc[train_indices])
    train_gdf3_sub_indices = np.where(pert_class_train == "gdf3")[0]
    # train_shh_sub_indices = np.where((pert_class_train == "shh_100") | (pert_class_train == "shh_75") | (pert_class_train == "shh_50"))[0]
    train_wik_sub_indices = np.random.choice(np.where(pert_class_train == "wck-AB")[0], len(train_gdf3_sub_indices),
                                             replace=False)
    train_sub_indices = np.asarray(train_gdf3_sub_indices.tolist() + train_wik_sub_indices.tolist())

    pert_class_test = np.asarray(embryo_df["master_perturbation"].iloc[test_indices])
    test_sub_indices = np.where((pert_class_test == "wck-AB") | (pert_class_test == "gdf3"))[0]

    # extract predictor variables
    X_train = embryo_df.iloc[train_indices, mu_indices].to_numpy().astype(float)
    X_test = embryo_df.iloc[test_indices, mu_indices].to_numpy().astype(float)

    ###################
    # run MLP classifier
    ###################
    clf = MLPClassifier(random_state=1, max_iter=5000).fit(X_train[train_sub_indices],
                                                           pert_class_train[train_sub_indices])
    accuracy_nonlin = clf.score(X_test[test_sub_indices], pert_class_test[test_sub_indices])

    ###################
    # Run multivariate logistic classifier
    ###################
    clf_lin = LogisticRegression(random_state=0).fit(X_train[train_sub_indices],
                                                     pert_class_train[train_sub_indices])
    accuracy_lin = clf_lin.score(X_test[test_sub_indices], pert_class_test[test_sub_indices])

    class_pd_nonlin_train = clf.predict(X_train[train_sub_indices, :])
    class_pd_nonlin_test = clf.predict(X_test[test_sub_indices, :])

    class_pd_lin_train = clf_lin.predict(X_train[train_sub_indices, :])
    class_pd_lin_test = clf_lin.predict(X_test[test_sub_indices, :])

    gdf3_df_train = gdf3_df.iloc[train_indices[train_sub_indices]]
    gdf3_df_test = gdf3_df.iloc[test_indices[test_sub_indices]]

    gdf3_df_train.loc[:, "class_nonlinear_pd"] = class_pd_nonlin_train
    gdf3_df_train.loc[:, "class_linear_pd"] = class_pd_lin_train

    gdf3_df_test.loc[:, "class_nonlinear_pd"] = class_pd_nonlin_test
    gdf3_df_test.loc[:, "class_linear_pd"] = class_pd_lin_test

    gdf3_df = pd.concat([gdf3_df_train, gdf3_df_test], axis=0, ignore_index=True)

    return accuracy_nonlin, accuracy_lin, gdf3_df

if __name__ == "__main__":

    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
    batch_size = 128  # batch size to use generating latent encodings and image reconstructions
    overwrite_flag = True
    main_dims = (576, 256)
    n_image_figures = 100  # make qualitative side-by-side figures
    n_contrastive_samples = 1000  # number of images to reconstruct for loss calc
    test_contrastive_pairs = True

    # load metadata
    metadata_path = os.path.join(root, 'metadata', '')

    train_name = "20230804_vae_test" #"20230915_vae"
    train_dir = os.path.join(root, "training_data", train_name, '')
    # get list of models in this folder
    model_name_list = sorted(glob.glob(train_dir + '*metric_test*'))

    for m_iter, model_name in enumerate(model_name_list):

        embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df_final.csv"), index_col=0)
        embryo_df = embryo_metadata_df[
            ["snip_id", "experiment_date", "medium", "master_perturbation", "predicted_stage_hpf", "surface_area_um",
             "length_um", "width_um"]].iloc[np.where(embryo_metadata_df["use_embryo_flag"] == 1)].copy()
        embryo_df = embryo_df.reset_index()

        output_dir = os.path.join(train_dir, model_name) #"/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/training_data/20230807_vae_test/"


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
        try:
            trained_model = AutoModel.load_from_folder(
                os.path.join(output_dir, last_training, 'final_model'))
        except:
            try:
                trained_model_list = glob.glob(os.path.join(output_dir, last_training, "*epoch*"))
                underscore_list = [s.rfind("_") for s in trained_model_list]
                epoch_num_list = [int(trained_model_list[s][underscore_list[s]+1:]) for s in range(len(underscore_list))]
                last_ind = np.argmax(epoch_num_list)

                # last_training = path_leaf(trained_model_list[last_ind])
                trained_model = AutoModel.load_from_folder(trained_model_list[last_ind])
                print("No final model found for " + output_dir + ". Using most recent saved training isntance.")
            except:
                print("No final model for " + output_dir + ". Still training?")
                continue
        ############
        # Question 1: how well does it reproduce train, eval, and test images?
        ############

        figure_path = os.path.join(output_dir, last_training, "figures")
        if not os.path.isdir(figure_path):
            os.makedirs(figure_path)

        prev_run_flag = os.path.isfile(os.path.join(figure_path, "embryo_stats_df.csv"))


        if prev_run_flag and overwrite_flag==False:
            print("Results already exist for: " + figure_path + ". Skipping.")
            continue

        print("Evaluating model " + model_name + f'({m_iter:02} of ' + str(len(model_name_list)) + ')')
        # print("Saving to: " + figure_path)

        np.random.seed(123)

        # initialize new columns
        embryo_df["train_cat"] = ''
        embryo_df["recon_mse"] = np.nan
        snip_id_vec = embryo_df["snip_id"]

        print("Making image figures...")
        for m, mode in enumerate(mode_vec):

            # make subdir for images
            image_path = os.path.join(figure_path, mode + "_images")
            if not os.path.isdir(image_path):
                os.makedirs(image_path)

            data_sampler = data_sampler_vec[m]
            n_images = len(data_sampler)
            n_image_figs = np.min([n_images, n_image_figures])
            # n_recon_samples = n_images #np.min([n_images, n_images_to_sample])

            # draw random samples
            sample_indices = np.random.choice(range(n_images), n_images, replace=False)
            figure_indices = np.random.choice(range(n_images), n_image_figs, replace=False)
            batch_id_vec = []
            n_batches = np.ceil(len(sample_indices)/batch_size).astype(int)
            for n in range(n_batches):
                ind1 = n*batch_size
                ind2 = (n+1)*batch_size
                batch_id_vec.append(sample_indices[ind1:ind2])

            # recon_loss_array = np.empty((n_recon_samples,))

            print("Scoring image reconstructions for " + mode + " images...")
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
                    snip_index_vec.append(np.where(snip_name == snip_id_vec)[0][0])

                    im_stack[b, :, :] = im_raw

                im_test = torch.reshape(torch.from_numpy(im_stack), (len(batch_ids), 1, main_dims[0], main_dims[1]))
                im_recon = trained_model.reconstruct(im_test).detach().cpu()

                recon_loss = F.mse_loss(
                    im_recon.reshape(im_test.shape[0], -1),
                    im_test.reshape(im_test.shape[0], -1),
                    reduction="none",
                ).sum(dim=-1)
                # recon_loss_array[i] = recon_loss
                for b in range(len(batch_ids)):
                    embryo_df.loc[snip_index_vec[b], "train_cat"] = mode
                    embryo_df.loc[snip_index_vec[b], "recon_mse"] = np.asarray(recon_loss)[b]

                    if batch_ids[b] in figure_indices:
                        # show results with normal sampler
                        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

                        axes[0].imshow(np.squeeze(im_test[b, :, :]), cmap='gray')
                        axes[0].axis('off')

                        axes[1].imshow(np.squeeze(im_recon[b, :, :]), cmap='gray')
                        axes[1].axis('off')

                        plt.tight_layout(pad=0.)

                        plt.savefig(os.path.join(image_path, snip_name_vec[b] + f'_loss{int(np.round(recon_loss[b],0)):05}.tiff'))
                        plt.close()

        embryo_df = embryo_df.dropna(ignore_index=True)
        snip_id_vec = embryo_df["snip_id"]

        ############
        # Question 2: what does latent space look like?
        ############
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

        embryo_df[new_cols] = np.nan
        print("Calculating latent embeddings...")
        # embryo_df = embryo_df.reset_index()
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
                    snip_index_vec.append(np.where(snip_name == snip_id_vec)[0][0])

                    im_stack[b, :, :] = im_raw

                im_test = torch.reshape(torch.from_numpy(im_stack), (len(batch_ids), 1, main_dims[0], main_dims[1]))
                encoder_out = trained_model.encoder(im_test)
                zm_vec = np.asarray(encoder_out[0].detach())
                zs_vec = np.asarray(encoder_out[1].detach())
                snip_ind_array = np.asarray(snip_index_vec)
                for z in range(trained_model.latent_dim):
                    if trained_model.model_name == "MetricVAE":
                        if z in trained_model.nuisance_indices:
                            embryo_df.loc[snip_ind_array, f"z_mu_n_{z:02}"] = zm_vec[:, z]
                            embryo_df.loc[snip_ind_array, f"z_sigma_n_{z:02}"] = zs_vec[:, z]
                        else:
                            embryo_df.loc[snip_ind_array, f"z_mu_b_{z:02}"] = zm_vec[:, z]
                            embryo_df.loc[snip_ind_array, f"z_sigma_b_{z:02}"] = zs_vec[:, z]
                    else:
                        embryo_df.loc[snip_ind_array, f"z_mu_{z:02}"] = zm_vec[:, z]
                        embryo_df.loc[snip_ind_array, f"z_sigma_{z:02}"] = zs_vec[:, z]

                # z_mu_array[n, :] = np.asarray(encoder_out[0].detach())
                # z_sigma_array[n, :] = np.asarray(np.exp(encoder_out[1].detach()/2))


        zm_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_" in embryo_df.columns[i]]
        z_mu_array = embryo_df.iloc[:, zm_indices].to_numpy()

        print(f"Calculating UMAP...")
        # calculate 2D morphology UMAPS
        reducer = umap.UMAP()
        scaled_z_mu = StandardScaler().fit_transform(z_mu_array)
        embedding2d = reducer.fit_transform(scaled_z_mu)
        embryo_df.loc[:, "UMAP_00"] = embedding2d[:, 0]
        embryo_df.loc[:, "UMAP_01"] = embedding2d[:, 1]

        if trained_model.model_name == "MetricVAE":
            reducer_bio = umap.UMAP()
            embedding2d_bio = reducer.fit_transform(scaled_z_mu[:, trained_model.biological_indices])
            embryo_df.loc[:, "UMAP_00_bio"] = embedding2d_bio[:, 0]
            embryo_df.loc[:, "UMAP_01_bio"] = embedding2d_bio[:, 1]

            reducer_n = umap.UMAP()
            embedding2d_n = reducer.fit_transform(scaled_z_mu[:, trained_model.nuisance_indices])
            embryo_df.loc[:, "UMAP_00_n"] = embedding2d_n[:, 0]
            embryo_df.loc[:, "UMAP_01_n"] = embedding2d_n[:, 1]

        print(f"Saving data...")
        #save latent arrays and UMAP
        embryo_df = embryo_df.iloc[:, 1:]
        embryo_df.to_csv(os.path.join(figure_path, "embryo_stats_df.csv"))

        ############################################
        # Compare latent encodings of contrastive pairs
        if test_contrastive_pairs:
            c_data_loader_vec = []
            n_total_samples = 0
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
                n_total_samples += np.min([n_contrastive_samples, len(data_loader)])


            metric_df_list = []
            zm_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_" in embryo_df.columns[i]]
            z_col_list = embryo_df.columns[zm_indices].to_list()
            sample_iter = 0


            # embryo_df = embryo_df.reset_index()
            for m, mode in enumerate(mode_vec):
                data_loader = c_data_loader_vec[m]

                print(f"Calculating {mode} contrastive differences...")
                for i, inputs in enumerate(tqdm(data_loader)):
                    # inputs = self._set_inputs_to_device(inputs)
                    x = inputs.data
                    bs = x.shape[0]

                    metric_df_temp = pd.DataFrame(np.empty((x.shape[0]*2, trained_model.latent_dim + 3 + 2)),
                                     columns=["sample_id", "contrast_id", "train_cat", "euc_all", "cos_all"] + z_col_list)

                    # latent_encodings = trained_model.encoder(inputs)
                    x0 = torch.reshape(x[:, 0, :, :, :],
                                       (x.shape[0], x.shape[2], x.shape[3], x.shape[4]))  # first set of images
                    x1 = torch.reshape(x[:, 1, :, :, :], (
                        x.shape[0], x.shape[2], x.shape[3], x.shape[4]))  # second set with matched contrastive pairs

                    encoder_output0 = trained_model.encoder(x0)
                    encoder_output1 = trained_model.encoder(x1)

                    mu0, log_var0 = encoder_output0.embedding, encoder_output0.log_covariance
                    mu1, log_var1 = encoder_output1.embedding, encoder_output1.log_covariance

                    # store
                    metric_df_temp.loc[:x.shape[0]-1, "contrast_id"] = 0
                    metric_df_temp.loc[:x.shape[0]-1, "sample_id"] = range(sample_iter, sample_iter+bs)

                    metric_df_temp.loc[x.shape[0]:, "contrast_id"] = 1
                    metric_df_temp.loc[x.shape[0]:, "sample_id"] = range(sample_iter, sample_iter + bs)

                    metric_df_temp.loc[:, "train_cat"] = mode

                    # calculate distance metrics
                    cos_d_all = np.diag(cosine_similarity(mu0.detach(), mu1.detach()))
                    metric_df_temp.loc[:x.shape[0] - 1, "cos_all"] = cos_d_all
                    metric_df_temp.loc[x.shape[0]:, "cos_all"] = cos_d_all

                    euc_d_all = np.diag(euclidean_distances(mu0.detach(), mu1.detach()))
                    metric_df_temp.loc[:x.shape[0] - 1, "euc_all"] = euc_d_all
                    metric_df_temp.loc[x.shape[0]:, "euc_all"] = euc_d_all

                    if trained_model.model_name == "MetricVAE":

                        bio_indices = np.asarray([i for i in range(len(z_col_list)) if "z_mu_b_" in z_col_list[i]])
                        nbio_indices = np.asarray([i for i in range(len(z_col_list)) if "z_mu_n_" in z_col_list[i]])

                        # biological partition
                        cos_d_bio = np.diag(cosine_similarity(mu0.detach()[:, bio_indices], mu1.detach()[:, bio_indices]))
                        metric_df_temp.loc[:x.shape[0] - 1, "cos_bio"] = cos_d_bio
                        metric_df_temp.loc[x.shape[0]:, "cos_bio"] = cos_d_bio

                        euc_d_bio = np.diag(euclidean_distances(mu0.detach()[:, bio_indices], mu1.detach()[:, bio_indices]))
                        metric_df_temp.loc[:x.shape[0] - 1, "euc_bio"] = euc_d_bio
                        metric_df_temp.loc[x.shape[0]:, "euc_bio"] = euc_d_bio

                        # nuisance partition
                        cos_d_nbio = np.diag(cosine_similarity(mu0.detach()[:, nbio_indices], mu1.detach()[:, nbio_indices]))
                        metric_df_temp.loc[:x.shape[0] - 1, "cos_nbio"] = cos_d_bio
                        metric_df_temp.loc[x.shape[0]:, "cos_nbio"] = cos_d_bio

                        euc_d_nbio = np.diag(euclidean_distances(mu0.detach()[:, nbio_indices], mu1.detach()[:, nbio_indices]))
                        metric_df_temp.loc[:x.shape[0] - 1, "euc_nbio"] = euc_d_nbio
                        metric_df_temp.loc[x.shape[0]:, "euc_nbio"] = euc_d_nbio


                    metric_df_temp.loc[:x.shape[0] - 1, z_col_list] = np.asarray(mu0.detach())
                    metric_df_temp.loc[x.shape[0]:, z_col_list] = np.asarray(mu1.detach())

                    metric_df_list.append(metric_df_temp)
                    sample_iter += bs

        metric_df_out = pd.concat(metric_df_list, axis=0, ignore_index=True)
        metric_df_out.to_csv(os.path.join(figure_path, "metric_df.csv"))

         # #########################################
        # Test how predictive latent space is of developmental age
        print("Training basic classifiers to test latent space information content...")
        mu_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_" in embryo_df.columns[i]]

        age_df = embryo_df.loc[:, ["snip_id", "predicted_stage_hpf", "train_cat", "master_perturbation"]].copy()

        y_pd_lin, y_score_lin, y_pd_nonlin, y_score_nonlin = get_embryo_age_predictions(embryo_df, mu_indices)

        age_df["stage_nonlinear_pd"] = y_pd_nonlin
        age_df["stage_linear_pd"] = y_pd_lin

        if trained_model.model_name == "MetricVAE":
            zmb_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_b" in embryo_df.columns[i]]
            zmn_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_n" in embryo_df.columns[i]]

            y_pd_lin_n, y_score_lin_n, y_pd_nonlin_n, y_score_nonlin_n = get_embryo_age_predictions(embryo_df,
                                                                                                    zmn_indices)
            y_pd_lin_b, y_score_lin_b, y_pd_nonlin_b, y_score_nonlin_b = get_embryo_age_predictions(embryo_df,
                                                                                                    zmb_indices)

            age_df["stage_nonlinear_pd_n"] = y_pd_nonlin_n
            age_df["stage_linear_pd_n"] = y_pd_lin_n
            age_df["stage_nonlinear_pd_b"] = y_pd_nonlin_b
            age_df["stage_linear_pd_b"] = y_pd_lin_b

        accuracy_nonlin, accuracy_lin, gdf3_df = get_gdf3_class_predictions(embryo_df, mu_indices)

        if trained_model.model_name == "MetricVAE":
            accuracy_nonlin_n, accuracy_lin_n, gdf3_df_n = get_gdf3_class_predictions(embryo_df, zmn_indices)
            accuracy_nonlin_b, accuracy_lin_b, gdf3_df_b = get_gdf3_class_predictions(embryo_df, zmb_indices)

            # subset
            gdf3_df_n = gdf3_df_n.loc[:, ["snip_id", "class_linear_pd", "class_nonlinear_pd"]]
            gdf3_df_n.rename({"class_linear_pd": "class_linear_pd_n", "class_nonlinear_pd": "class_nonlinear_pd_n"})

            gdf3_df_b = gdf3_df_b.loc[:, ["snip_id", "class_linear_pd", "class_nonlinear_pd"]]
            gdf3_df_b.rename({"class_linear_pd": "class_linear_pd_b", "class_nonlinear_pd": "class_nonlinear_pd_b"})

            gdf3_df = gdf3_df.merge(gdf3_df_n, how="left", on="snip_id")
            gdf3_df = gdf3_df.merge(gdf3_df_b, how="left", on="snip_id")


        age_df.to_csv(os.path.join(figure_path, "age_pd_df.csv"))
        gdf3_df.to_csv(os.path.join(figure_path, "gdf3_pd_df.csv"))

        print("Done.")


