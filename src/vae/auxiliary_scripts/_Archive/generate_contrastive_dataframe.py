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

def generate_contrastive_dataframe(root, train_name, architecture_name, overwrite_flag=False, batch_size=64, models_to_assess=None):
    mode_vec = ["train", "eval", "test"]

    # set paths
    metadata_path = os.path.join(root, 'metadata', '')
    train_dir = os.path.join(root, "training_data", train_name, '')

    # get list of models in this folder
    if models_to_assess is None:
        models_to_assess = sorted(glob.glob(os.path.join(train_dir, architecture_name, '*VAE*')))

    for m_iter, model_name in enumerate(models_to_assess):

        embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df_final.csv"), index_col=0)
        # strip down the full dataset
        contrastive_df = embryo_metadata_df[
            ["snip_id", "experiment_date", "medium", "master_perturbation", "predicted_stage_hpf"]].iloc[np.where(embryo_metadata_df["use_embryo_flag"] == 1)].copy()
        contrastive_df = contrastive_df.reset_index()

        # set path to output dir
        output_dir = os.path.join(train_dir, architecture_name, model_name)

        # initialize model assessment
        trained_model, figure_path, continue_flag, device = initialize_assessment(train_dir, output_dir, batch_size=batch_size)

        ########
        #  Skip if no model data or a previous assessment output exists and overwrite_flag==False
        if continue_flag:
            continue

        if not os.path.isdir(figure_path):
            os.makedirs(figure_path)

        prev_run_flag = os.path.isfile(os.path.join(figure_path, "contrastive_df.csv"))

        if prev_run_flag and overwrite_flag is False:
            print("Results already exist for: " + figure_path + ". Skipping.")
            continue

        print("Evaluating model " + model_name + f'({m_iter+1:02} of ' + str(len(models_to_assess)) + ')')

        np.random.seed(123)

        ############################################
        # Compare latent encodings of contrastive pairs
        latent_df = calculate_contrastive_distances(contrastive_df, trained_model, train_dir, device=device, batch_size=batch_size)#,

        # duplicate base contrastive DF and join on latent dimensions
        cdf0 = contrastive_df
        cdf1 = cdf0.copy()
        cdf0["contrast_id"] = 0
        cdf1["contrast_id"] = 1

        contrastive_df_final = pd.concat([cdf0, cdf1], axis=0, ignore_index=True)
        contrastive_df_final = contrastive_df_final.merge(latent_df, how="left", on=["snip_id", "contrast_id"])
        
        contrastive_df_final.to_csv(os.path.join(figure_path, "contrastive_df.csv"))

        print("Done.")

def calculate_contrastive_distances(contrastive_df, trained_model, train_dir, device, batch_size, mode_vec=None):

    if mode_vec is None:
        mode_vec = ["train", "eval", "test"]

    c_data_loader_vec = []
    # n_total_samples = 0
    for mode in mode_vec:
        temp_dataset = MyCustomDataset(root=os.path.join(train_dir, mode),
                                       transform=ContrastiveLearningViewGenerator(
                                           ContrastiveLearningDataset.get_simclr_pipeline_transform(),  # (96),
                                           2),
                                        return_name=True
                                       )
        data_loader = DataLoader(
            dataset=temp_dataset,
            batch_size=batch_size,
            collate_fn=collate_dataset_output,
        )

        c_data_loader_vec.append(data_loader)
        # n_total_samples += np.min([n_contrastive_samples, len(data_loader)])

    metric_df_list = []

    sample_iter = 0

    # contrastive_df = contrastive_df.reset_index()
    for m, mode in enumerate(mode_vec):
        data_loader = c_data_loader_vec[m]

        print(f"Calculating {mode} contrastive differences...")
        for i, inputs in enumerate(tqdm(data_loader)):
            inputs = set_inputs_to_device(device, inputs)
            x = inputs["data"]
            bs = x.shape[0]

            labels = list(inputs["label"][0])
            labels = labels * 2
            snip_id_list = clean_path_names(labels)

            # initialize temporary DF
            metric_df_temp = pd.DataFrame(np.empty((x.shape[0] * 2, 2)),
                                          columns=["snip_id", "contrast_id"])
            metric_df_temp["snip_id"] = snip_id_list

            # generate columns to store latent encodings
            new_cols = []
            for n in range(trained_model.latent_dim):

                if (trained_model.model_name == "MetricVAE") or (trained_model.model_name == "SeqVAE"):
                    if n in trained_model.nuisance_indices:
                        new_cols.append(f"z_mu_n_{n:02}")
                    else:
                        new_cols.append(f"z_mu_b_{n:02}")
                else:
                    raise Exceptionf("Incompatible model type found ({trained_model.model_name})")

            metric_df_temp.loc[:, new_cols] = np.nan

            # latent_encodings = trained_model.encoder(inputs)
            x0 = torch.reshape(x[:, 0, :, :, :],
                               (x.shape[0], x.shape[2], x.shape[3], x.shape[4]))  # first set of images
            x1 = torch.reshape(x[:, 1, :, :, :], 
                               (x.shape[0], x.shape[2], x.shape[3], x.shape[4]))  # second set with matched contrastive pairs

            encoder_output0 = trained_model.encoder(x0)
            encoder_output1 = trained_model.encoder(x1)

            mu0, log_var0 = encoder_output0.embedding, encoder_output0.log_covariance
            mu1, log_var1 = encoder_output1.embedding, encoder_output1.log_covariance

            mu0 = mu0.detach().cpu()
            mu1 = mu1.detach().cpu()

            # store
            metric_df_temp.loc[:x.shape[0] - 1, "contrast_id"] = 0
            metric_df_temp.loc[x.shape[0]:, "contrast_id"] = 1

            metric_df_temp.loc[:x.shape[0] - 1, new_cols] = np.asarray(mu0)
            metric_df_temp.loc[x.shape[0]:, new_cols] = np.asarray(mu1)

            metric_df_list.append(metric_df_temp)
            # sample_iter += bs

    metric_df_out = pd.concat(metric_df_list, axis=0, ignore_index=True)

    return metric_df_out


def initialize_assessment(train_dir, output_dir, batch_size, mode_vec=None):

    if mode_vec is None:
        mode_vec = ["train", "eval", "test"]

    continue_flag = False

    # check device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
        )

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

    if not continue_flag:
        # pass model to device
        trained_model = trained_model.to(device)

    figure_path = os.path.join(output_dir, "figures")

    
    return trained_model, figure_path, continue_flag, device



if __name__ == "__main__":

    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"

    batch_size = 64  # batch size to use generating latent encodings and image reconstructions
    overwrite_flag = True
    n_image_figures = 100  # make qualitative side-by-side reconstruction figures
    skip_figures_flag = False
    train_name = "20231106_ds"
    architecture_name_vec = ["SeqVAE_z100_ne250_gamma_temp_self_and_other"] #, "SeqVAE_z100_ne250_gamma_temp_SELF_ONLY", "SeqVAE_z100_ne250_triplet_loss_test_self_and_other"]
    # mode_vec = ["train", "eval", "test"]

    models_to_assess = None #["SeqVAE_training_2023-12-12_23-56-02"]

    for architecture_name in architecture_name_vec:
        generate_contrastive_dataframe(root, train_name, architecture_name, overwrite_flag=True, batch_size=64, models_to_assess=models_to_assess)

    


