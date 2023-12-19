import glob as glob
from sklearn.neural_network import MLPRegressor
from src.functions.dataset_utils import *
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
# from pythae.trainers.base_trainer_verbose import base_trainer_verbose
import ntpath


def infer_developmental_age(root, train_name, architecture_name, model_name):

    # set path to output dir
    train_dir = os.path.join(root, "training_data", train_name, '')
    output_dir = os.path.join(train_dir, architecture_name, model_name)
    
    data_path = os.path.join(output_dir, "figures")

    embryo_df = pd.read_csv(os.path.join(data_path, "embryo_stats_df.csv"), index_col=0)

    age_pd_nonlin, age_score_nonlin = get_embryo_age_predictions(embryo_df)

    embryo_df.loc[:, "inferred_stage_hpf"] = age_pd_nonlin

    # make a lightweight age key
    age_key_df = embryo_df.loc[:, ["snip_id", "predicted_stage_hpf", "inferred_stage_hpf"]]
    snip_id_vec = age_key_df["snip_id"]
    embryo_id_vec = [snip[:-10] for snip in snip_id_vec]
    age_key_df["embryo_id"] = embryo_id_vec
    embryo_id_index = np.unique(age_key_df["embryo_id"])

    for e, embryo_id in enumerate(embryo_id_index):
        embryo_indices = np.where(age_key_df["embryo_id"] == embryo_id)[0]
        age_vec = age_key_df.loc[embryo_indices, "inferred_stage_hpf"].to_numpy()
        if len(embryo_indices) > 2:
            embryo_vec = np.arange(len(embryo_indices))
            reg = LinearRegression().fit(embryo_vec[:, np.newaxis], age_vec[:, np.newaxis])
            age_pd = reg.predict(embryo_vec[:, np.newaxis])
            age_key_df.loc[embryo_indices, "inferred_stage_hpf_reg"] = age_pd
        else:
            age_key_df.loc[embryo_indices, "inferred_stage_hpf_reg"] = age_vec
    # finally, fit linear models to each embryo for internal constency


    age_key_df.to_csv(os.path.join(data_path, "age_key_df.csv"))

    return age_key_df



def get_embryo_age_predictions(embryo_df):

    mu_indices = [i for i in range(embryo_df.shape[1]) if "z_mu" in embryo_df.columns[i]]

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
    # initialize pandas dataframe to store results
    X_full = embryo_df.iloc[:, mu_indices].to_numpy().astype(float)

    y_pd_nonlin = clf_age_nonlin.predict(X_full)
    y_score_nonlin = clf_age_nonlin.score(X_test, y_test)

    return y_pd_nonlin, y_score_nonlin

    