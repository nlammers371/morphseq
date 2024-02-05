import glob as glob
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from src.functions.dataset_utils import *
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import ntpath


def infer_developmental_age(root, train_name, architecture_name, model_name, reference_perturbations=None, reference_datasets=None):

    if reference_perturbations is None:
        reference_perturbations =["wik", "wik-inj-ctrl", "wik-ctrl-inj"]

    # set path to output dir
    train_dir = os.path.join(root, "training_data", train_name, '')
    output_dir = os.path.join(train_dir, architecture_name, model_name)
    data_path = os.path.join(output_dir, "figures")
    metadata_path = os.path.join(root, "metadata")

    embryo_df = pd.read_csv(os.path.join(data_path, "embryo_stats_df.csv"), index_col=0)
    metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df_final.csv"), index_col=0)
    metadata_df = metadata_df.loc[:, ["snip_id", "temperature", "embryo_id", "Time Rel (s)"]]
    # add temperature and 
    embryo_df = embryo_df.merge(metadata_df, on="snip_id", how="left")
    embryo_df = get_embryo_age_predictions(embryo_df, reference_perturbations=reference_perturbations, reference_datasets=reference_datasets)


    # make a lightweight age key
    age_key_df = embryo_df.loc[:, ["snip_id", "experiment_date", "embryo_id", "temperature", "predicted_stage_hpf", "inferred_stage_hpf_reg", "Time Rel (s)", "master_perturbation"]]
    age_key_df["Time Rel (s)"] = age_key_df["Time Rel (s)"] / 3600
    age_key_df.rename(columns={"predicted_stage_hpf":"calc_stage_hpf", "Time Rel (s)":"abs_time_hr"}, inplace=True)
    age_key_df["train_dir"] = train_name
    age_key_df["model_name"] = model_name
    age_key_df["architecture_name"] = architecture_name
    # snip_id_vec = age_key_df["snip_id"]
    # embryo_id_vec = [snip[:-10] for snip in snip_id_vec]
    # age_key_df["embryo_id"] = embryo_id_vec
    # embryo_id_index = np.unique(age_key_df["embryo_id"])

    # for e, embryo_id in enumerate(embryo_id_index):
    #     embryo_indices = np.where(age_key_df["embryo_id"] == embryo_id)[0]
    #     age_vec = age_key_df.loc[embryo_indices, "inferred_stage_hpf_reg"].to_numpy()
    #     if len(embryo_indices) > 2:
    #         embryo_vec = np.arange(len(embryo_indices))
    #         reg = LinearRegression().fit(embryo_vec[:, np.newaxis], age_vec[:, np.newaxis])
    #         age_pd = reg.predict(embryo_vec[:, np.newaxis])
    #         age_key_df.loc[embryo_indices, "inferred_stage_hpf_reg"] = age_pd
    #     else:
    #         age_key_df.loc[embryo_indices, "inferred_stage_hpf_reg"] = age_vec
    # finally, fit linear models to each embryo for internal constency


    age_key_df.to_csv(os.path.join(metadata_path, "age_key_df.csv"))

    return age_key_df



def get_embryo_age_predictions(embryo_df, reference_perturbations, reference_datasets):

    # get indices of latent var columns
    mu_indices = [i for i in range(embryo_df.shape[1]) if "z_mu" in embryo_df.columns[i]]

    pert_vec_bool = np.asarray([embryo_df.loc[i, "master_perturbation"] in reference_perturbations for i in embryo_df.index])
    if reference_datasets is not None:
        data_vec_bool = np.asarray([embryo_df.loc[i, "experiment_date"].astype(str) in reference_datasets for i in embryo_df.index])
    else:
        data_vec_bool = np.full(pert_vec_bool.shape, True)

    train_indices = embryo_df.index[data_vec_bool & pert_vec_bool]
    # test_indices = embryo_df.index[~(data_vec_bool & pert_vec_bool)]

    # extract target vector
    y_train = embryo_df["predicted_stage_hpf"].iloc[train_indices].to_numpy().astype(float)
    # y_test = embryo_df["predicted_stage_hpf"].iloc[test_indices].to_numpy().astype(float)

    # extract predictor variables
    X_train = embryo_df.iloc[train_indices, mu_indices].to_numpy().astype(float)
    # X_test = embryo_df.iloc[test_indices, mu_indices].to_numpy().astype(float)

    ###################
    # run MLP regressor
    print("Fitting stage prediction model...")
    clf_age_nonlin = MLPRegressor(random_state=1, max_iter=5000).fit(X_train, y_train)

    ###################
    # get predictions across all datasets for the matching phenotypes
    # this will create maps from absolute experimental time to developmental time
    X_ref = embryo_df.iloc[np.where(pert_vec_bool)[0], mu_indices].to_numpy().astype(float)

    y_ref_pd = clf_age_nonlin.predict(X_ref)

    # make stage lookup DF
    stage_lookup_df = embryo_df.loc[pert_vec_bool, ["snip_id", "experiment_date", "master_perturbation", "predicted_stage_hpf", "temperature", "embryo_id", "Time Rel (s)"]]
    stage_lookup_df["Time Rel (s)"] = stage_lookup_df["Time Rel (s)"] / 3600
    stage_lookup_df.rename(columns={"predicted_stage_hpf":"calc_stage_hpf", "Time Rel (s)":"abs_time_hr"}, inplace=True)
    stage_lookup_df["inferred_stage_hpf_reg"] = y_ref_pd

    # map stage info to remaining embryos (mutant phenotypes)
    n_ref = 5
    max_stage_delta = 2

    date_index = np.unique(embryo_df["experiment_date"])
    embryo_df["inferred_stage_hpf_reg"] = np.nan
    for d, date in enumerate(tqdm(date_index)):

        # get indexing vectors
        ref_bool_vec = stage_lookup_df["experiment_date"]==date
        to_index_vec = embryo_df.index[embryo_df["experiment_date"]==date]
        exp_temperature = embryo_df.loc[to_index_vec[0], "temperature"]
        temp_ref_bool_vec = stage_lookup_df["temperature"] == exp_temperature
        
        # get stage ref vectors
        date_calc_stage_vec = stage_lookup_df.loc[ref_bool_vec, "calc_stage_hpf"].to_numpy()
        temp_calc_stage_vec = stage_lookup_df.loc[temp_ref_bool_vec, "calc_stage_hpf"].to_numpy()

        date_pd_stage_vec = stage_lookup_df.loc[ref_bool_vec, "inferred_stage_hpf_reg"].to_numpy() 
        temp_pd_stage_vec = stage_lookup_df.loc[temp_ref_bool_vec, "inferred_stage_hpf_reg"].to_numpy() 

        for to_ind in to_index_vec:
            calc_stage = embryo_df.loc[to_ind, "predicted_stage_hpf"]
            stage_diffs = np.abs(date_calc_stage_vec - calc_stage)
            if np.sum(stage_diffs <= max_stage_delta) >= n_ref:
                ref_indices = np.where(stage_diffs<=max_stage_delta)[0]
                ref_calc_stage = date_calc_stage_vec[ref_indices]
                ref_pd_stage = date_pd_stage_vec[ref_indices]

            else: # if not enough comps, use lookups from other experiments at the same temperature
                stage_diffs = np.abs(temp_calc_stage_vec - calc_stage)
                ref_indices = np.where(stage_diffs<=max_stage_delta)[0]
                if len(ref_indices) < n_ref:
                    option_indices = np.argsort(stage_diffs)
                    ref_indices = option_indices[:n_ref]
                ref_calc_stage = temp_calc_stage_vec[ref_indices]
                ref_pd_stage = temp_pd_stage_vec[ref_indices]

             # fit linear regression
            reg = LinearRegression(fit_intercept=False).fit(ref_calc_stage[:, np.newaxis], ref_pd_stage[:, np.newaxis])
            age_pd = reg.predict(np.asarray([calc_stage])[:, np.newaxis])
            embryo_df.loc[to_ind, "inferred_stage_hpf_reg"] = age_pd[0][0]
    # y_score_nonlin = clf_age_nonlin.score(X_test, y_test)

    return embryo_df


if __name__ == "__main__":

    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
    train_name = "20240204_ds_v2"
    architecture_name = "VAE_z100_ne250_vanilla_VAE"
    model_name = "VAE_training_2024-02-04_13-54-24"
    reference_datasets=["20231110", "20231206", "20231218"]
    infer_developmental_age(root, train_name, architecture_name, model_name, reference_datasets=reference_datasets)

    