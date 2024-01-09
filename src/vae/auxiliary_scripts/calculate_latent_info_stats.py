import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import time 
from glob2 import glob
from tqdm import tqdm


def calculate_latent_info_stats(root, train_name, model_name):
    train_dir = os.path.join(root, "training_data", train_name)
    output_dir = os.path.join(train_dir, model_name) 

    print("Evaluating models from: " + output_dir)

    # get list of models to load
    model_list = sorted(glob(os.path.join(output_dir, "*VAE*")))

    contrastive_df_list = []
    embryo_df_list = []
    mdl_path_list = []
    meta_df_list = []
    print("Loading trained models...")
    for m, mdl in enumerate(tqdm(model_list)):
        
        # get path to output dataframes
        mdl_path = os.path.join(output_dir, mdl, "figures")
        mdl_path_list.append(mdl_path)
        
        # load metadata 
        try:
            meta_df = pd.read_csv(os.path.join(mdl_path, "meta_summary_df.csv"), index_col=0)
            meta_df_list.append(meta_df)

            if "temperature" in meta_df.columns:
                temperature = meta_df["temperature"].values
            else: 
                temperature = [np.nan]

            if "gamma" in meta_df.columns:
                gamma = meta_df["gamma"].values
            else: 
                gamma = [np.nan]

            # load contrastive DF
            contrastive_df = pd.read_csv(os.path.join(mdl_path, "contrastive_df.csv"), index_col=0)
            contrastive_df["mdl_id"] = m
            contrastive_df["temperature"] = temperature[0]
            contrastive_df["gamma"] = gamma[0]

            contrastive_df_list.append(contrastive_df)
            
            # load embryo df
            embryo_df = pd.read_csv(os.path.join(mdl_path, "embryo_stats_df.csv"), index_col=0)
            embryo_df_list.append(embryo_df)
            
        except:
            pass
        
    # path to figures and data
    figure_path = os.path.join(output_dir,  "figures")
    if not os.path.isdir(figure_path):
        os.makedirs(figure_path)

    # concatenate lists to form one large master set
    cdf_master = pd.concat(contrastive_df_list, axis=0, ignore_index=True)
    # mdf_master = pd.concat(meta_df_list, axis=0, ignore_index=True)
    # edf_master = pd.concat(embryo_df_list, axis=0, ignore_index=True)

    # get column names for latent variables in bio and nbio partitions
    zm_bio_cols = [col for col in cdf_master.columns if "z_mu_b" in col]
    zm_n_cols = [col for col in cdf_master.columns if "z_mu_n" in col]

    model_index = np.unique(cdf_master["mdl_id"])
    var_df_list = []
    mse_df_list = []

    # calculate distance between random pairs to estimate overall entropy
    for m, mdi in enumerate(tqdm(model_index)):
        temp_df = cdf_master.loc[cdf_master["mdl_id"] == mdi, :]
        temp_df = temp_df.reset_index()
        
        # extract arrays for comparison
        c0_indices = np.where(temp_df["contrast_id"] == 0)[0]
        c1_indices = np.where(temp_df["contrast_id"] == 1)[0]

        bio_array00 = temp_df.loc[c0_indices, zm_bio_cols].to_numpy()
        n_array00 = temp_df.loc[c0_indices, zm_n_cols].to_numpy()
        bio_array01 = temp_df.loc[c1_indices, zm_bio_cols].to_numpy()
        n_array01 = temp_df.loc[c1_indices, zm_n_cols].to_numpy()

        ######
        # calculate "baseline" variance in latent variables
        metric_df_temp00 = temp_df.loc[c0_indices, ["medium"] + zm_n_cols + zm_bio_cols]
        medium_vec = metric_df_temp00["medium"].values
        medium_vec_clean = [med[:2] for med in medium_vec]
        metric_df_temp00["medium"] = medium_vec_clean
        base_var_df = metric_df_temp00.groupby(by="medium").agg(["var"])
        
        # add key metadata
        base_var_df.loc[:, "class"] = "base"
        base_var_df.loc[:, "medium"] = base_var_df.index.values
        
        ######
        # calculate variance conditioned on location of self pair
        metric_df_temp01 = temp_df.loc[c0_indices, ["medium"] + zm_n_cols + zm_bio_cols]
        metric_df_temp01["medium"] = medium_vec_clean
        
        metric_df_temp01.loc[:, zm_n_cols] = n_array00 - n_array01
        metric_df_temp01.loc[:, zm_bio_cols] = bio_array00 - bio_array01
        self_var_df = metric_df_temp01.groupby(by="medium").agg(["var"])
        
        self_var_df.loc[:, "class"] = "self"
        self_var_df.loc[:, "medium"] = self_var_df.index.values
        
        # calculate variance conditioned on sequential pair
        snip_id_vec = temp_df.loc[c0_indices, "snip_id"].values
        eid_vec = np.asarray([si[:-10] for si in snip_id_vec])
        shift_vec = np.asarray([len(snip_id_vec)-1] + list(range(len(snip_id_vec)-1)))
        same_embryo_indices = np.where(eid_vec == eid_vec[shift_vec])[0]
        
        metric_df_temp02 = temp_df.loc[c0_indices, ["medium"] + zm_n_cols + zm_bio_cols]
        metric_df_temp02["medium"] = medium_vec_clean
        metric_df_temp02.loc[:, zm_n_cols] = n_array00 - n_array01[shift_vec, :]
        metric_df_temp02.loc[:, zm_bio_cols] = bio_array00 - bio_array01[shift_vec, :]
        
        # filter for only same-embryo comparisons
        metric_df_temp02 = metric_df_temp02.loc[same_embryo_indices, :]
        seq_var_df = metric_df_temp02.groupby(by="medium").agg(["var"])
        
        seq_var_df.loc[:, "class"] = "seq"
        seq_var_df.loc[:, "medium"] = seq_var_df.index.values
        
        # concatenate
        var_df = pd.concat([base_var_df, self_var_df, seq_var_df], axis=0, ignore_index=True)
        
        # add info
        var_df.loc[:, "temperature"] = temp_df.loc[0, "temperature"]
        var_df.loc[:, "gamma"] = temp_df.loc[0, "gamma"]
        var_df.loc[:, "mdl_id"] = mdi

        var_df_list.append(var_df)

        # calculate MSE
        emb_df = embryo_df_list[m]
        medium_vec = emb_df["medium"].values
        medium_vec_clean = [med[:2] for med in medium_vec]
        emb_df["medium"] = medium_vec_clean
        mse_df = emb_df.loc[:, ["medium", "recon_mse"]].groupby(by="medium").agg("mean")
        mse_df["mdl_id"] = mdi
        mse_df["medium"] = mse_df.index
        mse_df_list.append(mse_df)

    var_df_master = pd.concat(var_df_list, axis=0, ignore_index=True)
    top_cols = [col[0] for col in var_df_master.columns]
    var_df_master = pd.DataFrame(var_df_master.to_numpy(), columns=top_cols)

    mse_df_master = pd.concat(mse_df_list, axis=0, ignore_index=True)

    # calculate the entropy for each class
    entropy_df_master = var_df_master.loc[:, ['class', 'medium', 'temperature', 'gamma', 'mdl_id']]
    entropy_df_master["n_entropy"] = len(zm_n_cols)* 0.5 * np.log(2*np.pi*np.exp(1)) + \
                0.5 * np.log(np.prod(var_df_master.loc[:, zm_n_cols].to_numpy(), axis=1).astype(float))
    entropy_df_master["bio_entropy"] = len(zm_bio_cols)* 0.5 * np.log(2*np.pi*np.exp(1)) + \
                0.5 * np.log(np.prod(var_df_master.loc[:, zm_bio_cols].to_numpy(), axis=1).astype(float))

    # calculate sequential entropy
    base_ent_df = entropy_df_master.loc[entropy_df_master["class"]=="base"]
    base_ent_df = base_ent_df.rename(columns={"n_entropy":"n_entropy_base", "bio_entropy":"bio_entropy_base"})
    base_ent_df = base_ent_df.drop(labels="class", axis=1)

    self_ent_df = entropy_df_master.loc[entropy_df_master["class"]=="self"]
    self_ent_df = self_ent_df.rename(columns={"n_entropy":"n_entropy_self", "bio_entropy":"bio_entropy_self"})
    self_ent_df = self_ent_df.drop("class", axis=1)

    seq_ent_df = entropy_df_master.loc[entropy_df_master["class"]=="seq"]
    seq_ent_df = seq_ent_df.rename(columns={"n_entropy":"n_entropy_seq", "bio_entropy":"bio_entropy_seq"})
    seq_ent_df = seq_ent_df.drop("class", axis=1)

    entropy_df_wide = base_ent_df.reset_index()
    entropy_df_wide = entropy_df_wide.merge(self_ent_df, on=["medium", "temperature", "gamma", "mdl_id"], how="left")
    entropy_df_wide = entropy_df_wide.merge(seq_ent_df, on=["medium", "temperature", "gamma", "mdl_id"], how="left")

    entropy_df_wide.drop(labels="index", axis=1, inplace=True)
    entropy_df_wide = entropy_df_wide.merge(mse_df_master, on=["mdl_id", "medium"], how="left")

    # one more pivot...
    entropy_df_wide_mc = entropy_df_wide.loc[entropy_df_wide["medium"]=="MC", :]
    mc_cols = [col + "_mc" if ("entropy" in col) or ("mse" in col) else col for col in entropy_df_wide_mc.columns]
    entropy_df_wide_mc = pd.DataFrame(entropy_df_wide_mc.to_numpy(), columns=mc_cols)
    entropy_df_wide_mc.drop(labels=["medium"], axis=1, inplace=True)

    entropy_df_wide_em = entropy_df_wide.loc[entropy_df_wide["medium"]=="EM", :]
    em_cols = [col + "_em" if ("entropy" in col) or ("mse" in col) else col for col in entropy_df_wide_em.columns]
    entropy_df_wide_em = pd.DataFrame(entropy_df_wide_em.to_numpy(), columns=em_cols)
    entropy_df_wide_em.drop(labels=["medium"], axis=1, inplace=True)

    entropy_df_wider = entropy_df_wide_mc.merge(entropy_df_wide_em, on=["temperature", "gamma", "mdl_id"], how="left")
    entropy_df_wider.to_csv(os.path.join(figure_path, "latent_info_df.csv"))

    return entropy_df_wider

if __name__ == "__main__":

    # root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    # root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
    train_name = "20231106_ds"
    # model_name = "SeqVAE_z100_ne250_triplet_loss_test_self_and_other" #"SeqVAE_z100_ne250_gamma_temp_SELF_ONLY"
    architecture_name_vec = ["SeqVAE_z100_ne250_gamma_temp_self_and_other", "SeqVAE_z100_ne250_gamma_temp_SELF_ONLY", "SeqVAE_z100_ne250_triplet_loss_test_self_and_other"]
    # mode_vec = ["train", "eval", "test"]

    models_to_assess = None #["SeqVAE_training_2023-12-12_23-56-02"]

    for architecture_name in architecture_name_vec:
        calculate_latent_info_stats(root, train_name, architecture_name)

    


