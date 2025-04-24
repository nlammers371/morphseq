# We will save the data in folders to mimic the desired example
import os
import numpy as np
import glob as glob
import pandas as pd
from src.functions.utilities import path_leaf



def make_seq_key(root):  # , time_window=3, self_target=0.5, other_age_penalty=2):

    # read in metadata database
    embryo_metadata_df = pd.read_csv(os.path.join(root, "metadata", "embryo_metadata_df_train.csv"), low_memory=False)
    if "inferred_stage_hpf" in embryo_metadata_df.columns:
        seq_key = embryo_metadata_df.loc[:,
                  ["snip_id", "experiment_id", "experiment_date", "inferred_stage_hpf", "short_pert_name"]]
        seq_key = seq_key.rename(columns={"inferred_stage_hpf": "stage_hpf"})
    else:
        print("Warning: no age inference info found. Using default stage estimates.")
        seq_key = embryo_metadata_df.loc[:,
                  ["snip_id", "experiment_id", "experiment_date", "predicted_stage_hpf", "short_pert_name"]]
        seq_key = seq_key.rename(columns={"predicted_stage_hpf": "stage_hpf"})

    # The above dataset comprises all available images. Some training folders will only use a subset of these. Check
    # which images are in the present one (specified by "train name")
    image_list = []
    image_path_list = []
    lb_folder_list = glob.glob(os.path.join(root, "images", '') + "*")
    subfolder_list = []
    for subdir in lb_folder_list:
        image_list_temp = glob.glob(os.path.join(subdir, '') + "*.jpg")
        image_names = [path_leaf(path)[:-4] for path in image_list_temp]
        image_list += image_names
        image_path_list += image_list_temp
        subfolder_list += [subdir] * len(image_list_temp)

    # make temporary dataframe for merge purposes
    temp_df = pd.DataFrame(np.asarray(image_list), columns=["snip_id"])
    # temp_df["image_folder"] = mode_list
    temp_df["train_cat"] = ""
    temp_df["image_path"] = image_path_list
    seq_key = seq_key.merge(temp_df, how="inner", on="snip_id")

    # join on perturbation ID variables for convenience
    pert_u = np.unique(seq_key["short_pert_name"].to_numpy())
    pert_index = np.arange(len(pert_u))
    pert_df = pd.DataFrame(pert_u, columns=["short_pert_name"])
    pert_df["perturbation_id"] = pert_index
    seq_key = seq_key.merge(pert_df, how="left", on="short_pert_name")

    emb_id_list = [snip_id[:-6] for snip_id in list(seq_key["snip_id"])]
    seq_key["embryo_id"] = emb_id_list

    # join on embryo ID variable for convenience
    emb_u = np.unique(seq_key["embryo_id"].to_numpy())
    emb_index = np.arange(len(emb_u))
    emb_df = pd.DataFrame(emb_u, columns=["embryo_id"])
    emb_df["embryo_id_num"] = emb_index
    seq_key = seq_key.merge(emb_df, how="left", on="embryo_id")

    return seq_key


#########
# STEP 2

def make_train_test_split(seq_key, r_seed=371, train_eval_test=None, frac_to_use=1.0, test_dates=None, 
                          test_perturbations=None, pert_time_key=None):
    
    np.random.seed(r_seed)
    
    # this needs to be done at the level of embryos, not images
    if train_eval_test == None:
        train_eval_test = [0.75, 0.15, 0.1]
    train_eval_test = (np.asarray(train_eval_test) * frac_to_use).tolist()

    # check to see if there are any experiment dates or perturb ation types that should be left out of training (kept in test)
    test_constraints_flag = False
    seq_key["embryo_id"] = seq_key["embryo_id"].astype(str)
    seq_key["experiment_date"] = seq_key["experiment_date"].astype(str)
    seq_key["short_pert_name"] = seq_key["short_pert_name"].astype(str)

    emb_key = seq_key.loc[:, ["embryo_id", "short_pert_name", "experiment_date"]].drop_duplicates().reset_index(
        drop=True)
    emb_id_vec = np.unique(emb_key["embryo_id"].astype(str))
    if pert_time_key is not None:
        test_constraints_flag = True
        min_age_vec = np.asarray([pert_time_key.loc[pert_time_key["short_pert_name"] == pert, "start_hpf"].values[0]
                                  for pert in emb_key["short_pert_name"].tolist()])
        max_age_vec = np.asarray([pert_time_key.loc[pert_time_key["short_pert_name"] == pert, "stop_hpf"].values[0]
                                  for pert in emb_key["short_pert_name"].tolist()])
    else:
        min_age_vec = np.zeros(emb_id_vec.shape)
        max_age_vec = np.zeros(emb_id_vec.shape) + np.inf

    if test_dates is not None:
        test_constraints_flag = True
        min_age_vec[np.isin(emb_key.loc[:, "experiment_date"].astype(str).to_numpy(), np.asarray(test_dates))] = 200

    if test_perturbations is not None:
        test_constraints_flag = True
        min_age_vec[np.isin(emb_key[:, "short_pert_names"].to_numpy(), test_perturbations)] = np.inf

    # shuffle and filter
    embryo_id_index_shuffle = np.random.choice(emb_id_vec, len(emb_id_vec), replace=False)
    image_indices_shuffle = []

    # if we specified specific embryos to test, remove them now
    if test_constraints_flag:

        test_indices_pre = []
        for tid in embryo_id_index_shuffle:
            max_age = max_age_vec[emb_id_vec == tid]
            min_age = min_age_vec[emb_id_vec == tid]
            # if tid[:8] not in morphseq_dates:
            emb_filter = (seq_key["embryo_id"].values == tid)
            time_filter = (seq_key["stage_hpf"].values > max_age) | (seq_key["stage_hpf"].values < min_age)
            df_ids = np.where(emb_filter & time_filter)[0]

            # snip_list = seq_key["snip_id"].iloc[df_ids].tolist()
            i_list = df_ids.tolist()

            # test_paths_pre += e_list
            test_indices_pre += i_list

        # embryo_id_index_shuffle = [e for e in embryo_id_index_shuffle if e not in test_ids]

    else:  # otherwise, sample embryos that were raised in MC, since these will give better stability over time
        test_indices_pre = []

    # iterate through shuffled IDs
    for eid in embryo_id_index_shuffle:
        max_age = max_age_vec[emb_id_vec == eid]
        min_age = min_age_vec[emb_id_vec == eid]
        # extract embryos that match current eid
        emb_filter = (seq_key["embryo_id"].values == eid)
        time_filter = (seq_key["stage_hpf"].values <= max_age) & (
                seq_key["stage_hpf"].values >= min_age)

        df_ids = np.where(emb_filter & time_filter)[0]
        i_list = df_ids.tolist()

        # remove frames that did not meet QC standards
        image_indices_shuffle += i_list

    # assign to groups. Note that a few frames from the same embryo may end up split between categories. I think that this is fine
    n_frames_total = len(image_indices_shuffle)

    n_train = np.round(train_eval_test[0] * n_frames_total).astype(int)
    # train_paths = image_list_shuffle[:n_train]
    train_indices = image_indices_shuffle[:n_train]
    # train_df_indices = df_ids_list[:n_train]

    n_eval = np.round(train_eval_test[1] * n_frames_total).astype(int)
    # eval_paths = image_list_shuffle[n_train:n_train+n_eval]
    eval_indices = image_indices_shuffle[n_train:n_train + n_eval]
    # eval_df_indices = df_ids_list[n_train:n_train+n_eval]

    n_test = np.round(train_eval_test[2] * n_frames_total).astype(int)
    # test_paths = test_paths_pre + image_list_shuffle[n_train+n_eval:n_train+n_eval+n_test]
    test_indices = test_indices_pre + image_indices_shuffle[n_train + n_eval:n_train + n_eval + n_test]
    # test_df_indices = df_indices_pre + df_ids_list[n_train+n_eval:n_train+n_eval+n_test]

    # update seq key
    seq_key.loc[train_indices, "train_cat"] = "train"
    seq_key.loc[test_indices, "train_cat"] = "test"
    seq_key.loc[eval_indices, "train_cat"] = "eval"

    return seq_key, train_indices, eval_indices, test_indices


if __name__ == "__main__":
    # set path to data
    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\morphseq\\"

    train_name = "20231106_ds"
    label_var = "experiment_date"

    make_seq_key(root, train_name)


# def split_train_test(self):
    #     """
    #     Load the dataset from the specified file path using pandas.
    #     """
    #     # get seq key
    #     seq_key = make_seq_key(self.data_root, self.train_folder)
    #
    #     if self.age_key_path != '':
    #         age_key_df = pd.read_csv(self.age_key_path, index_col=0)
    #         age_key_df = age_key_df.loc[:, ["snip_id", "inferred_stage_hpf_reg"]]
    #         seq_key = seq_key.merge(age_key_df, how="left", on="snip_id")
    #     else:
    #         # raise Warning("No age key path provided")
    #         seq_key["inferred_stage_hpf_reg"] = 1
    #
    #     if self.pert_time_key_path != '':
    #         pert_time_key = pd.read_csv(self.pert_time_key_path)
    #     else:
    #         pert_time_key = None
    #
    #     seq_key, train_indices, eval_indices, test_indices = make_train_test_split(seq_key, pert_time_key=pert_time_key)
    #
    #     self.seq_key = seq_key
    #     self.eval_indices = eval_indices
    #     self.test_indices = test_indices
    #     self.train_indices = train_indices