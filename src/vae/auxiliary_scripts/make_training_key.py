# We will save the data in folders to mimic the desired example
import os
import torch
import numpy as np
import glob as glob
import pandas as pd
from src.functions.utilities import path_leaf


###########
# STEP 1

def make_seq_key(root, train_name): #, time_window=3, self_target=0.5, other_age_penalty=2):

    metadata_path = os.path.join(root, "metadata", '')
    training_path = os.path.join(root, "training_data", train_name, '')
    # instance_path = os.path.join(training_path, instance_name)
    mode_vec = sorted(glob.glob(training_path + "*"))

    # read in metadata database
    embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df_final.csv"), index_col=0)
    seq_key = embryo_metadata_df.loc[:, ["snip_id", "experiment_id", "predicted_stage_hpf", "master_perturbation"]]

    # The above dataset comprises all available images. Some training folders will only use a subset of these. Check
    # which images are in the present one (specified by "train name")
    image_list = []
    image_path_list = []
    mode_list = []
    for mode in mode_vec:
        lb_folder_list = glob.glob(os.path.join(training_path, mode, '') + "*")
        for subdir in lb_folder_list:
            image_list_temp = glob.glob(os.path.join(subdir, '') + "*.jpg")
            image_names = [path_leaf(path)[:-4] for path in image_list_temp]
            image_list += image_names
            image_path_list += image_list_temp
            mode_list += [mode]*len(image_names)

    # make temporary dataframe for merge purposes
    temp_df = pd.DataFrame(np.asarray(image_list), columns=["snip_id"])
    temp_df["image_folder"] = mode_list
    temp_df["train_cat"] = ""
    temp_df["image_path"] = image_path_list
    seq_key = seq_key.merge(temp_df, how="inner", on="snip_id")

    # join on perturbation ID variables for convenience
    pert_u = np.unique(seq_key["master_perturbation"].to_numpy())
    pert_index = np.arange(len(pert_u))
    pert_df = pd.DataFrame(pert_u, columns=["master_perturbation"])
    pert_df["perturbation_id"] = pert_index
    seq_key = seq_key.merge(pert_df, how="left", on="master_perturbation")

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

def make_train_test_split(seq_key, r_seed=371, train_eval_test=None,
                            frac_to_use=1.0, test_ids=None, test_dates=None, test_perturbations=None,
                            overwrite_flag=False):


    np.random.seed(r_seed)

    # get list of training files
    image_list = seq_key["image_path"].to_numpy().tolist() 
    snip_id_list = [path_leaf(path) for path in image_list]

    # randomly partition into train, eval, and test
    # this needs to be done at the level of embryos, not images
    if train_eval_test == None:
        train_eval_test = [0.8, 0.20, 0.0]
    train_eval_test = (np.asarray(train_eval_test)*frac_to_use).tolist()

    # snip_id_vec = embryo_metadata_df["snip_id"].values
    # good_snip_indices = np.where(embryo_metadata_df["use_embryo_flag"].values == True)[0]
    embryo_id_index = np.unique(seq_key["embryo_id"].values)

    # check to see if there are any experiment dates or perturb ation types that should be left out of training (kept in test)
    if test_dates is not None:
        eid_date_list = [seq_key.loc[e, "embryo_id"] for e in seq_key.index if seq_key.loc[e, "experiment_date"].astype(str) in test_dates]
        eids_date_test = np.unique(eid_date_list).tolist()
        if test_ids is not None:
            test_ids += eids_date_test
        else:
            test_ids = eids_date_test

    if test_perturbations is not None:
        eid_pert_list = [seq_key.loc[e, "embryo_id"] for e in seq_key.index if seq_key.loc[e, "master_perturbation"] in test_perturbations]
        eids_pert_test = np.unique(eid_pert_list).tolist()
        if test_ids is not None:
            test_ids += eids_pert_test
        else:
            test_ids = eids_pert_test 

    # shuffle and filter
    embryo_id_index_shuffle = np.random.choice(embryo_id_index, len(embryo_id_index), replace=False)
    # image_list_shuffle = []
    image_indices_shuffle = []
    
    # if we specified specific embryos to test, remove them now
    if test_ids != None:
        test_ids = np.unique(test_ids)

        # test_paths_pre = []
        test_indices_pre = []
        # df_indices_pre = []
        for tid in test_ids:
            if tid[:8] not in morphseq_dates:
                df_ids = np.where((seq_key["embryo_id"].values == tid) & (seq_key["use_embryo_flag"].values == True))[0]
            else:
                df_ids = np.where((seq_key["embryo_id"].values == tid))[0]
                
            snip_list = seq_key["snip_id"].iloc[df_ids].tolist()
            # e_list = [os.path.join(data_path, s + ".jpg") for s in snip_list]
            i_list = df_ids.tolist()
    
            # test_paths_pre += e_list
            test_indices_pre += i_list
            # df_indices_pre += df_ids.tolist()
        
        # train_eval_test[-1] = 0
        # train_eval_test = train_eval_test / np.sum(train_eval_test)
        
        embryo_id_index_shuffle = [e for e in embryo_id_index_shuffle if e not in test_ids]

    else:  # otherwise, sample embryos that were raised in MC, since these will give better stability over time
        test_paths_pre = []
        test_indices_pre = []
        # df_indices_pre = []

    # itereate through shuffled IDs
    df_ids_list = []
    for eid in embryo_id_index_shuffle:
    
        # extract embryos that match current eid
        df_ids = np.where((seq_key["embryo_id"].values == eid))[0] #& (seq_key["use_embryo_flag"].values == True))[0]
        snip_list = seq_key["snip_id"].iloc[df_ids].tolist()
        i_list = df_ids.tolist()
    
        # remove frames that did not meet QC standards
        image_indices_shuffle += i_list
    
    # assign to groups. Note that a few frames from the same embryo may end up split between categories. I think that this is fine
    n_frames_total = len(image_indices_shuffle) #np.sum(seq_key["use_embryo_flag"].values == True)

    n_train = np.round(train_eval_test[0]*n_frames_total).astype(int)
    # train_paths = image_list_shuffle[:n_train]
    train_indices = image_indices_shuffle[:n_train]
    # train_df_indices = df_ids_list[:n_train]
    
    n_eval = np.round(train_eval_test[1]*n_frames_total).astype(int)
    # eval_paths = image_list_shuffle[n_train:n_train+n_eval]
    eval_indices = image_indices_shuffle[n_train:n_train+n_eval]
    # eval_df_indices = df_ids_list[n_train:n_train+n_eval]

    n_test = np.round(train_eval_test[2]*n_frames_total).astype(int)
    # test_paths = test_paths_pre + image_list_shuffle[n_train+n_eval:n_train+n_eval+n_test]
    test_indices = test_indices_pre + image_indices_shuffle[n_train+n_eval:n_train+n_eval+n_test]
    # test_df_indices = df_indices_pre + df_ids_list[n_train+n_eval:n_train+n_eval+n_test]

    # update seq key
    seq_key.loc[train_indices, "train_cat"] = "train"
    seq_key.loc[test_indices, "train_cat"] = "test"
    seq_key.loc[eval_indices, "train_cat"] = "eval"

    return seq_key, train_indices, eval_indices, test_indices

#########
# STEP 3

# def get_sequential_pairs(seq_key, time_window, self_target, other_age_penalty, mode_vec=None):

#     if mode_vec is None:
#         mode_vec = np.unique(seq_key["train_cat"])    #  ["train", "eval", "test"]

#     seq_pair_dict = dict({})

#     for m, mode in enumerate(mode_vec):

#         seq_key_mode = seq_key.loc[seq_key["train_cat"] == mode].copy()
#         seq_key_mode = seq_key_mode.reset_index()

#         # Strip paths to get data snip_ids
#         snip_id_list = seq_key_mode["snip_id"].to_numpy()

#         pert_id_vec = seq_key_mode["perturbation_id"].to_numpy()
#         e_id_vec = seq_key_mode["embryo_id"].to_numpy()
#         age_hpf_vec = seq_key_mode["predicted_stage_hpf"].to_numpy()


#         seq_key_dict_mode = dict({})
#         for s in range(len(snip_id_list)):
#             snip_i = np.where(seq_key_mode["snip_id"] == snip_id_list[s])[0][0]

#             # self_entry = pd.DataFrame(seq_key.iloc[snip_i, :]).transpose().reset_index()
#             age_hpf = seq_key_mode.loc[snip_i, "predicted_stage_hpf"]
#             embryo_id = seq_key_mode.loc[snip_i, "embryo_id"]
#             pert_id = seq_key_mode.loc[snip_i, "perturbation_id"]

#             # # find valid self comparisons
#             # seq_key_self = seq_key.merge(self_entry["embryo_id"], how="inner", on=["embryo_id"])
#             # self_age_deltas = np.abs(seq_key_self["predicted_stage_hpf"].to_numpy() - age_hpf)
#             # valid_self_indices = seq_key_self.loc[np.where(self_age_deltas <= time_window)[0], "Index"].to_numpy()

#             # find valid "other" comparisons (same class different embryo)
#             # seq_key_other = seq_key.merge(self_entry[["perturbation_id", "train_cat"]], how="inner",
#             #                               on=["perturbation_id", "train_cat"])
#             # other_age_deltas = np.abs(seq_key_other["predicted_stage_hpf"].to_numpy() - age_hpf)
#             age_deltas = np.abs(age_hpf_vec - age_hpf)
#             valid_all_indices = np.where((pert_id_vec == pert_id) & (age_deltas <= time_window))[0]
#             valid_self_sub_indices = np.where((e_id_vec[valid_all_indices] == embryo_id))[0]

#             # get overall and class-specific indices for each option. Assign weights
#             other_target = 1 - self_target
#             self_frac = len(valid_self_sub_indices) / (len(valid_all_indices))
#             self_weight = self_target / self_frac
#             if self_frac < 1.0:
#                 other_weight = other_target / (1 - self_frac)
#             else:
#                 other_weight = np.Inf

#             # generate weight vector
#             option_weights = np.ones(valid_all_indices.shape)*other_weight
#             option_weights[valid_self_sub_indices] = self_weight
#             option_weights = option_weights / np.sum(option_weights)

#             # generate time delta vector
#             age_delta_vec = age_deltas[valid_all_indices] + 1 + other_age_penalty
#             age_delta_vec[valid_self_sub_indices] = age_delta_vec[valid_self_sub_indices] - other_age_penalty

#             # option_weights = np.ones(valid_all_indices.shape) #np.asarray(
#             #     # [self_weight if (i in valid_self_indices) else other_weight for i in valid_all_indices])
#             # option_weights = option_weights / np.sum(option_weights)
#             # age_weight_factors = age_deltas[valid_all_indices] + 1
#             # extra_weight = 1
#             # if (np.random.rand() <= self_target) or (len(valid_other_indices) == 0):
#             #     seq_pair_index = np.random.choice(range(len(valid_self_indices)), 1, replace=False)[0]
#             # else:
#             #     seq_pair_index = np.random.choice(range(len(valid_other_indices)), 1, replace=False)[0]
#             #     extra_weight += other_age_penalty
#             # randomly select an index for comparison
#             # seq_pair_ind = np.random.choice(range(len(valid_all_indices)), 1, replace=False, p=option_weights)
#             # load_index = valid_all_indices[seq_pair_ind[0]]
#             snip_dict = dict({})
#             snip_dict["seq_pair_indices"] = valid_all_indices
#             snip_dict["seq_pair_weights"] = option_weights
#             snip_dict["seq_pair_deltas"] = age_delta_vec

#             seq_key_dict_mode[snip_id_list[s]] = snip_dict
#             # indices_to_load.append(seq_pair_index)
#             # pair_time_deltas.append(age_deltas[seq_pair_index] + extra_weight)
#             # if load_index in valid_self_indices:
#             #     pair_time_deltas.append(age_weight_factors[seq_pair_ind[0]])
#             # else:
#             #     pair_time_deltas.append(age_weight_factors[seq_pair_ind[0]] + other_age_penalty)

#         seq_pair_dict[mode] = seq_key_dict_mode

#         # # load input pairs into memory
#         # input_init = torch.reshape(inputs["data"], (inputs["data"].shape[0], 1, inputs["data"].shape[1],
#         #                                             inputs["data"].shape[2], inputs["data"].shape[3]))
#         # input_pairs = torch.empty(input_init.shape)
#         # # for i, ind in enumerate(indices_to_load):
#         # #     input_pairs[i, 0, 0, :, :] = self.train_loader.dataset[ind]["data"]
#         #
#         # inputs["data"] = torch.cat([input_init, input_pairs], dim=1)
#         # inputs["hpf_deltas"] = torch.ones((input_pairs.shape[0],)) #torch.FloatTensor(pair_time_deltas) / max_val

#     return seq_pair_dict
    # seq_key["Index"] = seq_key.index
    # # now, for each image compile a list of valid comparisons and corresponding delta T's
    # seq_key_dict = dict({})
    # # to be a valid comparison, an image needs to be of the same perturbation type and within 3 hpf
    # for i in tqdm(range(seq_key.shape[0])):
    #     self_entry = pd.DataFrame(seq_key.iloc[i, :]).transpose().reset_index()
    #     age_hpf = seq_key.loc[i, "predicted_stage_hpf"]
    #
    #     # find valid self comparisons
    #     seq_key_self = seq_key.merge(self_entry["embryo_id"], how="inner", on=["embryo_id"])
    #     self_age_deltas = np.abs(seq_key_self["predicted_stage_hpf"].to_numpy() - age_hpf)
    #     valid_self_indices = seq_key_self.loc[np.where(self_age_deltas <= time_window)[0], "Index"].to_numpy()
    #
    #     # find valid "other" comparisons (same class different embryo)
    #     seq_key_other = seq_key.merge(self_entry[["perturbation_id", "train_cat"]], how="inner", on=["perturbation_id", "train_cat"])
    #     other_age_deltas = np.abs(seq_key_other["predicted_stage_hpf"].to_numpy() - age_hpf)
    #     valid_all_indices = seq_key_other.loc[np.where(other_age_deltas <= time_window)[0], "Index"].to_numpy()
    #
    #     # get overall and class-specific indices for each option. Assign weights
    #     other_target = 1 - self_target
    #     self_frac = len(valid_self_indices) / (len(valid_all_indices))
    #     self_weight = self_target / self_frac
    #     other_weight = other_target / (1-self_frac)
    #
    #     option_weights = [self_weight if (i in valid_self_indices) else other_weight for i in valid_all_indices]
    #     option_time_deltas = other_age_deltas[np.where(other_age_deltas <= time_window)[0]] + 1
    #     option_time_deltas = [option_time_deltas[i] if (valid_all_indices[i] in valid_self_indices) else
    #                           option_time_deltas[i] + other_age_penalty for i in range(len(valid_all_indices))]
    #
    #     frame_dict = dict({})
    #     frame_dict["option_indices"] = valid_all_indices
    #     frame_dict["option_weights"] = option_weights
    #     frame_dict["option_hpf_deltas"] = option_time_deltas
    #     seq_key_dict[self_entry.loc[0, "snip_id"]] = frame_dict

    # seq_key.to_csv(os.path.join(instance_dir, "seq_key.csv"))


if __name__ == "__main__":
    # set path to data
    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\morphseq\\"

    train_name = "20231106_ds"
    label_var = "experiment_date"

    make_seq_key(root, train_name)