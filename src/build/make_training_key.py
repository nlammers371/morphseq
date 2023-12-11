# We will save the data in folders to mimic the desired example
import os
import torch
import numpy as np
import glob as glob
import pandas as pd
from src.functions.utilities import path_leaf

def make_seq_key(root, train_name): #, time_window=3, self_target=0.5, other_age_penalty=2):

    metadata_path = os.path.join(root, "metadata", '')
    training_path = os.path.join(root, "training_data", train_name, '')
    # instance_path = os.path.join(training_path, instance_name)
    mode_vec = ["train", "eval", "test"]

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
    temp_df["train_cat"] = mode_list
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
    seq_key["Index"] = seq_key.index

    return seq_key

def get_sequential_pairs(seq_key, time_window, self_target, other_age_penalty, mode_vec=None):

    if mode_vec is None:
        mode_vec = ["train", "eval", "test"]

    seq_pair_dict = dict({})

    for m, mode in enumerate(mode_vec):

        seq_key_mode = seq_key.loc[seq_key["train_cat"] == mode].copy()
        seq_key_mode = seq_key_mode.reset_index()
        # Strip paths to get data snip_ids
        # input_paths = list(inputs["label"][0])
        snip_id_list = seq_key_mode["snip_id"].to_numpy()

        # time_window = self.model_config.time_window
        # self_target = self.model_config.self_target_prob
        # other_age_penalty = self.model_config.other_age_penalty

        # max_val = 1 + other_age_penalty + time_window

        # seq_key = self.model_config.seq_key

        # seq_key = seq_key.reset_index()
        # seq_key["Index"] = seq_key.index

        pert_id_vec = seq_key_mode["perturbation_id"].to_numpy()
        e_id_vec = seq_key_mode["embryo_id"].to_numpy()
        age_hpf_vec = seq_key_mode["predicted_stage_hpf"].to_numpy()

        # indices_to_load = []
        # pair_time_deltas = []
        seq_key_dict_mode = dict({})
        for s in range(len(snip_id_list)):
            snip_i = np.where(seq_key_mode["snip_id"] == snip_id_list[s])[0][0]
            # self_entry = pd.DataFrame(seq_key.iloc[snip_i, :]).transpose().reset_index()
            age_hpf = seq_key_mode.loc[snip_i, "predicted_stage_hpf"]
            embryo_id = seq_key_mode.loc[snip_i, "embryo_id"]
            pert_id = seq_key_mode.loc[snip_i, "perturbation_id"]

            #
            # # find valid self comparisons
            # seq_key_self = seq_key.merge(self_entry["embryo_id"], how="inner", on=["embryo_id"])
            # self_age_deltas = np.abs(seq_key_self["predicted_stage_hpf"].to_numpy() - age_hpf)
            # valid_self_indices = seq_key_self.loc[np.where(self_age_deltas <= time_window)[0], "Index"].to_numpy()

            # find valid "other" comparisons (same class different embryo)
            # seq_key_other = seq_key.merge(self_entry[["perturbation_id", "train_cat"]], how="inner",
            #                               on=["perturbation_id", "train_cat"])
            # other_age_deltas = np.abs(seq_key_other["predicted_stage_hpf"].to_numpy() - age_hpf)
            age_deltas = np.abs(age_hpf_vec - age_hpf)
            valid_all_indices = np.where((pert_id_vec == pert_id) & (age_deltas <= time_window))[0]
            valid_self_sub_indices = np.where((e_id_vec[valid_all_indices] == embryo_id))[0]

            # get overall and class-specific indices for each option. Assign weights
            other_target = 1 - self_target
            self_frac = len(valid_self_sub_indices) / (len(valid_all_indices))
            self_weight = self_target / self_frac
            if self_frac < 1.0:
                other_weight = other_target / (1 - self_frac)
            else:
                other_weight = np.Inf

            # generate weight vector
            option_weights = np.ones(valid_all_indices.shape)*other_weight
            option_weights[valid_self_sub_indices] = self_weight
            option_weights = option_weights / np.sum(option_weights)

            # generate time delta vector
            age_delta_vec = age_deltas[valid_all_indices] + 1 + other_age_penalty
            age_delta_vec[valid_self_sub_indices] = age_delta_vec[valid_self_sub_indices] - other_age_penalty

            # option_weights = np.ones(valid_all_indices.shape) #np.asarray(
            #     # [self_weight if (i in valid_self_indices) else other_weight for i in valid_all_indices])
            # option_weights = option_weights / np.sum(option_weights)
            # age_weight_factors = age_deltas[valid_all_indices] + 1
            # extra_weight = 1
            # if (np.random.rand() <= self_target) or (len(valid_other_indices) == 0):
            #     seq_pair_index = np.random.choice(range(len(valid_self_indices)), 1, replace=False)[0]
            # else:
            #     seq_pair_index = np.random.choice(range(len(valid_other_indices)), 1, replace=False)[0]
            #     extra_weight += other_age_penalty
            # randomly select an index for comparison
            # seq_pair_ind = np.random.choice(range(len(valid_all_indices)), 1, replace=False, p=option_weights)
            # load_index = valid_all_indices[seq_pair_ind[0]]
            snip_dict = dict({})
            snip_dict["seq_pair_indices"] = valid_all_indices
            snip_dict["seq_pair_weights"] = option_weights
            snip_dict["seq_pair_deltas"] = age_delta_vec

            seq_key_dict_mode[snip_id_list[s]] = snip_dict
            # indices_to_load.append(seq_pair_index)
            # pair_time_deltas.append(age_deltas[seq_pair_index] + extra_weight)
            # if load_index in valid_self_indices:
            #     pair_time_deltas.append(age_weight_factors[seq_pair_ind[0]])
            # else:
            #     pair_time_deltas.append(age_weight_factors[seq_pair_ind[0]] + other_age_penalty)

        seq_pair_dict[mode] = seq_key_dict_mode

        # # load input pairs into memory
        # input_init = torch.reshape(inputs["data"], (inputs["data"].shape[0], 1, inputs["data"].shape[1],
        #                                             inputs["data"].shape[2], inputs["data"].shape[3]))
        # input_pairs = torch.empty(input_init.shape)
        # # for i, ind in enumerate(indices_to_load):
        # #     input_pairs[i, 0, 0, :, :] = self.train_loader.dataset[ind]["data"]
        #
        # inputs["data"] = torch.cat([input_init, input_pairs], dim=1)
        # inputs["hpf_deltas"] = torch.ones((input_pairs.shape[0],)) #torch.FloatTensor(pair_time_deltas) / max_val

    return seq_pair_dict
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