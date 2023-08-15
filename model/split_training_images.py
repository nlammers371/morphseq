# We will save the data in folders to mimic the desired example
import os
import torch
import numpy as np
import glob as glob
import pandas as pd
from _archive.functions_folder.utilities import path_leaf
import imageio
from tqdm import tqdm


def make_pythae_image_snips(root, train_name, r_seed=371, label_var="experiment_date", train_eval_test=None,
                            frac_to_use=1.0, test_ids=None):

    np.random.seed(r_seed)
    metadata_path = os.path.join(root, "metadata", '')
    data_path = os.path.join(root, "training_data", "bf_embryo_snips", '')
    
    # read in metadata database
    embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df_final.csv"), index_col=0)
    
    # get list of training files
    image_list = glob.glob(data_path + "*.tif")
    snip_id_list = [path_leaf(path) for path in image_list]
    emb_id_list = [eid[0:16] for eid in snip_id_list]
    
    # randomly partition into train, eval, and test
    # this needs to be done at the level of embryos, not images
    if train_eval_test == None:
        train_eval_test = [0.7, 0.15, 0.15]
    train_eval_test = (np.asarray(train_eval_test)*frac_to_use).tolist()
    # n_total = embryo_metadata_df.shape[0]
    
    
    n_frames_total = np.sum(embryo_metadata_df["use_embryo_flag"].values == True)
    # snip_id_vec = embryo_metadata_df["snip_id"].values
    # good_snip_indices = np.where(embryo_metadata_df["use_embryo_flag"].values == True)[0]
    embryo_id_index = np.unique(embryo_metadata_df["embryo_id"].values)
    
    # shuffle and filter
    embryo_id_index_shuffle = np.random.choice(embryo_id_index, len(embryo_id_index), replace=False)
    image_list_shuffle = []
    image_indices_shuffle = []
    
    # if we specified specific embryos to test, remove them now
    if test_ids != None:
        test_paths = []
        test_indices = []
        for tid in test_ids:
            df_ids = np.where((embryo_metadata_df["embryo_id"].values == tid) &
                     (embryo_metadata_df["use_embryo_flag"].values == True))[0]
            snip_list = embryo_metadata_df["snip_id"].iloc[df_ids].tolist()
            e_list = [os.path.join(data_path, s + ".tif") for s in snip_list]
            i_list = df_ids.tolist()
    
            test_paths += e_list
            test_indices += i_list
        
        train_eval_test[-1] = 0
        # train_eval_test = train_eval_test / np.sum(train_eval_test)
        
        embryo_id_index_shuffle = [e for e in embryo_id_index_shuffle if e not in test_ids]

    else:  # otherwise, sample embryos that were raised in MC, since these will give better stability over time
        test_paths = []
        test_indices = []

        # n_test = np.round(train_eval_test[2]*n_frames_total).astype(int)
        # n_test_curr = 0
        # iter_i = 0
        # # mc_ids = np.where("MC" in embryo_metadata_df["medium"])[0]
        # used_indices = []
        #
        # while n_test_curr < n_test:
        #     eid = embryo_id_index_shuffle[iter_i]
        #
        #     # extract embryos that match current eid
        #     df_ids = np.where((embryo_metadata_df["embryo_id"].values == eid) & (
        #                 embryo_metadata_df["use_embryo_flag"].values == True))[0]
        #     if len(df_ids) > 0:
        #         medium = embryo_metadata_df["medium"].iloc[df_ids[0]]
        #         if "MC" in medium:
        #             used_indices.append(iter_i)
        #             snip_list = embryo_metadata_df["snip_id"].iloc[df_ids].tolist()
        #             e_list = [os.path.join(data_path, s + ".tif") for s in snip_list]
        #             i_list = df_ids.tolist()
        #
        #             # remove frames that did not meet QC standards
        #             test_paths += e_list
        #             test_indices += i_list
        #
        #             n_test_curr += len(e_list)
        #
        #     iter_i += 1
        #
        # # reset fractions and avaialable eids
        # n_rem = np.max([0, n_test - n_test_curr])
        # test_frac_new = n_rem / n_frames_total
        # train_eval_test[-1] = test_frac_new
        #
        # # train_eval_test = train_eval_test / np.sum(train_eval_test)
        #
        # embryo_id_index_shuffle = [embryo_id_index_shuffle[e] for e in range(len(embryo_id_index_shuffle)) if e not in used_indices]

    # itereate through shuffled IDs
    for eid in embryo_id_index_shuffle:
    
        # extract embryos that match current eid
        df_ids = np.where((embryo_metadata_df["embryo_id"].values == eid) & (embryo_metadata_df["use_embryo_flag"].values == True))[0]
        snip_list = embryo_metadata_df["snip_id"].iloc[df_ids].tolist()
        e_list = [os.path.join(data_path, s + ".tif") for s in snip_list]
        i_list = df_ids.tolist()
    
        # remove frames that did not meet QC standards
        image_list_shuffle += e_list
        image_indices_shuffle += i_list
    
    # assign to groups. Note that a few frames from the same embryo may end up split between categories. I think that this is fine
    n_train = np.round(train_eval_test[0]*n_frames_total).astype(int)
    train_paths = image_list_shuffle[:n_train]
    train_indices = image_indices_shuffle[:n_train]
    
    n_eval = np.round(train_eval_test[1]*n_frames_total).astype(int)
    eval_paths = image_list_shuffle[n_train:n_train+n_eval]
    eval_indices = image_indices_shuffle[n_train:n_train+n_eval]
    
    n_test = np.round(train_eval_test[2]*n_frames_total).astype(int)
    test_paths += image_list_shuffle[n_train+n_eval:n_train+n_eval+n_test]
    test_indices += image_indices_shuffle[n_train+n_eval:n_train+n_eval+n_test]
    
    
    train_dir = os.path.join(root, "training_data", train_name)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(os.path.join(train_dir, "train")):
        os.mkdir(os.path.join(train_dir, "train"))
    if not os.path.exists(os.path.join(train_dir, "eval")):
        os.mkdir(os.path.join(train_dir, "eval"))
    if not os.path.exists(os.path.join(train_dir, "test")):
        os.mkdir(os.path.join(train_dir, "test"))
    
    #################
    # Write snips to file
    
    # training snips
    print("Generating training snips...")
    for i in tqdm(range(len(train_paths))):
        img = torch.from_numpy(imageio.imread(train_paths[i]))
        img_name = path_leaf(train_paths[i])
    
        if label_var != None:
            lb_name = embryo_metadata_df[label_var].iloc[train_indices[i]].astype(str)
            img_folder = os.path.join(train_dir, "train", lb_name)
        else:
            img_folder = os.path.join(train_dir, "train", "0")  # I'm assuming the code will expect a subfolder
    
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)
    
        imageio.imwrite(os.path.join(img_folder, img_name[:-4] + ".jpg"), np.repeat(img[:, :, np.newaxis], repeats=3, axis=2).type(torch.uint8))
    
    # eval
    print("Generating evalation snips...")
    for i in tqdm(range(len(eval_paths))):
        img = torch.from_numpy(imageio.imread(eval_paths[i]))
        img_name = path_leaf(eval_paths[i])
    
        if label_var != None:
            lb_name = embryo_metadata_df[label_var].iloc[eval_indices[i]].astype(str)
            img_folder = os.path.join(train_dir, "eval", lb_name)
        else:
            img_folder = os.path.join(train_dir, "eval", "0")  # I'm assuming the code will expect a subfolder
    
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)
    
        imageio.imwrite(os.path.join(img_folder, img_name[:-4] + ".jpg"), np.repeat(img[:, :, np.newaxis], repeats=3, axis=2).type(torch.uint8))
    
    # test
    print("Generating testing snips...")
    for i in tqdm(range(len(test_paths))):
        img = torch.from_numpy(imageio.imread(test_paths[i]))
        img_name = path_leaf(test_paths[i])
    
        if label_var != None:
            lb_name = embryo_metadata_df[label_var].iloc[test_indices[i]].astype(str)
            img_folder = os.path.join(train_dir, "test", lb_name)
        else:
            img_folder = os.path.join(train_dir, "test", "0")  # I'm assuming the code will expect a subfolder
    
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)
    
        imageio.imwrite(os.path.join(img_folder, img_name[:-4] + ".jpg"), np.repeat(img[:, :, np.newaxis], repeats=3, axis=2).type(torch.uint8))

        
    print("Done.")


if __name__ == "__main__":
    # set path to data
    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"

    train_name = "20230815_vae"
    label_var = "experiment_date"

    make_pythae_image_snips(root, train_name, label_var="experiment_date", frac_to_use=1.0)