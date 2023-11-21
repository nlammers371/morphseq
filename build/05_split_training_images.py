# We will save the data in folders to mimic the desired example
import os
import torch
import numpy as np
import glob as glob
import pandas as pd
from functions.utilities import path_leaf
import imageio
from tqdm import tqdm
from skimage.transform import rescale

def make_pythae_image_snips(root, train_name, r_seed=371, label_var="experiment_date", train_eval_test=None,
                            frac_to_use=1.0, test_ids=None, flip_dataset_path=None, rs_factor=1.0):

    flip_flag = False
    if flip_dataset_path != None:
        flip_flag = True
        flip_df = pd.read_csv(flip_dataset_path, index_col=0)
        flip_snip_id_vec = flip_df["snip_id"]
        flip_flag_vec = flip_df["revision_labels"]

    rs_flag = rs_factor != 1

    np.random.seed(r_seed)
    metadata_path = os.path.join(root, "metadata", '')
    data_path = os.path.join(root, "training_data", "bf_embryo_snips", '')
    
    # read in metadata database
    embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df_final.csv"), index_col=0)
    
    # get list of training files
    image_list = glob.glob(data_path + "*.tif")
    snip_id_list = [path_leaf(path) for path in image_list]
    # emb_id_list = [eid[0:16] for eid in snip_id_list]
    
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
        img_raw = imageio.v2.imread(train_paths[i])
        if rs_flag:
            img = torch.from_numpy(rescale(img_raw.astype(np.float16), rs_factor, anti_aliasing=True))
        else:
            img = torch.from_numpy(img_raw)

        img_name = path_leaf(train_paths[i])

        if label_var != None:
            lb_name = embryo_metadata_df[label_var].iloc[train_indices[i]].astype(str)
            img_folder = os.path.join(train_dir, "train", lb_name)
        else:
            img_folder = os.path.join(train_dir, "train", "0")  # I'm assuming the code will expect a subfolder

        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

        write_flag = True
        if flip_flag:
            snip_ind = np.where(np.asarray(flip_snip_id_vec) == snip_id_list[train_indices[i]][:-4])[0][0]
            flip_id = int(flip_flag_vec[snip_ind])
            if flip_id == 1:
                img = torch.fliplr(img)
            elif flip_id == -1:
                write_flag = False

        if write_flag:
            imageio.imwrite(os.path.join(img_folder, img_name[:-4] + ".jpg"), np.repeat(img[:, :, np.newaxis], repeats=3, axis=2).type(torch.uint8))

    # eval
    print("Generating evalation snips...")
    for i in tqdm(range(len(eval_paths))):
        img_raw = imageio.v2.imread(eval_paths[i])
        if rs_flag:
            img = torch.from_numpy(rescale(img_raw.astype(np.float16), rs_factor, anti_aliasing=True))
        else:
            img = torch.from_numpy(img_raw)
        img_name = path_leaf(eval_paths[i])

        if label_var != None:
            lb_name = embryo_metadata_df[label_var].iloc[eval_indices[i]].astype(str)
            img_folder = os.path.join(train_dir, "eval", lb_name)
        else:
            img_folder = os.path.join(train_dir, "eval", "0")  # I'm assuming the code will expect a subfolder

        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

        write_flag = True
        if flip_flag:
            snip_ind = np.where(np.asarray(flip_snip_id_vec) == snip_id_list[eval_indices[i]][:-4])[0][0]
            flip_id = int(flip_flag_vec[snip_ind])
            if flip_id == 1:
                img = torch.fliplr(img)
            elif flip_id == -1:
                write_flag = False

        if write_flag:
            imageio.imwrite(os.path.join(img_folder, img_name[:-4] + ".jpg"),
                            np.repeat(img[:, :, np.newaxis], repeats=3, axis=2).type(torch.uint8))
    
    # test
    print("Generating testing snips...")
    for i in tqdm(range(len(test_paths))):
        img_raw = imageio.v2.imread(test_paths[i])
        if rs_flag:
            img = torch.from_numpy(rescale(img_raw.astype(np.float16), rs_factor, anti_aliasing=True))
        else:
            img = torch.from_numpy(img_raw)
        img_name = path_leaf(test_paths[i])
    
        if label_var != None:
            lb_name = embryo_metadata_df[label_var].iloc[test_indices[i]].astype(str)
            img_folder = os.path.join(train_dir, "test", lb_name)
        else:
            img_folder = os.path.join(train_dir, "test", "0")  # I'm assuming the code will expect a subfolder

        write_flag = True
        if flip_flag:
            snip_ind = np.where(np.asarray(flip_snip_id_vec) == snip_id_list[test_indices[i]][:-4])[0][0]
            flip_id = int(flip_flag_vec[snip_ind])
            if flip_id == 1:
                img = torch.fliplr(img)
            elif flip_id == -1:
                write_flag = False

        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

        if write_flag:
            imageio.imwrite(os.path.join(img_folder, img_name[:-4] + ".jpg"),
                            np.repeat(img[:, :, np.newaxis], repeats=3, axis=2).type(torch.uint8))

        
    print("Done.")


if __name__ == "__main__":
    # set path to data
    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
    # root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"

    train_name = "20231120_ds_small"
    label_var = "experiment_date"

    make_pythae_image_snips(root, train_name, label_var="experiment_date", frac_to_use=0.1, rs_factor=0.5)