# We will save the data in folders to mimic the desired example
import os
import torch
import numpy as np
import glob as glob
import pandas as pd
from src.functions.utilities import path_leaf
import imageio
from tqdm import tqdm
from skimage.transform import rescale

def make_image_snips(root, train_name, r_seed=371, label_var="experiment_date", train_eval_test=None,
                            frac_to_use=1.0, rs_factor=1.0, test_ids=None, test_dates=None, test_perturbations=None,
                            overwrite_flag=False):


    rs_flag = rs_factor != 1
    
    morphseq_dates = ["20230830", "20230831", "20231207", "20231208"]

    np.random.seed(r_seed)
    metadata_path = os.path.join(root, "metadata", '')
    data_path = os.path.join(root, "training_data", "bf_embryo_snips", '')
    
    # read in metadata database
    embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df_final.csv"), index_col=0)
    
    # get list of training files
    image_list = glob.glob(data_path + "*.jpg")
    snip_id_list = [path_leaf(path) for path in image_list]
    # emb_id_list = [eid[0:16] for eid in snip_id_list]
    
    # randomly partition into train, eval, and test
    # this needs to be done at the level of embryos, not images
    if train_eval_test == None:
        train_eval_test = [0.7, 0.15, 0.15]
    train_eval_test = (np.asarray(train_eval_test)*frac_to_use).tolist()

    # snip_id_vec = embryo_metadata_df["snip_id"].values
    # good_snip_indices = np.where(embryo_metadata_df["use_embryo_flag"].values == True)[0]
    embryo_id_index = np.unique(embryo_metadata_df["embryo_id"].values)

    # check to see if there are any experiment dates or perturb ation types that should be left out of training (kept in test)
    if test_dates is not None:
        eid_date_list = [embryo_metadata_df.loc[e, "embryo_id"] for e in embryo_metadata_df.index if embryo_metadata_df.loc[e, "experiment_date"].astype(str) in test_dates]
        eids_date_test = np.unique(eid_date_list).tolist()
        if test_ids is not None:
            test_ids += eids_date_test
        else:
            test_ids = eids_date_test

    if test_perturbations is not None:
        eid_pert_list = [embryo_metadata_df.loc[e, "embryo_id"] for e in embryo_metadata_df.index if embryo_metadata_df.loc[e, "master_perturbation"] in test_perturbations]
        eids_pert_test = np.unique(eid_pert_list).tolist()
        if test_ids is not None:
            test_ids += eids_pert_test
        else:
            test_ids = eids_pert_test 

    # shuffle and filter
    embryo_id_index_shuffle = np.random.choice(embryo_id_index, len(embryo_id_index), replace=False)
    image_list_shuffle = []
    image_indices_shuffle = []
    
    # if we specified specific embryos to test, remove them now
    if test_ids != None:
        test_ids = np.unique(test_ids)

        test_paths_pre = []
        test_indices_pre = []
        # df_indices_pre = []
        for tid in test_ids:
            if tid[:8] not in morphseq_dates:
                df_ids = np.where((embryo_metadata_df["embryo_id"].values == tid) &
                        (embryo_metadata_df["use_embryo_flag"].values == True))[0]
            else:
                df_ids = np.where((embryo_metadata_df["embryo_id"].values == tid))[0]
                
            snip_list = embryo_metadata_df["snip_id"].iloc[df_ids].tolist()
            e_list = [os.path.join(data_path, s + ".jpg") for s in snip_list]
            i_list = df_ids.tolist()
    
            test_paths_pre += e_list
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
        df_ids = np.where((embryo_metadata_df["embryo_id"].values == eid) & (embryo_metadata_df["use_embryo_flag"].values == True))[0]
        snip_list = embryo_metadata_df["snip_id"].iloc[df_ids].tolist()
        e_list = [os.path.join(data_path, s + ".jpg") for s in snip_list]
        i_list = df_ids.tolist()
    
        # remove frames that did not meet QC standards
        image_list_shuffle += e_list
        image_indices_shuffle += i_list
        # df_ids_list += df_ids.tolist()
    
    # assign to groups. Note that a few frames from the same embryo may end up split between categories. I think that this is fine
    n_frames_total = len(image_list_shuffle) #np.sum(embryo_metadata_df["use_embryo_flag"].values == True)

    n_train = np.round(train_eval_test[0]*n_frames_total).astype(int)
    train_paths = image_list_shuffle[:n_train]
    train_indices = image_indices_shuffle[:n_train]
    # train_df_indices = df_ids_list[:n_train]
    
    n_eval = np.round(train_eval_test[1]*n_frames_total).astype(int)
    eval_paths = image_list_shuffle[n_train:n_train+n_eval]
    eval_indices = image_indices_shuffle[n_train:n_train+n_eval]
    # eval_df_indices = df_ids_list[n_train:n_train+n_eval]

    n_test = np.round(train_eval_test[2]*n_frames_total).astype(int)
    test_paths = test_paths_pre + image_list_shuffle[n_train+n_eval:n_train+n_eval+n_test]
    test_indices = test_indices_pre + image_indices_shuffle[n_train+n_eval:n_train+n_eval+n_test]
    # test_df_indices = df_indices_pre + df_ids_list[n_train+n_eval:n_train+n_eval+n_test]

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
    # make metadata file to keep track of things
    print("Builting training key DF...")
    training_key_df = embryo_metadata_df.loc[:, ["snip_id", "experiment_date", "master_perturbation", "embryo_id", "well_id", "predicted_stage_hpf"]]
    training_key_df["train_cat"] = ""
    training_key_df.loc[train_indices, "train_cat"] = "train"
    training_key_df.loc[eval_indices, "train_cat"] = "eval"
    training_key_df.loc[test_indices, "train_cat"] = "test"
    training_key_df.loc[test_indices_pre, "train_cat"] = "test_pre"

    training_key_df = training_key_df.loc[training_key_df["train_cat"]!="", :]
    training_key_df.to_csv(os.path.join(train_dir, "training_key_df.csv"))

    #################
    # Write snips to file
    
    # training snips
    print("Generating training snips...")
    for i in tqdm(range(len(train_paths))):

        img_name = path_leaf(train_paths[i])

        if label_var != None:
            lb_name = embryo_metadata_df[label_var].iloc[train_indices[i]].astype(str)
            img_folder = os.path.join(train_dir, "train", lb_name)
        else:
            img_folder = os.path.join(train_dir, "train", "0")  # I'm assuming the code will expect a subfolder

        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

        write_name = os.path.join(img_folder, img_name[:-4] + ".jpg")
        
        if (not os.path.isfile(write_name)) or overwrite_flag:
            img_raw = imageio.v2.imread(train_paths[i])
            if rs_flag:
                img = torch.from_numpy(rescale(img_raw.astype(np.float16), rs_factor, anti_aliasing=True))
            else:
                img = torch.from_numpy(img_raw)

            

            imageio.imwrite(write_name, np.repeat(img[:, :, np.newaxis], repeats=3, axis=2).type(torch.uint8))

    # eval
    print("Generating evalation snips...")
    for i in tqdm(range(len(eval_paths))):

        img_name = path_leaf(eval_paths[i])

        if label_var != None:
            lb_name = embryo_metadata_df[label_var].iloc[eval_indices[i]].astype(str)
            img_folder = os.path.join(train_dir, "eval", lb_name)
        else:
            img_folder = os.path.join(train_dir, "eval", "0")  # I'm assuming the code will expect a subfolder

        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

        write_name_eval = os.path.join(img_folder, img_name[:-4] + ".jpg")

        if (not os.path.isfile(write_name_eval)) or overwrite_flag:
            img_raw = imageio.v2.imread(eval_paths[i])
            if rs_flag:
                img = torch.from_numpy(rescale(img_raw.astype(np.float16), rs_factor, anti_aliasing=True))
            else:
                img = torch.from_numpy(img_raw)

            imageio.imwrite(write_name_eval,
                            np.repeat(img[:, :, np.newaxis], repeats=3, axis=2).type(torch.uint8))
    
    # test
    print("Generating testing snips...")
    for i in tqdm(range(len(test_paths))):
        img_name = path_leaf(test_paths[i])

        if label_var != None:
            lb_name = embryo_metadata_df[label_var].iloc[test_indices[i]].astype(str)
            img_folder = os.path.join(train_dir, "test", lb_name)
        else:
            img_folder = os.path.join(train_dir, "test", "0")  # I'm assuming the code will expect a subfolder

        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

        write_name_test = os.path.join(img_folder, img_name[:-4] + ".jpg")

        if (not os.path.isfile(write_name_test)) or overwrite_flag:
            img_raw = imageio.v2.imread(test_paths[i])
            if rs_flag:
                img = torch.from_numpy(rescale(img_raw.astype(np.float16), rs_factor, anti_aliasing=True))
            else:
                img = torch.from_numpy(img_raw)
            
            imageio.imwrite(write_name_test,
                            np.repeat(img[:, :, np.newaxis], repeats=3, axis=2).type(torch.uint8))

        
    print("Done.")


if __name__ == "__main__":
    # set path to data
    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    # root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"

    train_name = "20240204_ds_v2"
    label_var = "experiment_date"
    test_dates = ["20230830", "20230831", "20231207", "20231208"]
    test_perturbations = ["lmx1b"]
    make_image_snips(root, train_name, label_var="experiment_date", frac_to_use=1.0, rs_factor=0.5, test_dates=test_dates, test_perturbations=test_perturbations)