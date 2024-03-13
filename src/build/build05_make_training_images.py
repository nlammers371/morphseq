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

def make_image_snips(root, train_name, label_var=None, rs_factor=1.0, overwrite_flag=False):

    rs_flag = rs_factor != 1
    
    morphseq_dates = ["20230830", "20230831", "20231207", "20231208"]

    # np.random.seed(r_seed)
    metadata_path = os.path.join(root, "metadata", '')
    data_path = os.path.join(root, "training_data", "bf_embryo_snips", '')
    
    # read in metadata database
    embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df_final.csv"), index_col=0)
    
    # get list of training files
    image_list = glob.glob(data_path + "*.jpg")
    snip_id_list = [path_leaf(path) for path in image_list]
    

    # snip_id_vec = embryo_metadata_df["snip_id"].values
    # good_snip_indices = np.where(embryo_metadata_df["use_embryo_flag"].values == True)[0]
    embryo_id_index = np.unique(embryo_metadata_df["embryo_id"].values)

    # shuffle and filter
    embryo_id_index_shuffle = np.random.choice(embryo_id_index, len(embryo_id_index), replace=False)
    image_list_shuffle = []
    image_indices_shuffle = []

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

    # n_train = np.round(train_eval_test[0]*n_frames_total).astype(int)
    # image_list_shuffle = image_list_shuffle[:n_train]
    # image_indices_shuffle = image_indices_shuffle[:n_train]
    # train_df_indices = df_ids_list[:n_train]

    train_dir = os.path.join(root, "training_data", train_name)
    if not os.path.exists(os.path.join(train_dir, "images")):
        os.makedirs(os.path.join(train_dir, "images"))
    
    #################
    # make metadata file to keep track of things
    print("Builting training key DF...")
    training_key_df = embryo_metadata_df.loc[:, ["snip_id", "experiment_date", "master_perturbation", "embryo_id", "well_id", "predicted_stage_hpf"]]
    training_key_df.to_csv(os.path.join(train_dir, "training_key_df.csv"))

    #################
    # Write snips to file
    
    # training snips
    print("Generating training snips...")
    for i in tqdm(range(len(image_list_shuffle))):

        img_name = path_leaf(image_list_shuffle[i])

        if label_var != None:
            lb_name = embryo_metadata_df[label_var].iloc[image_indices_shuffle[i]].astype(str)
            img_folder = os.path.join(train_dir, "images", lb_name)
        else:
            img_folder = os.path.join(train_dir, "images", "0")  # I'm assuming the code will expect a subfolder

        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

        write_name = os.path.join(img_folder, img_name[:-4] + ".jpg")
        
        if (not os.path.isfile(write_name)) or overwrite_flag:
            img_raw = imageio.v2.imread(image_list_shuffle[i])
            if rs_flag:
                img = torch.from_numpy(rescale(img_raw.astype(np.float16), rs_factor, anti_aliasing=True))
            else:
                img = torch.from_numpy(img_raw)

            

            imageio.imwrite(write_name, np.repeat(img[:, :, np.newaxis], repeats=3, axis=2).type(torch.uint8))

    print("Done.")


if __name__ == "__main__":

    # set path to data
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    train_name = "20240312_test"
    make_image_snips(root, train_name, rs_factor=1.0)