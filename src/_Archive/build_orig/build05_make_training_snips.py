# We will save the data in folders to mimic the desired example
import os
import torch
import numpy as np
import glob as glob
import pandas as pd
from src.functions.utilities import path_leaf
import skimage.io as io
from tqdm import tqdm
from skimage.transform import rescale

def make_image_snips(root, train_name, label_var=None, rs_factor=1.0, overwrite_flag=False):

    rs_flag = rs_factor != 1
    
    # morphseq_dates = ["20230830", "20230831", "20231207", "20231208"]

    # np.random.seed(r_seed)
    metadata_path = os.path.join(root, "metadata", "combined_metadata_files", '')
    data_path = os.path.join(root, "training_data", "bf_embryo_snips", '')
    
    # read in metadata database
    embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df02.csv"))

    # remove extra columns
    rm_cols = ['time_string', 'Height (um)', 'Width (um)', 'Height (px)', 'Width (px)', 'Time (s)', 'embryos_per_well', 'region_label', 'time_of_addition']
    embryo_metadata_df = embryo_metadata_df.drop(labels=rm_cols, axis=1)

    ###################
    # incorporate manual curation info
    # frame-specific
    curation_df_path = os.path.join(metadata_path, "curation", "curation_df.csv")
    curation_df = pd.read_csv(curation_df_path)
    # check that curation data is up-to-date
    snip_df = embryo_metadata_df.loc[:, ["snip_id"]]
    snip_df = snip_df.merge(curation_df.loc[:, ["snip_id"]], how="left", on="snip_id", indicator=True)
    if np.any(snip_df["_merge"]=="left_only"):
        raise Exception("Latest metadata table contains snips not found in curation dataset. Have you run build04?")
    
    curation_df = curation_df.loc[:, ["snip_id", "manual_stage_hpf", "use_embryo_manual"]].rename(
                    columns={"use_embryo_manual":"use_embryo_flag"})
    manual_update_flags = np.any(~pd.isnull(curation_df.loc[:, ["manual_stage_hpf", "use_embryo_flag"]]).to_numpy(), axis=1)
    curation_df = curation_df.loc[manual_update_flags, :]
    embryo_metadata_df["manual_stage_hpf"] = np.nan
    if curation_df.shape[0] > 0:
        embryo_metadata_df = embryo_metadata_df.set_index("snip_id")
        curation_df = curation_df.set_index("snip_id")
        embryo_metadata_df = curation_df.combine_first(embryo_metadata_df)
        embryo_metadata_df.reset_index(inplace=True)

    # embryo-specific
    emb_curation_df_path = os.path.join(metadata_path, "curation", "embryo_curation_df.csv")
    curation_df_emb = pd.read_csv(emb_curation_df_path)
    # check for manual updates
    manual_update_flags = (curation_df_emb["manual_update_flag"] == 1).to_numpy() | (curation_df_emb["phenotype"] != curation_df_emb["phenotype_orig"]).to_numpy() | \
                    np.any(~pd.isnull(curation_df_emb.loc[:, [ "use_embryo_flag_manual"]]).to_numpy(), axis=1)
    # preserve only entries that have been manually updated
    curation_df_emb = curation_df_emb.loc[manual_update_flags, :]
    if curation_df_emb.shape[0] > 0:
        curation_df_emb.loc[:, "short_pert_name"] = curation_df_emb["phenotype"] + "_" + curation_df_emb["background"]
        curation_df_emb = curation_df_emb.loc[:, ["embryo_id", "short_pert_name", "phenotype", "use_embryo_flag_manual"]].rename(columns={"use_embryo_flag_manual":"use_embryo_flag"})

        # combine with full metadata_table
        embryo_metadata_df = embryo_metadata_df.set_index("embryo_id")
        curation_df_emb = curation_df_emb.set_index("embryo_id")
        embryo_metadata_df = curation_df_emb.combine_first(embryo_metadata_df)
        embryo_metadata_df.reset_index(inplace=True)
    
    # snip_id_vec = embryo_metadata_df["snip_id"].values
    # good_snip_indices = np.where(embryo_metadata_df["use_embryo_flag"].values == True)[0]
    embryo_id_index = np.unique(embryo_metadata_df["embryo_id"].values)

    # shuffle and filter
    # embryo_id_index = np.random.choice(embryo_id_index, len(embryo_id_index), replace=False)
    image_list = []
    image_indices = []

    # itereate through shuffled IDs
    # df_ids_list = []
    for eid in embryo_id_index:

        # extract embryos that match current eid
        # if eid[:8] not in morphseq_dates:
        df_ids = np.where((embryo_metadata_df["embryo_id"].values == eid) & (embryo_metadata_df["use_embryo_flag"].values==True))[0]
        # else:
        #     df_ids = np.where((embryo_metadata_df["embryo_id"].values == eid))[0]
        if len(df_ids) > 0:
            e_date = str(embryo_metadata_df["experiment_date"].iloc[df_ids[0]])
            snip_list = embryo_metadata_df["snip_id"].iloc[df_ids].tolist()
            e_list = [os.path.join(data_path, e_date, s + ".jpg") for s in snip_list]
            i_list = df_ids.tolist()
        
            # remove frames that did not meet QC standards
            image_list += e_list
            image_indices += i_list
            # df_ids_list += df_ids.tolist()
    
    # assign to groups. Note that a few frames from the same embryo may end up split between categories. I think that this is fine
    #n_frames_total = len(image_list) #np.sum(embryo_metadata_df["use_embryo_flag"].values == True)

    # n_train = np.round(train_eval_test[0]*n_frames_total).astype(int)
    # image_list = image_list[:n_train]
    # image_indices = image_indices[:n_train]
    # train_df_indices = df_ids_list[:n_train]

    train_dir = os.path.join(root, "training_data", train_name)
    if not os.path.exists(os.path.join(train_dir, "images")):
        os.makedirs(os.path.join(train_dir, "images"))

    # save copy of the metadata file
    embryo_metadata_df.to_csv(os.path.join(train_dir, "embryo_metadata_df_train.csv"), index=False)

    #################
    # Write snips to file
    
    # training snips
    print("Generating training snips...")
    for i in tqdm(range(len(image_list)), "Exporting image snips to training folder..."):

        img_name = path_leaf(image_list[i])

        if label_var != None:
            lb_name = embryo_metadata_df[label_var].iloc[image_indices[i]].astype(str)
            img_folder = os.path.join(train_dir, "images", lb_name)
        else:
            img_folder = os.path.join(train_dir, "images", "0")  # I'm assuming the code will expect a subfolder

        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

        write_name = os.path.join(img_folder, img_name[:-4] + ".jpg")
        
        if (not os.path.isfile(write_name)) or overwrite_flag:
            img_raw = io.imread(image_list[i])
            if rs_flag:
                img = rescale(img_raw.astype(np.float16), rs_factor, anti_aliasing=True)
            else:
                img = img_raw

            io.imsave(write_name, img.astype(np.uint8), check_contrast=False)

    print("Done.")


if __name__ == "__main__":

    # set path to data
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    train_name = "20240312_test"
    make_image_snips(root, train_name, rs_factor=1.0)