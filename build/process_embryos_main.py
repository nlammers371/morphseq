import os
import numpy as np
import glob
import ntpath
from aicsimageio import AICSImage
from tqdm import tqdm
from skimage.measure import label, regionprops, regionprops_table
import cv2
import pandas as pd
from functions.utilities import path_leaf

def make_well_names():
    row_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
    well_name_list = []
    for r in row_list:
        well_names = [r + f'{c+1:02}' for c in range(12)]
        well_name_list += well_names

    return well_name_list
def build_well_metadata_master(root, well_sheets=None):

    if well_sheets == None:
        well_sheets = ["medium", "genotype", "chem_perturbation", "start_age_hpf", "embryos_per_well"]

    metadata_path = os.path.join(root, 'metadata', '')
    well_meta_path = os.path.join(metadata_path, 'well_metadata', '*.xlsx')
    ff_im_path = os.path.join(root, 'built_keyence_data', 'FF_images', '*')

    # Load and contanate well metadata into one long pandas table
    project_list = sorted(glob.glob(ff_im_path))
    project_list = [p for p in project_list if "ignore" not in p]
    for p, project in enumerate(project_list):
        readname = os.path.join(project, 'metadata.csv')
        pname = path_leaf(project)
        temp_table = pd.read_csv(readname, index_col=0)
        temp_table["experiment_date"] = pname
        temp_table["experiment_id"] = p
        if p == 0:
            master_well_table = temp_table.copy()
        else:
            master_well_table = pd.concat([master_well_table, temp_table], axis=0, ignore_index=True)

    # join on data from experiment table
    exp_table = pd.read_csv(os.path.join(metadata_path, 'experiment_metadata.csv'))
    exp_table = exp_table[["experiment_id", "start_date", "temperature", "use_flag"]]

    master_well_table = master_well_table.merge(exp_table, on="experiment_id", how='left')
    if master_well_table['use_flag'].isnull().values.any():
        raise Exception("Error: mismatching experiment IDs between experiment- and well-level metadata")

    # pull metadata from individual well sheets
    project_list_well = sorted(glob.glob(well_meta_path))
    well_name_list = make_well_names()
    for p, project in enumerate(project_list_well):
        pname = path_leaf(project)
        if "$" not in pname:
            date_string = pname[:8]
            # read in excel file
            xl_temp = pd.ExcelFile(project)
            # sheet_names = xl_temp.sheet_names  # see all sheet names
            well_df = pd.DataFrame(well_name_list, columns=["well"])

            for sheet in well_sheets:
                sheet_temp = xl_temp.parse(sheet)  # read a specific sheet to DataFrame
                well_df[sheet] = sheet_temp.iloc[0:8, 1:13].values.ravel()
            well_df["experiment_date"] = date_string
            if p == 0:
                long_df = well_df.copy()
            else:
                long_df = pd.concat(([long_df, well_df]), axis=0, ignore_index=True)
    # add to main dataset
    master_well_table = master_well_table.merge(long_df, on=["well", "experiment_date"], how='left')

    if master_well_table[well_sheets[0]].isnull().values.any():
        raise Exception("Error: missing well-specific metadata")

    # subset columns
    all_cols = master_well_table.columns
    rm_cols = ["start_date"]
    keep_cols = [col for col in all_cols if col not in rm_cols]
    master_well_table = master_well_table[keep_cols]

    # calculate approximate stage using linear formula from Kimmel et al 1995 (is there a better formula out there?)
    # dev_time = actual_time*(0.055 T - 0.57) where T is in Celsius...
    master_well_table["predicted_stage_hpf"] = master_well_table["start_age_hpf"] + \
                                               master_well_table["Time Rel (s)"]/3600*(0.055*22-0.57)  # rough estimate for room temp

    # generate new master index
    master_well_table["well_id"] = master_well_table["experiment_date"] + "_" + master_well_table["well"]
    cols = master_well_table.columns.values.tolist()
    cols_new = [cols[-1]] + cols[:-1]
    master_well_table = master_well_table[cols_new]

    # save to file
    master_well_table.to_csv(os.path.join(metadata_path, 'master_well_metadata.csv'))

    return {}


def segment_wells(root, min_sa=2500, max_sa=10000, overwrite_flag=False):

    # generate paths to useful directories
    metadata_path = os.path.join(root, 'metadata', '')
    segmentation_path = os.path.join(root, 'built_keyence_data', 'segmentation', '')

    # load well-level metadata
    master_df = pd.read_csv(os.path.join(metadata_path, 'master_well_metadata.csv'), index_col=0)

    # get list of segmentation directories
    seg_dir_list_raw = glob.glob(segmentation_path + "*")
    seg_dir_list = [s for s in seg_dir_list_raw if os.path.isdir(s)]

    ######
    # Track number of embryos and position over time
    ######
    ldb_path = [m for m in seg_dir_list if "ldb" in m][0]
    # get list of experiments
    experiment_list = sorted(glob.glob(os.path.join(ldb_path, "*")))
    experiment_list = [e for e in experiment_list if "ignore" not in e]

    # initialize empty columns to store embryo information
    master_df_update = master_df.copy()
    master_df_update["n_embryos_observed"] = np.nan
    for n in range(4):
        master_df_update["e" + str(n) + "_x"] = np.nan
        master_df_update["e" + str(n) + "_y"] = np.nan
        master_df_update["e" + str(n) + "_frac_alive"] = np.nan

    # extract position and live/dead status of each embryo in each well
    for e, experiment_path in enumerate(experiment_list):
        ename = path_leaf(experiment_path)
        # get list of tif files to process
        image_list = sorted(glob.glob(os.path.join(experiment_path, "*.tif")))
        for i, image_path in enumerate(image_list):
            iname = path_leaf(image_path)

            # extract metadata from image name
            dash_index = iname.find("_")
            well = iname[:dash_index]
            t_index = int(iname[dash_index+2:dash_index+6])

            # find corresponding index in master dataset
            t_indices = np.where(master_df_update["time_int"] == t_index)[0]
            well_indices = np.where(master_df_update[["well"]] == well)[0]
            e_indices = np.where(master_df_update[["experiment_date"]] == int(ename))[0]
            master_index = [i for i in t_indices if (i in well_indices) and (i in e_indices)]
            if len(master_index) != 1:
                raise Exception("Incorect number of matching entries found for " + iname + f". Expected 1, got {len(master_index)}." )
            else:
                master_index = master_index[0]

            # load label image
            im = cv2.imread(image_path)
            im = im/np.min(im) - 1
            im = im[:, :, 0].astype(int)

            # merge live/dead labels for now
            im_merge = np.zeros(im.shape, dtype="uint8")
            im_merge[np.where(im == 1)] = 1
            im_merge[np.where(im == 2)] = 1
            im_merge_lb = label(im_merge)
            regions = regionprops(im_merge_lb)

            # get surface areas
            sa_vec = np.empty((len(regions), ))
            for r, region in enumerate(regions):
                sa_vec[r] = region.area

            sa_vec = sa_vec[np.where(sa_vec <= max_sa)]
            sa_vec = sorted(sa_vec)

            # revise cutoff to ensure we do not track more embryos than initially
            n_prior = int(master_df_update.loc[master_index, ["embryos_per_well"]].values[0])
            min_sa_new = np.max([sa_vec[-n_prior], min_sa])


            i_pass = 0
            for r in regions:
                sa = r.area
                if (sa >= min_sa_new) and (sa <= max_sa):
                    master_df_update.loc[master_index, ["e" + str(i_pass) + "_x"]] = r.centroid[1]
                    master_df_update.loc[master_index, ["e" + str(i_pass) + "_y"]] = r.centroid[0]
                    lb_indices = np.where(im_merge_lb == r.label)
                    master_df_update.loc[master_index, ["e" + str(i_pass) + "_frac_alive"]] = np.mean(im[lb_indices] == 1)

                    i_pass += 1

            master_df_update.loc[master_index, ["n_embryos_observed"]] = i_pass

    # Next, iterate through the extracted positions and use rudimentary tracking to assign embryo instances to stable
    # embryo_id that persists over time

    # get list of uniqure well instances
    well_id_list = np.unique(master_df_update["well_id"])
    i_pass = 0
    for w, well_id in enumerate(well_id_list):
        well_indices = np.where(master_df_update[["well_id"]].values == well_id)[0]

        # check how many embryos we are dealing with
        n_emb_col = master_df_update.loc[well_indices, ["n_embryos_observed"]].values.ravel()

        if np.max(n_emb_col) == 0:  # skip
            pass
        elif np.max(n_emb_col) == 1:  # no need for tracking
            use_indices = [well_indices[w] for w in range(len(well_indices)) if n_emb_col[w] == 1]
            df_temp = master_df.iloc[use_indices].copy()
            df_temp["xpos"] = master_df_update.loc[use_indices, ["e1_x"]]
            df_temp["ypos"] = master_df_update.loc[use_indices, ["e1_y"]]
            df_temp["fraction_alive"] = master_df_update.loc[use_indices, ["e1_fraction_alive"]]
            if i_pass == 0:
                embryo_metadata_df = df_temp.copy()
            else:
                embryo_metadata_df = pd.concat([embryo_metadata_df, df_temp.copy()], axis=0, ignore_index=True)
            i_pass += 1
        else: # this case is more complicated 
            print("track")


if __name__ == "__main__":

    root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"

    print('Compiling well metadata...')
    build_well_metadata_master(root)