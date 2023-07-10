import os
import numpy as np
import glob
import ntpath
from aicsimageio import AICSImage
from tqdm import tqdm
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


if __name__ == "__main__":

    root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"

    print('Compiling well metadata...')
    build_well_metadata_master(root)