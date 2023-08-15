import os
import numpy as np
import glob
from skimage.measure import label, regionprops
import cv2
import pandas as pd
from _archive.functions_folder.utilities import path_leaf
import scipy
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

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


def segment_wells(root, min_sa=2500, max_sa=10000, ld_rat_thresh=0.75, qc_scale_um=150, overwrite_flag=False):

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
    for e, experiment_path in enumerate([experiment_list[1]]):
        ename = path_leaf(experiment_path)
        # get list of tif files to process
        image_list = sorted(glob.glob(os.path.join(experiment_path, "*.tif")))
        for i, image_path in enumerate(image_list[:3000]):
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
            im = np.round(im[:, :, 0]).astype(int)

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
            if len(sa_vec) > n_prior:
                min_sa_new = np.max([sa_vec[-n_prior], min_sa])
            else:
                min_sa_new = min_sa

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
    master_df = master_df.iloc[np.where(~np.isnan(master_df_update["n_embryos_observed"].values))]
    master_df.reset_index(inplace=True)
    master_df_update = master_df_update.dropna(subset=["n_embryos_observed"])
    master_df_update.reset_index(inplace=True)

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
            df_temp.reset_index(inplace=True)
            df_temp = df_temp.iloc[:, 2:]

            df_temp["xpos"] = master_df_update.loc[use_indices, ["e0_x"]].values
            df_temp["ypos"] = master_df_update.loc[use_indices, ["e0_y"]].values
            df_temp["fraction_alive"] = master_df_update.loc[use_indices, ["e0_frac_alive"]].values
            df_temp["embryo_id"] = well_id + '_e00'

            if i_pass == 0:
                embryo_metadata_df = df_temp.copy()
            else:
                embryo_metadata_df = pd.concat([embryo_metadata_df, df_temp.copy()], axis=0, ignore_index=True)
            i_pass += 1
        else:  # this case is more complicated
            last_i = np.max(np.where(n_emb_col > 1)[0]) + 1
            track_indices = well_indices[:last_i]
            n_emb = master_df_update.loc[track_indices[0], "n_embryos_observed"].astype(int)
            n_emb_orig = n_emb.copy()

            # initialize helper arrays for tracking
            id_array = np.empty((len(track_indices), n_emb))
            id_array[:] = np.nan
            last_pos_array = np.empty((n_emb, 2))
            last_pos_array[:] = np.nan
            id_array[0, :] = range(n_emb)
            for n in range(n_emb):
                last_pos_array[n, 0] = master_df_update.loc[track_indices[0], ["e" + str(n) + "_x"]]
                last_pos_array[n, 1] = master_df_update.loc[track_indices[0], ["e" + str(n) + "_y"]]

            # carry out tracking
            for t, ind in enumerate(track_indices[1:]):
                # check how many embryos were detected
                n_emb = n_emb_col[t+1].astype(int)
                if n_emb == 0:
                    pass  # note that we carry over last_pos_array
                else:
                    curr_pos_array = np.empty((n_emb, 2))
                    for n in range(n_emb):
                        curr_pos_array[n, 0] = master_df_update.loc[ind, ["e" + str(n) + "_x"]]
                        curr_pos_array[n, 1] = master_df_update.loc[ind, ["e" + str(n) + "_y"]]
                    # ensure 2D
                    curr_pos_array = np.reshape(curr_pos_array, (n_emb, 2))

                    # get pairwise distances
                    dist_matrix = pairwise_distances(last_pos_array, curr_pos_array)
                    dist_matrix = np.reshape(dist_matrix, (last_pos_array.shape[0], n_emb))

                    # get min cost assignments
                    from_ind, to_ind = linear_sum_assignment(dist_matrix)

                    # update ID assignments
                    id_array[t+1, from_ind] = to_ind

                    # update positions
                    last_pos_array[from_ind, :] = curr_pos_array[to_ind]  # note that unassigned positions carried over

            # carry assignments forward if necessary
            id_array[t+2:, :] = id_array[t+1, :]

            # use ID array to generate stable embryo IDs
            for n in range(n_emb_orig):
                use_indices = [well_indices[w] for w in range(len(well_indices)) if ~np.isnan(id_array[w, n])]
                use_subindices = [w for w in range(len(well_indices)) if ~np.isnan(id_array[w, n])]
                df_temp = master_df.iloc[use_indices].copy()
                df_temp.reset_index(inplace=True)
                df_temp = df_temp.iloc[:, 2:]
                for iter, ui in enumerate(use_indices):

                    id = int(id_array[use_subindices[iter], n])

                    df_temp.loc[iter, "xpos"] = master_df_update.loc[ui, "e" + str(id) + "_x"]
                    df_temp.loc[iter, "ypos"] = master_df_update.loc[ui, "e" + str(id) + "_y"]
                    df_temp.loc[iter, "fraction_alive"] = master_df_update.loc[ui, "e" + str(id) + "_frac_alive"]
                df_temp["embryo_id"] = well_id + f'_e{n:02}'

                if i_pass == 0:
                    embryo_metadata_df = df_temp.copy()
                else:
                    embryo_metadata_df = pd.concat([embryo_metadata_df, df_temp.copy()], axis=0, ignore_index=True)
                i_pass += 1

    ######
    # Add key embryo characteristics and flag QC issues
    ######
    # initialize new variables
    embryo_metadata_df["surface_area"] = np.nan
    embryo_metadata_df["length"] = np.nan
    embryo_metadata_df["width"] = np.nan
    embryo_metadata_df["speed"] = np.nan
    embryo_metadata_df["bubble_flag"] = False
    embryo_metadata_df["focus_flag"] = False
    embryo_metadata_df["frame_flag"] = False
    embryo_metadata_df["dead_flag"] = False
    embryo_metadata_df["no_yolk_flag"] = False
    # embryo_metadata_df["use_embryo_flag"] = False



    for index, row in embryo_metadata_df.iterrows():

        # generate path and image name
        segmentation_path = os.path.join(root, 'built_keyence_data', 'segmentation', '')
        seg_dir_list_raw = glob.glob(segmentation_path + "*")
        seg_dir_list = [s for s in seg_dir_list_raw if os.path.isdir(s)]

        ldb_path = [m for m in seg_dir_list if "ldb" in m][0]
        focus_path = [m for m in seg_dir_list if "focus" in m][0]
        yolk_path = [m for m in seg_dir_list if "yolk" in m][0]

        well = embryo_metadata_df.loc[index, "well"]
        time_int = embryo_metadata_df.loc[index, "time_int"]
        date = str(embryo_metadata_df.loc[index, "experiment_date"])

        im_name = well + f"_t{time_int:04}_ch01_stitch.tif"

        im_ldb_path = os.path.join(ldb_path, date, im_name)
        im_ldb = cv2.imread(im_ldb_path)
        im_ldb = im_ldb[:, :, 0]
        im_ldb = np.round(im_ldb/np.min(im_ldb) - 1).astype(int)
        im_merge = np.zeros(im_ldb.shape, dtype="uint8")
        im_merge[np.where(im_ldb == 1)] = 1
        im_merge[np.where(im_ldb == 2)] = 1
        im_merge_lb = label(im_merge)

        im_focus_path = os.path.join(focus_path, date, im_name)
        im_focus = cv2.imread(im_focus_path)
        im_focus = im_focus[:, :, 0]
        im_focus = np.round(im_focus / np.min(im_focus) - 1).astype(int)

        im_yolk_path = os.path.join(yolk_path, date, im_name)
        im_yolk = cv2.imread(im_yolk_path)
        im_yolk = im_yolk[:, :, 0]
        im_yolk = np.round(im_yolk / np.min(im_yolk) - 1).astype(int)

        # get surface area
        px_dim = row["Height (um)"] / row["Height (px)"] * 2  # to adjust for size reduction (need to automate this)
        qc_scale_px = int(np.ceil(qc_scale_um / px_dim))
        ih, iw = im_yolk.shape
        yi = np.min([np.max([int(row["ypos"]), 1]), iw])
        xi = np.min([np.max([int(row["xpos"]), 1]), ih])
        lbi = im_merge_lb[yi, xi]
        im_merge_lb = (im_merge_lb == lbi).astype(int)

        # calculate sa-related metrics
        rg = regionprops(im_merge_lb)
        embryo_metadata_df.loc[index, "surface_area_um"] = rg[0].area * px_dim ** 2
        embryo_metadata_df.loc[index, "length_um"] = rg[0].axis_major_length * px_dim
        embryo_metadata_df.loc[index, "width_um"] = rg[0].axis_minor_length * px_dim

        # calculate speed
        if row["time_int"] > 1:
            dr = np.sqrt((embryo_metadata_df.loc[index, "xpos"] - embryo_metadata_df.loc[index-1, "xpos"])**2 +
                         (embryo_metadata_df.loc[index, "ypos"] - embryo_metadata_df.loc[index-1, "ypos"])**2) * px_dim
            dt = embryo_metadata_df.loc[index, "Time Rel (s)"] - embryo_metadata_df.loc[index-1, "Time Rel (s)"]
            embryo_metadata_df.loc[index, "speed"] = dr / dt

        ######
        # now do QC checks
        ######

        # Assess live/dead status
        embryo_metadata_df.loc[index, "dead_flag"] = embryo_metadata_df.loc[index, "fraction_alive"] < ld_rat_thresh

        # is there a yolk detected in the vicinity of the embryo body?
        im_intersect = (im_yolk == 1) & (im_merge_lb == 1)
        embryo_metadata_df.loc[index, "no_yolk_flag"] = ~np.any(im_intersect)

        # is a part of the embryo mask at or near the image boundary?
        im_trunc = im_merge_lb[qc_scale_px:-qc_scale_px, qc_scale_px:-qc_scale_px]
        embryo_metadata_df.loc[index, "frame_flag"] = np.sum(im_merge_lb) != np.sum(im_trunc)

        # is there an out-of-focus region in the vicinity of the mask?
        if np.any(im_focus) or np.any(im_ldb == 3):
            im_dist = scipy.ndimage.distance_transform_edt(im_merge_lb == 0)

        if np.any(im_focus):
            min_dist = np.min(im_dist[np.where(im_focus == 1)])
            embryo_metadata_df.loc[index, "focus_flag"] = min_dist <= 2*qc_scale_px

        # is there bubble in the vicinity of embryo?
        if np.any(im_ldb == 3):
            min_dist_bubb = np.min(im_dist[np.where(im_ldb == 3)])
            embryo_metadata_df.loc[index, "bubble_flag"] = min_dist_bubb <= 2 * qc_scale_px

    embryo_metadata_df["use_embryo_flag"] = ~(
                embryo_metadata_df["bubble_flag"].values | embryo_metadata_df["focus_flag"].values |
                embryo_metadata_df["frame_flag"].values | embryo_metadata_df["dead_flag"].values |
                embryo_metadata_df["no_yolk_flag"].values)

    print("phew")






if __name__ == "__main__":

    root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"

    print('Compiling well metadata...')
    build_well_metadata_master(root)

    print('Extracing embryo metadata...')
    segment_wells(root)