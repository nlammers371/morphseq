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
import scipy
from parfor import pmap
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
import numpy as np
from skimage.morphology import disk, dilation

def export_embryo_snips(r, embryo_metadata_df, dl_rad_um, outscale, outshape):

    # set path to segmentation data
    ff_image_path = os.path.join(root, 'built_keyence_data', 'stitched_FF_images', '')

    # set path to segmentation data
    segmentation_path = os.path.join(root, 'built_keyence_data', 'segmentation', '')

    # make directory for embryo snips
    im_snip_dir = os.path.join(root, 'training_data', 'bf_embryo_snips', '')
    mask_snip_dir = os.path.join(root, 'training_data', 'bf_embryo_masks', '')

    # generate path and image name
    seg_dir_list_raw = glob.glob(segmentation_path + "*")
    seg_dir_list = [s for s in seg_dir_list_raw if os.path.isdir(s)]

    ldb_path = [m for m in seg_dir_list if "ldb" in m][0]
    yolk_path = [m for m in seg_dir_list if "yolk" in m][0]

    row = embryo_metadata_df.iloc[r].copy()

    well = row["well"]
    time_int = row["time_int"]
    date = str(row["experiment_date"])

    im_name = well + f"_t{time_int:04}_ch01_stitch.tif"

    ############
    # Load masks from segmentation
    ############

    # load main embryo mask
    im_ldb_path = os.path.join(ldb_path, date, im_name)
    im_ldb = cv2.imread(im_ldb_path)
    im_ldb = im_ldb[:, :, 0]
    im_ldb = np.round(im_ldb / np.min(im_ldb) - 1).astype(int)
    im_merge = np.zeros(im_ldb.shape, dtype="uint8")
    im_merge[np.where(im_ldb == 1)] = 1
    im_merge[np.where(im_ldb == 2)] = 1
    im_merge_lb = label(im_merge)

    # load yolk mask
    im_yolk_path = os.path.join(yolk_path, date, im_name)
    im_yolk = cv2.imread(im_yolk_path)
    im_yolk = im_yolk[:, :, 0]
    im_yolk = np.round(im_yolk / np.min(im_yolk) - 1).astype(int)

    # get surface area
    px_dim_raw = row["Height (um)"] / row["Height (px)"]  # to adjust for size reduction (need to automate this)
    size_factor_mask = row["Width (px)"] / 640 * 630 / 640
    # px_dim_mask = px_dim_raw * size_factor_mask

    lbi = row["region_label"]  # im_merge_lb[yi, xi]

    assert lbi != 0  # make sure we're not grabbing empty space

    im_merge_ft = (im_merge_lb == lbi).astype(int)
    im_merge_other = ((im_merge_lb > 0) & (im_merge_lb != lbi)).astype(int)

    ############
    # Load raw image
    ############
    im_ff_path = os.path.join(ff_image_path, date, im_name)
    im_ff = cv2.imread(im_ff_path)
    im_ff = im_ff[:, :, 0]

    # rescale masks and image
    im_ff_rs = cv2.resize(im_ff, None, fx=px_dim_raw / outscale, fy=px_dim_raw / outscale)
    mask_emb_rs = cv2.resize(im_merge_ft, im_ff_rs.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    # mask_other_rs = cv2.resize(im_merge_other, im_ff_rs.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    mask_yolk_rs = cv2.resize(im_yolk, im_ff_rs.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    other_pixel_array = im_ff_rs[np.where(mask_emb_rs == 0)]

    rp = regionprops(mask_emb_rs)
    angle = rp[0].orientation
    cm = rp[0].centroid

    # find the orientation that puts yolk at top
    # yr1 = im_ff_rotated = rotate_image(mask_yolk_rs, np.rad2deg(-angle), cm[1], cm[0])
    # cm1 = scipy.ndimage.center_of_mass(yr1, labels=1)
    # yr2 = im_ff_rotated = rotate_image(mask_yolk_rs, np.rad2deg(-angle+np.pi), cm[1], cm[0])
    # cm2 = scipy.ndimage.center_of_mass(yr2, labels=1)
    im_ff_rotated = rotate_image(im_ff_rs, np.rad2deg(-angle))
    im_mask_rotated = rotate_image(mask_emb_rs.astype(np.uint8), np.rad2deg(-angle))
    # im_other_rotated = rotate_image(mask_other_rs.astype(np.uint8), np.rad2deg(-angle), cm[1], cm[0])
    im_yolk_rotated = rotate_image(mask_yolk_rs.astype(np.uint8), np.rad2deg(-angle))

    # extract snip
    dl_rad_px = int(np.ceil(dl_rad_um / px_dim_raw))
    im_mask_dl = dilation(im_mask_rotated, disk(dl_rad_px))

    masked_image = im_ff_rotated.copy()
    masked_image[np.where(im_mask_dl == 0)] = np.random.choice(other_pixel_array, np.sum(im_mask_dl == 0))

    y_indices = np.where(np.max(im_mask_dl, axis=1) == 1)[0]
    x_indices = np.where(np.max(im_mask_dl, axis=0) == 1)[0]
    y_mean = int(np.mean(y_indices))
    x_mean = int(np.mean(x_indices))

    fromshape = im_mask_dl.shape
    raw_range_y = [y_mean - int(outshape[0] / 2), y_mean + int(outshape[0] / 2)]
    from_range_y = np.asarray([np.max([raw_range_y[0], 0]), np.min([raw_range_y[1], fromshape[0]])])
    to_range_y = [0 + (from_range_y[0] - raw_range_y[0]), outshape[0] + (from_range_y[1] - raw_range_y[1])]

    raw_range_x = [x_mean - int(outshape[1] / 2), x_mean + int(outshape[1] / 2)]
    from_range_x = np.asarray([np.max([raw_range_x[0], 0]), np.min([raw_range_x[1], fromshape[1]])])
    to_range_x = [0 + (from_range_x[0] - raw_range_x[0]), outshape[1] + (from_range_x[1] - raw_range_x[1])]

    im_mask_cropped = np.random.choice(other_pixel_array, size=outshape).astype(np.uint8)
    im_mask_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
        masked_image[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

    emb_mask_cropped = np.zeros(outshape).astype(np.uint8)
    emb_mask_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
        im_mask_rotated[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

    yolk_mask_cropped = np.zeros(outshape).astype(np.uint8)
    yolk_mask_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
        im_yolk_rotated[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

    # write to file
    im_name = row["snip_id"]

    write_flag = cv2.imwrite(os.path.join(im_snip_dir, im_name + ".tif"), im_mask_cropped)
    cv2.imwrite(os.path.join(mask_snip_dir, "emb_" + im_name + ".tif"), emb_mask_cropped)
    cv2.imwrite(os.path.join(mask_snip_dir, "emb_" + im_name + ".tif"), yolk_mask_cropped)

    return write_flag

def rotate_image(mat, angle):
    """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
    width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def make_well_names():
    row_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
    well_name_list = []
    for r in row_list:
        well_names = [r + f'{c+1:02}' for c in range(12)]
        well_name_list += well_names

    return well_name_list

def get_images_to_process(meta_df_path, experiment_list, master_df, overwrite_flag):

    # get list of image files that need to be processed
    images_to_process = []
    if (not os.path.isfile(meta_df_path)) or overwrite_flag:
        master_df_update = []
        df_diff = master_df
        for e, experiment_path in enumerate(experiment_list):
            images_to_process += sorted(glob.glob(os.path.join(experiment_path, "*.tif")))

    else:
        master_df_update = pd.read_csv(meta_df_path, index_col=0)
        # master_df_update = master_df_update.iloc[:-250]
        # get list of row indices in new master table need to be processed
        df_all = master_df.merge(master_df_update.drop_duplicates(), on=["well", "experiment_date", "time_int"],
                           how='left', indicator=True)
        diff_indices = np.where(df_all['_merge'].values == 'left_only')[0]
        df_diff = master_df.iloc[diff_indices]

        well_id_list = master_df_update["well"].values
        well_date_list = master_df_update["experiment_date"].values.astype(str)
        well_time_list = master_df_update["time_int"].values

        # imname_chunk_list = []
        # for w in range(len(well_id_list)):
        #     ic = well_id_list[w] + f'_t{well_time_list[w]:04}'
        #     imname_chunk_list.append(ic)

        for e, experiment_path in enumerate(experiment_list):
            ename = path_leaf(experiment_path)
            date_indices = [d for d in range(len(well_date_list)) if str(well_date_list[d]) == ename]
            # get list of tif files to process
            image_list = sorted(glob.glob(os.path.join(experiment_path, "*.tif")))
            for im in image_list:
                # check if well-date combination already exists. I'm treating all time
                iname = path_leaf(im)
                dash_index = iname.find("_")
                well = iname[:dash_index]
                wd_indices = [wd for wd in date_indices if well_id_list[wd] == well]
                t_index = int(iname[dash_index + 2:dash_index + 6])
                wdt_indices = [wdt for wdt in wd_indices if well_time_list[wdt] == t_index]
                if len(wdt_indices) == 0:
                    images_to_process += [im]
                # if (iname[:9] not in imname_chunk_list) or (ename not in well_date_list):
                #     images_to_process += [im]

    df_diff.reset_index(inplace=True)

    return images_to_process, df_diff, master_df_update


def count_embryo_regions(index, image_list, master_df_update, max_sa, min_sa):

    image_path = image_list[index]

    iname = path_leaf(image_path)
    ename = path_leaf(os.path.dirname(image_path))

    # extract metadata from image name
    dash_index = iname.find("_")
    well = iname[:dash_index]
    t_index = int(iname[dash_index + 2:dash_index + 6])

    # find corresponding index in master dataset
    t_indices = np.where(master_df_update["time_int"] == t_index)[0]
    well_indices = np.where(master_df_update[["well"]] == well)[0]
    e_indices = np.where(master_df_update[["experiment_date"]] == int(ename))[0]
    master_index = [i for i in t_indices if (i in well_indices) and (i in e_indices)]
    if len(master_index) != 1:
        raise Exception(
            "Incorect number of matching entries found for " + iname + f". Expected 1, got {len(master_index)}.")
    else:
        master_index = master_index[0]

    row = master_df_update.loc[master_index].copy()

    # load label image
    im = cv2.imread(image_path)
    im = im / np.min(im) - 1
    im = np.round(im[:, :, 0]).astype(int)

    # merge live/dead labels for now
    im_merge = np.zeros(im.shape, dtype="uint8")
    im_merge[np.where(im == 1)] = 1
    im_merge[np.where(im == 2)] = 1
    im_merge_lb = label(im_merge)
    regions = regionprops(im_merge_lb)

    # get surface areas
    sa_vec = np.empty((len(regions),))
    for r, region in enumerate(regions):
        sa_vec[r] = region.area

    sa_vec = sa_vec[np.where(sa_vec <= max_sa)]
    sa_vec = sorted(sa_vec)

    # revise cutoff to ensure we do not track more embryos than initially
    n_prior = int(row.loc["embryos_per_well"])
    if len(sa_vec) > n_prior:
        min_sa_new = np.max([sa_vec[-n_prior], min_sa])
    else:
        min_sa_new = min_sa

    i_pass = 0
    for r in regions:
        sa = r.area
        if (sa >= min_sa_new) and (sa <= max_sa):
            row.loc["e" + str(i_pass) + "_x"] = r.centroid[1]
            row.loc["e" + str(i_pass) + "_y"] = r.centroid[0]
            row.loc["e" + str(i_pass) + "_label"] = r.label
            lb_indices = np.where(im_merge_lb == r.label)
            row.loc["e" + str(i_pass) + "_frac_alive"] = np.mean(im[lb_indices] == 1)

            i_pass += 1

    row.loc["n_embryos_observed"] = i_pass
    row_out = pd.DataFrame(row).transpose()
    return [master_index, row_out]
def do_embryo_tracking(well_id, master_df, master_df_update):
    well_indices = np.where(master_df_update[["well_id"]].values == well_id)[0]

    # check how many embryos we are dealing with
    n_emb_col = master_df_update.loc[well_indices, ["n_embryos_observed"]].values.ravel()

    if np.max(n_emb_col) == 0:  # skip
        df_temp = []

    elif np.max(n_emb_col) == 1:  # no need for tracking
        use_indices = [well_indices[w] for w in range(len(well_indices)) if n_emb_col[w] == 1]

        df_temp = master_df.iloc[use_indices].copy()
        df_temp.reset_index(inplace=True)
        keep_cols = [n for n in df_temp.columns if n != "index"]
        df_temp = df_temp.loc[:, keep_cols]

        df_temp["xpos"] = master_df_update.loc[use_indices, ["e0_x"]].values
        df_temp["ypos"] = master_df_update.loc[use_indices, ["e0_y"]].values
        df_temp["fraction_alive"] = master_df_update.loc[use_indices, ["e0_frac_alive"]].values
        df_temp["region_label"] = master_df_update.loc[use_indices, ["e0_label"]].values
        df_temp.loc[:, "embryo_id"] = well_id + '_e00'

        # if i_pass == 0:
        #     embryo_metadata_df = df_temp.copy()
        # else:
        #     embryo_metadata_df = pd.concat([embryo_metadata_df, df_temp.copy()], axis=0, ignore_index=True)
        # i_pass += 1
    else:  # this case is more complicated
        last_i = np.max(np.where(n_emb_col > 0)[0]) + 1
        first_i = np.min(np.where(n_emb_col > 0)[0])
        track_indices = well_indices[first_i:last_i]
        n_emb = master_df_update.loc[track_indices[0], "n_embryos_observed"].astype(int)
        n_emb_orig = n_emb.copy()

        # initialize helper arrays for tracking
        id_array = np.empty((len(well_indices), n_emb))
        id_array[:] = np.nan
        last_pos_array = np.empty((n_emb, 2))
        last_pos_array[:] = np.nan
        id_array[first_i, :] = range(n_emb)
        for n in range(n_emb):
            last_pos_array[n, 0] = master_df_update.loc[track_indices[0], "e" + str(n) + "_x"]
            last_pos_array[n, 1] = master_df_update.loc[track_indices[0], "e" + str(n) + "_y"]

        # carry out tracking
        for t, ind in enumerate(track_indices[1:]):
            # check how many embryos were detected
            n_emb = n_emb_col[t + 1 + first_i].astype(int)
            if n_emb == 0:
                pass  # note that we carry over last_pos_array
            else:
                curr_pos_array = np.empty((n_emb, 2))
                for n in range(n_emb):
                    curr_pos_array[n, 0] = master_df_update.loc[ind, "e" + str(n) + "_x"]
                    curr_pos_array[n, 1] = master_df_update.loc[ind, "e" + str(n) + "_y"]
                # ensure 2D
                curr_pos_array = np.reshape(curr_pos_array, (n_emb, 2))

                # get pairwise distances
                dist_matrix = pairwise_distances(last_pos_array, curr_pos_array)
                dist_matrix = np.reshape(dist_matrix, (last_pos_array.shape[0], n_emb))

                # get min cost assignments
                from_ind, to_ind = linear_sum_assignment(dist_matrix)

                # update ID assignments
                id_array[t + 1 + first_i, from_ind] = to_ind

                # update positions
                last_pos_array[from_ind, :] = curr_pos_array[to_ind]  # note that unassigned positions carried over

        # carry assignments forward if necessary
        # id_array[t + 2 + first_i:, :] = id_array[t + 1 + first_i, :]

        # use ID array to generate stable embryo IDs
        for n in range(n_emb_orig):
            use_indices = [well_indices[w] for w in range(len(well_indices)) if ~np.isnan(id_array[w, n])]

            use_subindices = [w for w in range(len(well_indices)) if ~np.isnan(id_array[w, n])]

            df_temp = master_df.iloc[use_indices].copy()
            df_temp.reset_index(inplace=True)
            keep_cols = [n for n in df_temp.columns if n != "index"]
            df_temp = df_temp.loc[:, keep_cols]
            for iter, ui in enumerate(use_indices):
                id = int(id_array[use_subindices[iter], n])

                df_temp.loc[iter, "xpos"] = master_df_update.loc[ui, "e" + str(id) + "_x"]
                df_temp.loc[iter, "ypos"] = master_df_update.loc[ui, "e" + str(id) + "_y"]
                df_temp.loc[iter, "fraction_alive"] = master_df_update.loc[ui, "e" + str(id) + "_frac_alive"]
                df_temp.loc[iter, "region_label"] = master_df_update.loc[ui, "e" + str(id) + "_label"]
            df_temp["embryo_id"] = well_id + f'_e{n:02}'

            # if i_pass == 0:
            #     embryo_metadata_df = df_temp.copy()
            # else:
            #     embryo_metadata_df = pd.concat([embryo_metadata_df, df_temp.copy()], axis=0, ignore_index=True)
            # i_pass += 1

    return df_temp


def get_embryo_stats(index, embryo_metadata_df, qc_scale_um, ld_rat_thresh):

    row = embryo_metadata_df.loc[index].copy()

    # generate path and image name
    segmentation_path = os.path.join(root, 'built_keyence_data', 'segmentation', '')
    seg_dir_list_raw = glob.glob(segmentation_path + "*")
    seg_dir_list = [s for s in seg_dir_list_raw if os.path.isdir(s)]

    ldb_path = [m for m in seg_dir_list if "ldb" in m][0]
    focus_path = [m for m in seg_dir_list if "focus" in m][0]
    yolk_path = [m for m in seg_dir_list if "yolk" in m][0]

    well = row["well"]
    time_int = row["time_int"]
    date = str(row["experiment_date"])

    im_name = well + f"_t{time_int:04}_ch01_stitch.tif"

    im_ldb_path = os.path.join(ldb_path, date, im_name)
    im_ldb = cv2.imread(im_ldb_path)
    im_ldb = im_ldb[:, :, 0]
    im_ldb = np.round(im_ldb / np.min(im_ldb) - 1).astype(int)
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
    px_dim_raw = row["Height (um)"] / row["Height (px)"]   # to adjust for size reduction (need to automate this)
    size_factor = row["Width (px)"] / 640 * 630/320
    px_dim = px_dim_raw * size_factor
    qc_scale_px = int(np.ceil(qc_scale_um / px_dim))
    ih, iw = im_yolk.shape
    # yi = np.min([np.max([int(row["ypos"]), 1]), ih])
    # xi = np.min([np.max([int(row["xpos"]), 1]), iw])
    lbi = row["region_label"]# im_merge_lb[yi, xi]

    assert lbi != 0 # make sure we're not grabbing empty space

    im_merge_lb = (im_merge_lb == lbi).astype(int)

    # calculate sa-related metrics
    rg = regionprops(im_merge_lb)
    row.loc["surface_area_um"] = rg[0].area * px_dim ** 2
    row.loc["length_um"] = rg[0].axis_major_length * px_dim
    row.loc["width_um"] = rg[0].axis_minor_length * px_dim

    # calculate speed
    if (row["time_int"] > 1) and index > 0:
        dr = np.sqrt((row["xpos"] - embryo_metadata_df.loc[index - 1, "xpos"]) ** 2 +
                     (row["ypos"] - embryo_metadata_df.loc[index - 1, "ypos"]) ** 2) * px_dim
        dt = row["Time Rel (s)"] - embryo_metadata_df.loc[index - 1, "Time Rel (s)"]
        row.loc["speed"] = dr / dt

    ######
    # now do QC checks
    ######

    # Assess live/dead status
    row.loc["dead_flag"] = row["fraction_alive"] < ld_rat_thresh

    # is there a yolk detected in the vicinity of the embryo body?
    im_intersect = np.multiply((im_yolk == 1)*1, (im_merge_lb == 1)*1)
    row.loc["no_yolk_flag"] = ~np.any(im_intersect)

    # is a part of the embryo mask at or near the image boundary?
    im_trunc = im_merge_lb[qc_scale_px:-qc_scale_px, qc_scale_px:-qc_scale_px]
    row.loc["frame_flag"] = np.sum(im_merge_lb) != np.sum(im_trunc)

    # is there an out-of-focus region in the vicinity of the mask?
    if np.any(im_focus) or np.any(im_ldb == 3):
        im_dist = scipy.ndimage.distance_transform_edt(im_merge_lb == 0)

    if np.any(im_focus):
        min_dist = np.min(im_dist[np.where(im_focus == 1)])
        row.loc["focus_flag"] = min_dist <= 2 * qc_scale_px

    # is there bubble in the vicinity of embryo?
    if np.any(im_ldb == 3):
        min_dist_bubb = np.min(im_dist[np.where(im_ldb == 3)])
        row.loc["bubble_flag"] = min_dist_bubb <= 2 * qc_scale_px

    row_out = pd.DataFrame(row).transpose()
    return row_out

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


def segment_wells(root, min_sa=2500, max_sa=10000, ld_rat_thresh=0.75, qc_scale_um=150, par_flag=False,
                  overwrite_well_stats=False, overwrite_embryo_stats=False):

    # generate paths to useful directories
    metadata_path = os.path.join(root, 'metadata', '')
    segmentation_path = os.path.join(root, 'built_keyence_data', 'segmentation', '')

    # load well-level metadata
    master_df = pd.read_csv(os.path.join(metadata_path, 'master_well_metadata.csv'), index_col=0)

    # get list of segmentation directories
    seg_dir_list_raw = glob.glob(segmentation_path + "*")
    seg_dir_list = [s for s in seg_dir_list_raw if os.path.isdir(s)]

    ###################
    # Track number of embryos and position over time
    ###################

    ldb_path = [m for m in seg_dir_list if "ldb" in m][0]
    # get list of experiments
    experiment_list = sorted(glob.glob(os.path.join(ldb_path, "*")))
    experiment_list = [e for e in experiment_list if "ignore" not in e]

    ckpt1_path = (os.path.join(metadata_path, "embryo_metadata_df_ckpt1.csv"))

    images_to_process, df_to_process, prev_meta_df\
        = get_images_to_process(ckpt1_path, experiment_list, master_df, overwrite_well_stats)

    assert len(images_to_process) == df_to_process.shape[0]

    if len(images_to_process) > 0:
        # initialize empty columns to store embryo information
        master_df_update = df_to_process.copy()
        master_df_update["n_embryos_observed"] = np.nan
        for n in range(4):
            master_df_update["e" + str(n) + "_x"] = np.nan
            master_df_update["e" + str(n) + "_y"] = np.nan
            master_df_update["e" + str(n) + "_label"] = np.nan
            master_df_update["e" + str(n) + "_frac_alive"] = np.nan

        # extract position and live/dead status of each embryo in each well
        emb_df_list = []
        print("Extracting embryo locations...")
        # for i, experiment_path in enumerate(experiment_list):
        #     ename = path_leaf(experiment_path)
        #     # get list of tif files to process
        #     image_list = sorted(glob.glob(os.path.join(experiment_path, "*.tif")))
        if par_flag:
            emb_df_temp = pmap(count_embryo_regions, range(len(images_to_process)),
                               (images_to_process, master_df_update, max_sa, min_sa), rP=0.75)
            emb_df_list += emb_df_temp
        else:
            for index in tqdm(range(len(images_to_process))):
                df_temp = count_embryo_regions(index, images_to_process, master_df_update, max_sa, min_sa)
                emb_df_list.append(df_temp)

        # udate the df
        for e in range(len(emb_df_list)):
            master_index = emb_df_list[e][0]
            row = emb_df_list[e][1]
            master_df_update.loc[master_index, :] = row.iloc[0, :]

        if isinstance(prev_meta_df, pd.DataFrame):
            # prev_table = pd.read_csv(ckpt1_path, index_col=0)
            master_df_update = pd.concat([prev_meta_df, master_df_update], ignore_index=True)
        master_df_update.to_csv(ckpt1_path)

    else:
        master_df_update = pd.read_csv(ckpt1_path, index_col=0)
    # Next, iterate through the extracted positions and use rudimentary tracking to assign embryo instances to stable
    # embryo_id that persists over time

    # get list of unique well instances
    if np.any(np.isnan(master_df_update["n_embryos_observed"].values.astype(float))):
        raise Exception("Missing rows found in metadata df")

    # master_df = master_df.iloc[np.where(~np.isnan(master_df_update["n_embryos_observed"].values.astype(float)))]
    # master_df.reset_index(inplace=True)
    # master_df_update = master_df_update.dropna(subset="n_embryos_observed")
    # master_df_update.reset_index(inplace=True)

    well_id_list = np.unique(master_df_update["well_id"])
    track_df_list = []
    print("Performing embryo tracking...")
    for w in tqdm(range(len(well_id_list))):
        well_id = well_id_list[w]
        df_temp = do_embryo_tracking(well_id, master_df, master_df_update)
        track_df_list.append(df_temp)

    track_df_list = [df for df in track_df_list if isinstance(df, pd.DataFrame)]
    embryo_metadata_df = pd.concat(track_df_list, ignore_index=True)
    # embryo_metadata_df = embryo_metadata_df.iloc[:, 1:]

    ######
    # Add key embryo characteristics and flag QC issues
    ######
    # initialize new variables
    new_cols = ["surface_area_um", "surface_area_um", "length_um", "width_um", "bubble_flag",
                "focus_flag", "frame_flag", "dead_flag", "no_yolk_flag"]
    embryo_metadata_df["surface_area_um"] = np.nan
    embryo_metadata_df["length_um"] = np.nan
    embryo_metadata_df["width_um"] = np.nan
    embryo_metadata_df["bubble_flag"] = False
    embryo_metadata_df["focus_flag"] = False
    embryo_metadata_df["frame_flag"] = False
    embryo_metadata_df["dead_flag"] = False
    embryo_metadata_df["no_yolk_flag"] = False
    # embryo_metadata_df["use_embryo_flag"] = False
    # check for existing embryo metadata
    if os.path.isfile(os.path.join(metadata_path, "embryo_metadata_df.csv")) and not overwrite_embryo_stats:
        embryo_metadata_df_prev = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df.csv"), index_col=0)
        # First, check to see if there are new embryo-well-time entries (rows)
        merge_skel = embryo_metadata_df.loc[:, ["well", "experiment_date", "time_int"]]
        df_all = merge_skel.merge(embryo_metadata_df_prev.drop_duplicates(), on=["well", "experiment_date", "time_int"],
                                 how='left', indicator=True)
        diff_indices = np.where(df_all['_merge'].values == 'left_only')[0].tolist()

        # second, check to see if some or all of the stat columns are already filled in prev table
        for nc in new_cols:
            embryo_metadata_df[nc] = df_all.loc[:, nc]
        nan_indices = np.where(np.isnan(embryo_metadata_df["surface_area_um"].values.astype(float)))[0].tolist()
        indices_to_process = np.unique(diff_indices + nan_indices).tolist()
    else:
        indices_to_process = range(embryo_metadata_df.shape[0])

    print("Extracting embryo stats")
    if par_flag:
        emb_df_list = pmap(get_embryo_stats, indices_to_process,
                                (embryo_metadata_df, qc_scale_um, ld_rat_thresh), rP=0.75)
    else:
        emb_df_list = []
        for index in tqdm(indices_to_process):
            df_temp = get_embryo_stats(index, embryo_metadata_df, qc_scale_um, ld_rat_thresh)
            emb_df_list.append(df_temp)

    for i, ind in enumerate(indices_to_process):
        row_df = emb_df_list[i]
        for nc in new_cols:
            embryo_metadata_df.loc[ind, nc] = row_df[nc].values

    embryo_metadata_df["use_embryo_flag"] = ~(
            embryo_metadata_df["bubble_flag"].values | embryo_metadata_df["focus_flag"].values |
            embryo_metadata_df["frame_flag"].values | embryo_metadata_df["dead_flag"].values |
            embryo_metadata_df["no_yolk_flag"].values)

    embryo_metadata_df.to_csv(os.path.join(metadata_path, "embryo_metadata_df.csv"))

    print("phew")

def extract_embryo_snips(root, outscale=5.66, par_flag=False, outshape=None, dl_rad_um=150):

    if outshape == None:
        outshape = [576, 192]

    # read in metadata
    metadata_path = os.path.join(root, 'metadata', '')
    embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df.csv"), index_col=0)

    # make directory for embryo snips
    im_snip_dir = os.path.join(root, 'training_data', 'bf_embryo_snips', '')
    if not os.path.isdir(im_snip_dir):
        os.makedirs(im_snip_dir)
    mask_snip_dir = os.path.join(root, 'training_data', 'bf_embryo_masks', '')
    if not os.path.isdir(mask_snip_dir):
        os.makedirs(mask_snip_dir)

    # make stable embryo ID
    embryo_metadata_df["snip_id"] = embryo_metadata_df["embryo_id"] + "_" + embryo_metadata_df["time_int"].astype(str)
    export_indices = range(embryo_metadata_df.shape[0])
    if not par_flag:
        for r in tqdm(export_indices):
            export_embryo_snips(r, embryo_metadata_df, dl_rad_um, outscale, outshape)
    else:
        temp = pmap(export_embryo_snips, export_indices, (embryo_metadata_df, dl_rad_um, outscale, outshape), rP=0.25)

        # rotate and crop image
        # Now, we will perform actual image rotation
        # rotatingimage = cv2.warpAffine(
        #     rotateImage, rotationMatrix, (newImageWidth, newImageHeight))


if __name__ == "__main__":

    root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    # root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"

    print('Compiling well metadata...')
    #build_well_metadata_master(root)

    print('Compiling embryo metadata...')
    #segment_wells(root, par_flag=True, overwrite_well_stats=False, overwrite_embryo_stats=False)

    # print('Extracting embryo snips...')
    extract_embryo_snips(root, par_flag=False)