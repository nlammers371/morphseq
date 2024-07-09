import os
import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours
import skimage
import cv2
import pandas as pd
from src.functions.utilities import path_leaf
from skimage.morphology import disk, binary_closing, remove_small_objects
import scipy
# from parfor import pmap
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
import skimage
from scipy.stats import truncnorm
import numpy as np
import skimage.io as io
import multiprocessing
from functools import partial
from tqdm.contrib.concurrent import process_map 
import time
from pathlib import Path

# def get_fov_size_px(row):
#     if row["microscope"] == "YX1":
#         if row["Height (um)"] / row["Height (px)"] < 2:
#             ff_size = 10086912
#         else:
#             ff_size = 10086912 / 4    

#     elif row["microscope"] == "keyence":
#         if row["experiment_date"].astype(str) in ["20231207"]:
#             ff_size = 4536000
#         elif row["experiment_date"].astype(str) in ["20230830", "20230831", "20231208"]:
#             ff_size = 1134000
#         elif row["Width (px)"] >= 800:
#             ff_size = 2.25*718624
#         else:
#             ff_size = 718624

#     return ff_size

def estimate_image_background(root, embryo_metadata_df, bkg_seed=309, n_bkg_samples=100):

    np.random.seed(bkg_seed)
    bkg_sample_indices = np.random.choice(range(embryo_metadata_df.shape[0]), n_bkg_samples, replace=True)
    bkg_pixel_list = []

    for r in tqdm(range(len(bkg_sample_indices)), "Estimating background..."):
        sample_i = bkg_sample_indices[r]
        row = embryo_metadata_df.iloc[sample_i].copy()

        # set path to segmentation data
        ff_image_path = os.path.join(root, 'built_image_data', 'stitched_FF_images', '')
        segmentation_path = os.path.join(root, 'built_image_data', 'segmentation', '')

        # generate path and image name
        seg_dir_list_raw = glob.glob(segmentation_path + "*")
        seg_dir_list = [s for s in seg_dir_list_raw if os.path.isdir(s)]

        emb_path = [m for m in seg_dir_list if "mask" in m][0]
        via_path = [m for m in seg_dir_list if "via" in m][0]

        well = row["well"]
        time_int = row["time_int"]
        date = str(row["experiment_date"])

        ############
        # Load masks from segmentation
        ############
        stub_name = well + f"_t{time_int:04}*"
        im_emb_path = glob.glob(os.path.join(emb_path, date, stub_name))[0]
        im_via_path = glob.glob(os.path.join(via_path, date, stub_name))[0]

        # load main embryo mask
        # im_emb_path = os.path.join(emb_path, date, lb_name)
        im_mask = io.imread(im_emb_path)
        im_mask = np.round(im_mask / 255 * 2 - 1).astype(int)

        im_via = io.imread(im_via_path)
        im_via = np.round(im_via / 255 * 2 - 1).astype(int)

        im_bkg = np.ones(im_mask.shape, dtype="uint8")
        im_bkg[np.where(im_mask == 1)] = 0
        im_bkg[np.where(im_via == 1)] = 0

        # load image
        im_ff_path = glob.glob(os.path.join(ff_image_path, date, stub_name))[0] #os.path.join(ff_image_path, date, im_name)
        im_ff = io.imread(im_ff_path)
        if im_ff.shape[0] < im_ff.shape[1]:
            im_ff = im_ff.transpose(1, 0)
        if im_ff.dtype != "uint8":
            im_ff = skimage.util.img_as_ubyte(im_ff)

        # extract background pixels
        bkg_pixels = im_ff[np.where(im_bkg == 1)].tolist()

        bkg_pixel_list += bkg_pixels

    # get mean and standard deviation
    px_mean = np.mean(bkg_pixel_list)
    px_std = np.std(bkg_pixel_list)

    return px_mean, px_std


def export_embryo_snips(r, root, embryo_metadata_df, dl_rad_um, outscale, outshape, px_mean, px_std,
                        overwrite_flag=False, close_radius=15):

    # set path to segmentation data
    ff_image_path = os.path.join(root, 'built_image_data', 'stitched_FF_images', '')

    # set path to segmentation data
    segmentation_path = os.path.join(root, 'built_image_data', 'segmentation', '')

    # make directory for embryo snips
    im_snip_dir = os.path.join(root, 'training_data', 'bf_embryo_snips', '')
    mask_snip_dir = os.path.join(root, 'training_data', 'bf_embryo_masks', '')

    # generate path and image name
    seg_dir_list_raw = glob.glob(segmentation_path + "*")
    seg_dir_list = [s for s in seg_dir_list_raw if os.path.isdir(s)]

    emb_path = [m for m in seg_dir_list if "mask" in m][0]
    yolk_path = [m for m in seg_dir_list if "yolk" in m][0]

    row = embryo_metadata_df.iloc[r].copy()

    # write to file
    im_name = row["snip_id"]

    ff_save_path = os.path.join(im_snip_dir, im_name + ".jpg")
    ff_save_path_uc = os.path.join(im_snip_dir[:-1] + "_uncropped", im_name + ".jpg")
    if not os.path.isdir(im_snip_dir[:-1] + "_uncropped"):
        os.makedirs(im_snip_dir[:-1] + "_uncropped")

    if (not os.path.isfile(ff_save_path)) | overwrite_flag:

        well = row["well"]
        time_int = row["time_int"]
        date = str(row["experiment_date"])

        ############
        # Load masks from segmentation
        ############
        im_stub = well + f"_t{time_int:04}*"
        # load main embryo mask
        im_emb_path = glob.glob(os.path.join(emb_path, date, im_stub))[0]
        im_mask = io.imread(im_emb_path)
        im_mask = np.round(im_mask / 255 * 2 - 1).astype(np.uint8)
        # im_mask = np.zeros(im_ldb.shape, dtype="uint8")
        # im_mask[np.where(im_ldb == 1)] = 1
        # im_mask[np.where(im_ldb == 2)] = 1
        im_mask_lb = label(im_mask)

        # load yolk mask
        im_yolk_path = glob.glob(os.path.join(yolk_path, date, im_stub))[0]
        im_yolk = io.imread(im_yolk_path)
        im_yolk = np.round(im_yolk / 255).astype(np.uint8)
        if np.any(im_yolk == 1):
            im_yolk = skimage.morphology.remove_small_objects(im_yolk.astype(bool), min_size=75).astype(int)  # remove small stuff

        # get surface area
        px_dim_raw = row["Height (um)"] / row["Height (px)"]  # to adjust for size reduction (need to automate this)

        lbi = row["region_label"]  # im_mask_lb[yi, xi]

        assert lbi != 0  # make sure we're not grabbing empty space

        im_mask_ft = (im_mask_lb == lbi).astype(int)

        # apply simple morph operations to fill small holes
        i_disk = disk(close_radius)
        im_mask_ft = binary_closing(im_mask_ft, i_disk).astype(int)

        # filter out yolk regions that don't contact the embryo ROI
        im_intersect = np.multiply(im_yolk * 1, im_mask_ft * 1)

        if np.sum(im_intersect) < 10:
            im_yolk = np.zeros(im_yolk.shape).astype(int)
        else:
            y_lb = label(im_yolk)
            lbu = np.unique(y_lb[np.where(im_intersect)])
            if len(lbu) == 1:
                im_yolk = (y_lb == lbu[0]).astype(int)
            else:
                i_lb = label(im_intersect)
                rgi = regionprops(i_lb)
                a_vec = [r.area for r in rgi]
                i_max = np.argmax(a_vec)
                lu = np.unique(y_lb[np.where(i_lb == i_max+1)])
                im_yolk = (y_lb == lu[0])*1

        # im_mask_other = ((im_mask_lb > 0) & (im_mask_lb != lbi)).astype(int)

        ############
        # Load raw image
        ############
        im_ff_path = glob.glob(os.path.join(ff_image_path, date, im_stub))[0]
        im_ff = io.imread(im_ff_path)
        if im_ff.shape[0] < im_ff.shape[1]:
            im_ff = im_ff.transpose(1,0)

        if im_ff.dtype != "uint8":
            im_ff = skimage.util.img_as_ubyte(im_ff)
        # rescale masks and image
        im_ff_rs = cv2.resize(im_ff, None, fx=px_dim_raw / outscale, fy=px_dim_raw / outscale)
        mask_emb_rs = cv2.resize(im_mask_ft, im_ff_rs.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        # mask_other_rs = cv2.resize(im_mask_other, im_ff_rs.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        mask_yolk_rs = cv2.resize(im_yolk, im_ff_rs.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        rp = regionprops(mask_emb_rs)
        angle = rp[0].orientation
        # cm = rp[0].centroid

        # find the orientation that puts yolk at top
        er1 = rotate_image(mask_emb_rs, np.rad2deg(-angle))
        e_cm1 = scipy.ndimage.center_of_mass(er1, labels=1)
        if np.any(mask_yolk_rs):
            yr1 = rotate_image(mask_yolk_rs, np.rad2deg(-angle))
            y_cm1 = scipy.ndimage.center_of_mass(yr1, labels=1)
            e_cm1 = scipy.ndimage.center_of_mass(er1, labels=1)
            if (e_cm1[0] - y_cm1[0]) >= 0:
                angle_to_use = -angle
            else:
                angle_to_use = -angle+np.pi
        else:
            y_indices = np.where(np.max(er1, axis=1))[0]
            vert_rat = np.sum(y_indices > e_cm1[0]) / len(y_indices)
            if vert_rat >= 0.5:
                angle_to_use = -angle
            else:
                angle_to_use = -angle+np.pi

        im_ff_rotated = rotate_image(im_ff_rs, np.rad2deg(angle_to_use))
        im_mask_rotated = rotate_image(mask_emb_rs.astype(np.uint8), np.rad2deg(angle_to_use))
        # im_other_rotated = rotate_image(mask_other_rs.astype(np.uint8), np.rad2deg(-angle), cm[1], cm[0])
        im_yolk_rotated = rotate_image(mask_yolk_rs.astype(np.uint8), np.rad2deg(angle_to_use))

        # extract snip
        # im_dist = scipy.ndimage.distance_transform_edt(1 * (im_mask_rotated == 0))
        # im_mask_dl = 1*(im_dist <= dl_rad_px)

        # masked_image = im_ff_rotated.copy()
        # masked_image[np.where(im_mask_dl == 0)] = np.random.choice(other_pixel_array, np.sum(im_mask_dl == 0))

        y_indices = np.where(np.max(im_mask_rotated, axis=1) == 1)[0]
        x_indices = np.where(np.max(im_mask_rotated, axis=0) == 1)[0]
        y_mean = int(np.mean(y_indices))
        x_mean = int(np.mean(x_indices))

        fromshape = im_mask_rotated.shape
        raw_range_y = [y_mean - int(outshape[0] / 2), y_mean + int(outshape[0] / 2)]
        from_range_y = np.asarray([np.max([raw_range_y[0], 0]), np.min([raw_range_y[1], fromshape[0]])])
        to_range_y = [0 + (from_range_y[0] - raw_range_y[0]), outshape[0] + (from_range_y[1] - raw_range_y[1])]

        raw_range_x = [x_mean - int(outshape[1] / 2), x_mean + int(outshape[1] / 2)]
        from_range_x = np.asarray([np.max([raw_range_x[0], 0]), np.min([raw_range_x[1], fromshape[1]])])
        to_range_x = [0 + (from_range_x[0] - raw_range_x[0]), outshape[1] + (from_range_x[1] - raw_range_x[1])]

        im_cropped = np.zeros(outshape).astype(np.uint8)
        im_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
            im_ff_rotated[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

        emb_mask_cropped = np.zeros(outshape).astype(np.uint8)
        emb_mask_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
            im_mask_rotated[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

        yolk_mask_cropped = np.zeros(outshape).astype(np.uint8)
        yolk_mask_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
            im_yolk_rotated[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

        # im_dist_cropped = np.zeros(outshape).astype(np.uint8)
        # im_dist_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
        #     im_dist[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]


        # fill holes in embryo and yolk masks
        emb_mask_cropped = scipy.ndimage.binary_fill_holes(emb_mask_cropped).astype(np.uint8)
        yolk_mask_cropped = scipy.ndimage.binary_fill_holes(yolk_mask_cropped).astype(np.uint8)

        # calculate the distance transform
        im_dist_cropped = scipy.ndimage.distance_transform_edt(1 * (emb_mask_cropped == 0))

        # crop out background
        dl_rad_px = int(np.ceil(dl_rad_um / outscale))
        # dl_rad_px = 0
        # im_dist_cropped = im_dist_cropped * dl_rad_px**-1
        # im_dist_cropped[np.where(im_dist_cropped > 2)] = 1
        # noise_array = np.random.normal(px_mean, px_std, outshape)
        noise_array_raw = np.reshape(truncnorm.rvs(-px_mean/px_std, 4, size=outshape[0]*outshape[1]), outshape)
        noise_array = noise_array_raw*px_std + px_mean
        noise_array[np.where(noise_array < 0)] = 0 # This is redundant, but just in case someone fiddles with the above distributioon
        # noise_array_scaled = np.multiply(noise_array, im_dist_cropped).astype(np.uint8)

        im_masked_cropped = im_cropped.copy()
        # im_masked_cropped += noise_array_scaled # [np.where(emb_mask_cropped == 0)] = np.random.choice(other_pixel_array, np.sum(emb_mask_cropped == 0)).astype(np.uint8)
        im_masked_cropped[np.where(im_dist_cropped > dl_rad_px)] = np.round(noise_array[np.where(im_dist_cropped > dl_rad_px)]).astype(np.uint8)
        # im_masked_cropped[np.where(im_masked_cropped < 0)] = 0
        im_masked_cropped[np.where(im_masked_cropped > 255)] = 255    # NL: I think this is redundant but will leave it
        im_masked_cropped = np.round(im_masked_cropped).astype(np.uint8)

        # check whether we cropped out part of the embryo
        out_of_frame_flag = np.sum(emb_mask_cropped == 1) / np.sum(im_mask_rotated == 1) < 0.99

        # write to file
        # im_name = row["snip_id"]

        io.imsave(ff_save_path, im_masked_cropped, check_contrast=False)
        io.imsave(ff_save_path_uc, im_cropped, check_contrast=False)
        io.imsave(os.path.join(mask_snip_dir, "emb_" + im_name + ".jpg"), emb_mask_cropped, check_contrast=False)
        io.imsave(os.path.join(mask_snip_dir, "yolk_" + im_name + ".jpg"), yolk_mask_cropped, check_contrast=False)

    else:
        out_of_frame_flag = -1


    return out_of_frame_flag

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
        master_df_to_update = []
        df_diff = master_df

    else:
        master_df_to_update = pd.read_csv(meta_df_path, index_col=0)
        # master_df_update = master_df_update.iloc[:-250]
        # get list of row indices in new master table need to be processed
        df_all = master_df.merge(master_df_to_update.drop_duplicates(), on=["well", "experiment_date", "time_int"],
                           how='left', indicator=True)
        diff_indices = np.where(df_all['_merge'].values == 'left_only')[0]
        df_diff = master_df.iloc[diff_indices].drop_duplicates()

    # unprocessed_date_index = np.unique(df_diff["experiment_date"]).astype(str)
    well_id_list = df_diff["well"].values
    well_date_list = df_diff["experiment_date"].values.astype(str)
    well_time_list = df_diff["time_int"].values

    for e, experiment_path in enumerate(experiment_list):
        ename = path_leaf(experiment_path)
        date_indices = np.where(well_date_list == ename)[0]
        # get channel info
        image_list_full = sorted(glob.glob(os.path.join(experiment_path, "*.jpg")))
        im_name_test = path_leaf(image_list_full[0])
        image_suffix = im_name_test[9:]
        # get list of image prefixes
        im_names = [os.path.join(experiment_path, well_id_list[i] + f"_t{well_time_list[i]:04}" + image_suffix) for i in date_indices]
        images_to_process += im_names
        # for im in image_list:
        #     # check if well-date combination already exists. I'm treating all time
        #     iname = path_leaf(im)
        #     # iname = iname.replace("ff_", "")
        #     dash_index = iname.find("_")
        #     well = iname[:dash_index]
        #     wd_indices = [wd for wd in date_indices if well_id_list[wd] == well]
        #     t_index = int(iname[dash_index + 2:dash_index + 6])
        #     wdt_indices = [wdt for wdt in wd_indices if well_time_list[wdt] == t_index]
        #     if len(wdt_indices) == 0:
        #         images_to_process += [im]
             

        df_diff.reset_index(inplace=True, drop=True)

    # get list of FF and mask sizes
    
    image_size_vec = np.empty(len(images_to_process))
    mask_size_vec = np.empty(len(images_to_process))

    if len(images_to_process) > 0:
        image_root = str(Path(images_to_process[0]).parents[3])
        experiment_dates = [exp.split("/")[-2] for exp in images_to_process]
        date_index = np.unique(experiment_dates)
        # load the first FF image from each experiment folder
        for d, date in enumerate(date_index):
            ff_path = os.path.join(image_root, "stitched_FF_images", date, "")
            ff_images = glob.glob(ff_path + "*.png")

            date_indices = np.where(np.asarray(experiment_dates) == date)[0]

            sample_image = io.imread(ff_images[0])
            sample_mask = io.imread(images_to_process[date_indices[0]])

            image_size_vec[date_indices] = sample_image.size
            mask_size_vec[date_indices] = sample_mask.size

    return images_to_process, df_diff, master_df_to_update, image_size_vec, mask_size_vec


def count_embryo_regions(index, image_list, master_df_update, max_sa_um, min_sa_um):

    image_path = image_list[index]

    iname = path_leaf(image_path)
    iname = iname.replace("ff_", "")
    ename = path_leaf(os.path.dirname(image_path))
    ename = ename[:8]       

    # extract metadata from image name
    dash_index = iname.find("_")
    well = iname[:dash_index]
    t_index = int(iname[dash_index + 2:dash_index + 6])

    entry_filter = (master_df_update["time_int"].values == t_index) & \
                   (master_df_update["well"].values == well) & (master_df_update["experiment_date"].values == int(ename))
    master_index = master_df_update.index[entry_filter] #[i for i in t_indices if (i in well_indices) and (i in e_indices)]
    if len(master_index) != 1:
        raise Exception(
            "Incorrect number of matching entries found for " + iname + f". Expected 1, got {len(master_index)}.")
    else:
        master_index = master_index[0]

    row = master_df_update.loc[master_index].copy()

    ff_size = row['FOV_size_px']

    # load label image
    im = io.imread(image_path)
    im_mask = (np.round(im / 255 * 2) - 1).astype(np.uint8)

    # load viability image
    seg_path = os.path.dirname(os.path.dirname(os.path.dirname(image_path)))
    date_dir = path_leaf(os.path.dirname(image_path))
    im_stub = path_leaf(image_path)[:9]
    via_dir = glob.glob(os.path.join(seg_path, "via_*"))[0]
    via_path = glob.glob(os.path.join(via_dir, date_dir, im_stub + "*"))[0]
    im_via = io.imread(via_path)
    im_via = (np.round(im_via / 255 * 2) - 1).astype(np.uint8)
    
    # make a combined mask
    cb_mask = np.ones_like(im_mask)
    cb_mask[np.where(im_mask == 1)] = 1  # alive
    cb_mask[np.where((im_mask == 1) & (im_via == 1))] = 2  # dead
    
    # merge live/dead labels for now
    # im_mask = np.zeros(im.shape, dtype="uint8")
    
    im_mask_lb = label(im_mask)
    regions = regionprops(im_mask_lb)

    # recalibrate things relative to "standard" dimensions
    pixel_size_raw = row["Height (um)"] / row["Height (px)"]
    # im_area_um2 = pixel_size_raw**2 * ff_size   # to adjust for size reduction 
    lb_size = im_mask_lb.size

    sa_vec = np.asarray([rg["Area"] for rg in regions]) * ff_size / lb_size * pixel_size_raw**2
    sa_filter = (sa_vec <= max_sa_um) & (sa_vec >= min_sa_um)
    sa_ranks = len(sa_vec) - np.argsort(sa_vec) - 1

    i_pass = 0
    for i, r in enumerate(regions):
        # sa = r.area
        if sa_filter[i] & (sa_ranks[i] < 4): #(sa >= min_sa_um_new) and (sa <= max_sa_um):
            row.loc["e" + str(i_pass) + "_x"] = r.centroid[1]
            row.loc["e" + str(i_pass) + "_y"] = r.centroid[0]
            row.loc["e" + str(i_pass) + "_label"] = r.label
            lb_indices = np.where(im_mask_lb == r.label)
            row.loc["e" + str(i_pass) + "_frac_alive"] = np.mean(cb_mask[lb_indices] == 1)

            i_pass += 1

    row.loc["n_embryos_observed"] = i_pass
    row_out = pd.DataFrame(row).transpose()
    return [master_index, row_out]

def do_embryo_tracking(well_id, master_df, master_df_update):

    well_indices = np.where(master_df_update[["well_id"]].values == well_id)[0]
    temp_cols = master_df.columns.tolist() + ["FOV_size_px"]
    # check how many embryos we are dealing with
    n_emb_col = master_df_update.loc[well_indices, ["n_embryos_observed"]].values.ravel()

    if np.max(n_emb_col) == 0:  # skip
        df_temp = []

    elif np.max(n_emb_col) == 1:  # no need for tracking
        use_indices = [well_indices[w] for w in range(len(well_indices)) if n_emb_col[w] == 1]

        df_temp = master_df_update.loc[use_indices, temp_cols].copy()
        df_temp.reset_index(inplace=True)
        keep_cols = [n for n in df_temp.columns if n != "index"]
        df_temp = df_temp.loc[:, keep_cols]

        df_temp["xpos"] = master_df_update.loc[use_indices, "e0_x"].values
        df_temp["ypos"] = master_df_update.loc[use_indices, "e0_y"].values
        df_temp["fraction_alive"] = master_df_update.loc[use_indices, "e0_frac_alive"].values
        df_temp["region_label"] = master_df_update.loc[use_indices, "e0_label"].values
        df_temp.loc[:, "embryo_id"] = well_id + '_e00'

    else:  # this case is more complicated
        last_i = np.max(np.where(n_emb_col > 0)[0]) + 1
        first_i = np.min(np.where(n_emb_col > 0)[0])
        track_indices = well_indices[first_i:last_i]
        n_emb = int(master_df_update.loc[track_indices[0], "n_embryos_observed"])
        n_emb_orig = n_emb

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
            n_emb = int(n_emb_col[t + 1 + first_i])#.astype(int)
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
        df_list = []
        # use ID array to generate stable embryo IDs
        for n in range(n_emb_orig):
            use_indices = [well_indices[w] for w in range(len(well_indices)) if ~np.isnan(id_array[w, n])]

            use_subindices = [w for w in range(len(well_indices)) if ~np.isnan(id_array[w, n])]

            df_temp = master_df_update.loc[use_indices, temp_cols].copy()
            df_temp.reset_index(inplace=True, drop=True)
            # keep_cols = [n for n in df_temp.columns if n != "index"]
            # df_temp = df_temp.loc[:, keep_cols]
            for iter, ui in enumerate(use_indices):
                id = int(id_array[use_subindices[iter], n])

                df_temp.loc[iter, "xpos"] = master_df_update.loc[ui, "e" + str(id) + "_x"]
                df_temp.loc[iter, "ypos"] = master_df_update.loc[ui, "e" + str(id) + "_y"]
                df_temp.loc[iter, "fraction_alive"] = master_df_update.loc[ui, "e" + str(id) + "_frac_alive"]
                df_temp.loc[iter, "region_label"] = master_df_update.loc[ui, "e" + str(id) + "_label"]
            df_temp["embryo_id"] = well_id + f'_e{n:02}'

            df_list.append(df_temp)

        df_temp = pd.concat(df_list, axis=0, ignore_index=True)

            # if i_pass == 0:
            #     embryo_metadata_df = df_temp.copy()
            # else:
            #     embryo_metadata_df = pd.concat([embryo_metadata_df, df_temp.copy()], axis=0, ignore_index=True)
            # i_pass += 1

    return df_temp


def get_embryo_stats(index, root, embryo_metadata_df, qc_scale_um, ld_rat_thresh):

    row = embryo_metadata_df.loc[index].copy()

    # FF path
    # FF_path = os.path.join(root, 'built_image_data', 'stitched_FF_images', '')

    # generate path and image name
    segmentation_path = os.path.join(root, 'built_image_data', 'segmentation', '')
    seg_dir_list_raw = glob.glob(segmentation_path + "*")
    seg_dir_list = [s for s in seg_dir_list_raw if os.path.isdir(s)]

    emb_path = [m for m in seg_dir_list if "mask" in m][0]
    bubble_path = [m for m in seg_dir_list if "bubble" in m][0]
    focus_path = [m for m in seg_dir_list if "focus" in m][0]
    yolk_path = [m for m in seg_dir_list if "yolk" in m][0]

    well = row["well"]
    time_int = row["time_int"]
    date = str(row["experiment_date"])

    im_stub = well + f"_t{time_int:04}"
    im_name = glob.glob(os.path.join(emb_path, date, "*" + im_stub + "*"))
    ff_size = row["FOV_size_px"]

    # # load masked images
    # im_emb_path = os.path.join(emb_path, date, im_name)

    im = io.imread(im_name[0])
    im_mask = (np.round(im / 255 * 2) - 1).astype(np.uint8)

    # merge live/dead labels for now
    # im_mask = np.zeros(im.shape, dtype="uint8")
    # im_mask[np.where(im == 1)] = 1
    # im_mask[np.where(im == 2)] = 1
    im_mask_lb = label(im_mask)

    im_bubble_path = glob.glob(os.path.join(bubble_path, date, "*" + im_stub + "*"))[0]
    im_bubble = io.imread(im_bubble_path)
    im_bubble = np.round(im_bubble / 255 * 2 - 1).astype(int)
    im_bubble = remove_small_objects(label(im_bubble), 128)
    im_bubble[im_bubble > 0] = 1

    im_focus_path = glob.glob(os.path.join(focus_path, date, "*" + im_stub + "*"))[0] #os.path.join(focus_path, date, im_name)
    im_focus = io.imread(im_focus_path)
    im_focus = np.round(im_focus / 255 * 2 - 1).astype(int)
    im_focus = remove_small_objects(label(im_focus), 128)
    im_focus[im_focus > 0] = 1

    im_yolk_path = glob.glob(os.path.join(yolk_path, date, "*" + im_stub + "*"))[0] #os.path.join(yolk_path, date, im_name)
    im_yolk = io.imread(im_yolk_path)
    im_yolk = np.round(im_yolk / 255 * 2 - 1).astype(int)
    # im_yolk = remove_small_objects(label(im_yolk), 128)
    # im_yolk[im_yolk > 0] = 1

    # get surface area
    px_dim_raw = row["Height (um)"] / row["Height (px)"]   # to adjust for size reduction (need to automate this)
    size_factor = np.sqrt(ff_size / im_yolk.size) #row["Width (px)"] / 640 * 630/320
    px_dim = px_dim_raw * size_factor
    qc_scale_px = int(np.ceil(qc_scale_um / px_dim))

    lbi = row["region_label"]  # im_mask_lb[yi, xi]

    assert lbi != 0  # make sure we're not grabbing empty space

    im_mask_lb = (im_mask_lb == lbi).astype(int)

    # calculate sa-related metrics
    rg = regionprops(im_mask_lb)
    row.loc["surface_area_um"] = rg[0].area_filled * px_dim ** 2
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
    im_intersect = np.multiply((im_yolk == 1)*1, (im_mask_lb == 1)*1)
    row.loc["no_yolk_flag"] = ~np.any(im_intersect)

    # is a part of the embryo mask at or near the image boundary?
    im_trunc = im_mask_lb[qc_scale_px:-qc_scale_px, qc_scale_px:-qc_scale_px]
    row.loc["frame_flag"] = np.sum(im_mask_lb) != np.sum(im_trunc)

    # is there an out-of-focus region in the vicinity of the mask?
    if np.any(im_focus) or np.any(im_bubble):
        im_dist = scipy.ndimage.distance_transform_edt(im_mask_lb == 0)

    if np.any(im_focus):
        min_dist = np.min(im_dist[np.where(im_focus == 1)])
        row.loc["focus_flag"] = min_dist <= 2 * qc_scale_px
        
    # is there bubble in the vicinity of embryo?
    if np.any(im_bubble == 1):
        min_dist_bubble = np.min(im_dist[np.where(im_bubble == 1)])
        row.loc["bubble_flag"] = min_dist_bubble <= 2 * qc_scale_px

    row_out = pd.DataFrame(row).transpose()
    return row_out


####################
# Main process function 1
####################
def build_well_metadata_master(root, well_sheets=None):

    print("Compiling metadata...")

    if well_sheets == None:
        well_sheets = ["medium", "genotype", "chem_perturbation", "start_age_hpf", "embryos_per_well"]

    metadata_path = os.path.join(root, 'metadata', '')
    well_meta_path = os.path.join(metadata_path, 'well_metadata', '*.xlsx')
    built_meta_path = os.path.join(metadata_path, 'built_metadata_files', '')
    # ff_im_path = os.path.join(root, 'built_image_data')#, 'FF_images', '*')

    # load master experiment table
    exp_df = pd.read_csv(os.path.join(metadata_path, 'experiment_metadata.csv'))
    exp_df = exp_df[["experiment_id", "start_date", "temperature", "use_flag", "has_sci_data", "microscope"]]
    exp_df.loc[np.isnan(exp_df["has_sci_data"]), "has_sci_data"] = 0
    exp_df = exp_df.loc[exp_df["use_flag"] == 1, :]
    exp_df = exp_df.rename(columns={"start_date": "experiment_date"})
    exp_df["experiment_date"] = exp_df["experiment_date"].astype(int)
    exp_date_list = exp_df["experiment_date"].astype(str).to_list()
    # Load and concatenate well metadata into one long pandas table
    well_df_list = []

    project_list = sorted(glob.glob(os.path.join(built_meta_path, "*.csv")))

    for p, readname in enumerate(project_list):
        pname = path_leaf(readname)
        exp_date = pname[:8]
        if exp_date in exp_date_list:
            temp_table = pd.read_csv(readname)
            temp_table["experiment_date"] = exp_date
            # temp_table["experiment_id"] = p
            # temp_table["microscope"] = microscope
            # if exp_date == "20230830":
            temp_table = temp_table.drop_duplicates(subset=["well", "time_int"]) # these dupes only happen for Keyence experiments with no timelapse
            # add to list of metadata dfs
            well_df_list.append(temp_table)

    master_well_table = pd.concat(well_df_list, axis=0, ignore_index=True)
    if "microscope" in master_well_table.columns:
        master_well_table = master_well_table.drop(labels="microscope", axis=1)

    # recast experiment_date variables
    master_well_table["experiment_date"] = master_well_table["experiment_date"].astype(str)
    exp_df["experiment_date"] = exp_df["experiment_date"].astype(str)
    master_well_table = master_well_table.merge(exp_df, on="experiment_date", how='left')
    if master_well_table['use_flag'].isnull().values.any():
        raise Exception("Error: mismatching experiment IDs between experiment- and well-level metadata")

    # pull metadata from individual well sheets
    project_list_well = sorted(glob.glob(well_meta_path))
    well_name_list = make_well_names()
    well_df_list = []
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

            if "qc" in xl_temp.sheet_names:
                sheet_temp = xl_temp.parse("qc")
                well_df["well_qc_flag"] = sheet_temp.iloc[0:8, 1:13].values.ravel()
            else:
                well_df["well_qc_flag"] = 0

            well_df["experiment_date"] = date_string

            # add to list
            well_df_list.append(well_df)
            
    well_df_long = pd.concat(well_df_list, axis=0, ignore_index=True)
    well_df_long["experiment_date"] = well_df_long["experiment_date"].astype(str)
    # add to main dataset
    master_well_table = master_well_table.merge(well_df_long, on=["well", "experiment_date"], how='left')

    if master_well_table[well_sheets[0]].isnull().values.any():
        raise Exception("Error: missing well-specific metadata")

    # calculate approximate stage using linear formula from Kimmel et al 1995 (is there a better formula out there?)
    # dev_time = actual_time*(0.055 T - 0.57) where T is in Celsius...
    master_well_table["predicted_stage_hpf"] = master_well_table["start_age_hpf"] + \
                                               master_well_table["Time Rel (s)"]/3600*(0.055*master_well_table["temperature"]-0.57)  # linear formula to estimate stage 

    # generate new master index
    master_well_table["well_id"] = master_well_table["experiment_date"] + "_" + master_well_table["well"]
    cols = master_well_table.columns.values.tolist()
    cols_new = [cols[-1]] + cols[:-1]
    master_well_table = master_well_table[cols_new]

    # save to file
    drop_cols = [col for col in master_well_table.columns if "Unnamed" in col]
    master_well_table = master_well_table.drop(labels=drop_cols, axis=1)

    out_path = os.path.join(metadata_path, "combined_metadata_files", "")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    master_well_table.to_csv(os.path.join(out_path, 'master_well_metadata.csv'))

    # if np.any(master_well_table["master_perturbation"].astype(str) == "nan"):
    #     raise Exception("Error: missing master perturbation info")

    print("Done.")
    return {}

####################
# Main process function 2
####################

def segment_wells(root, min_sa_um=250000, max_sa_um=2000000, par_flag=False, overwrite_well_stats=False):

    print("Processing wells...")
    # generate paths to useful directories
    metadata_path = os.path.join(root, 'metadata', 'combined_metadata_files')
    segmentation_path = os.path.join(root, 'built_image_data', 'segmentation', '')

    # load well-level metadata
    master_df = pd.read_csv(os.path.join(metadata_path, 'master_well_metadata.csv'), index_col=0)
    experiments_to_use = np.unique(master_df["experiment_date"]).astype(str)

    # get list of segmentation directories
    seg_dir_list_raw = glob.glob(segmentation_path + "*")
    seg_dir_list = [s for s in seg_dir_list_raw if os.path.isdir(s)]

    ###################
    # Track number of embryos and position over time
    ###################

    emb_path = [m for m in seg_dir_list if "mask" in m][0]

    # get list of experiments
    experiment_list = sorted(glob.glob(os.path.join(emb_path, "*")))
    experiment_name_list = [path_leaf(e) for e in experiment_list]
    experiment_list = [experiment_list[e] for e in range(len(experiment_list)) if "ignore" not in experiment_list[e] and experiment_name_list[e][:8] in experiments_to_use ]

    ckpt1_path = os.path.join(metadata_path, "embryo_metadata_df_ckpt1.csv")

    images_to_process, df_to_process, prev_meta_df, image_size_vec, mask_size_vec\
         = get_images_to_process(ckpt1_path, experiment_list, master_df, overwrite_well_stats)

    assert len(images_to_process) == df_to_process.shape[0]

    # print("Remember to un-comment this stuff")
    if len(images_to_process) > 0:
        # initialize empty columns to store embryo information
        master_df_update = df_to_process.copy()
        # remove nuisance columns 
        drop_cols = [col for col in master_df_update.columns if "Unnamed" in col]
        master_df_update = master_df_update.drop(labels=drop_cols, axis=1)

        master_df_update["n_embryos_observed"] = np.nan
        master_df_update["FOV_size_px"] = image_size_vec
        for n in range(4): # allow for a maximum of 4 embryos per well
            master_df_update["e" + str(n) + "_x"] = np.nan
            master_df_update["e" + str(n) + "_y"] = np.nan
            master_df_update["e" + str(n) + "_label"] = np.nan
            master_df_update["e" + str(n) + "_frac_alive"] = np.nan

        ##########################
        # extract position and live/dead status of each embryo in each well
        ##########################
        
        emb_df_list = []

        n_workers = np.ceil(os.cpu_count()/4).astype(int)
        if par_flag:
            emb_df_temp = process_map(partial(count_embryo_regions, image_list=images_to_process, master_df_update=master_df_update,
                                              max_sa_um=max_sa_um, min_sa_um=min_sa_um),
                                        range(len(images_to_process)), max_workers=n_workers, chunksize=10)
            emb_df_list += emb_df_temp
        else:
            for index in tqdm(range(len(images_to_process)), "Calculating embryo stats..."):
                df_temp = count_embryo_regions(index, images_to_process, master_df_update, max_sa_um, min_sa_um)
                emb_df_list.append(df_temp)

        # update the df
        print("Updating metadata entries...")
        df_vec = [e[1] for e in emb_df_list]
        df_temp = pd.concat(df_vec, axis=0, ignore_index=True)
        index_vec = np.asarray([e[0] for e in emb_df_list])

        master_df_update.iloc[index_vec, :] = df_temp.loc[:, :]
        # for e in tqdm(range(len(emb_df_list))):
        #     master_index = emb_df_list[e][0]
        #     row = emb_df_list[e][1]
        #     master_df_update.loc[master_index, :] = row.iloc[0, :]

        if isinstance(prev_meta_df, pd.DataFrame):
            # prev_table = pd.read_csv(ckpt1_path, index_col=0)
            master_df_update = pd.concat([prev_meta_df, master_df_update], ignore_index=True)
        master_df_update.to_csv(ckpt1_path)

    else:
        master_df_update = pd.read_csv(ckpt1_path, index_col=0)

    drop_cols = [col for col in master_df_update.columns if "Unnamed" in col]
    master_df_update = master_df_update.drop(labels=drop_cols, axis=1)
    # Next, iterate through the extracted positions and use rudimentary tracking to assign embryo instances to stable
    # embryo_id that persists over time

    # get list of unique well instances
    if np.any(np.isnan(master_df_update["n_embryos_observed"].values.astype(float))):
        raise Exception("Missing rows found in metadata df")

    well_id_list = np.unique(master_df_update["well_id"])
    track_df_list = []
    # print("Performing embryo tracking...")
    for w in tqdm(range(len(well_id_list)), "Doing embryo tracking..."):
        well_id = well_id_list[w]
        df_temp = do_embryo_tracking(well_id, master_df, master_df_update)
        track_df_list.append(df_temp)

    track_df_list = [df for df in track_df_list if isinstance(df, pd.DataFrame)]
    embryo_metadata_df = pd.concat(track_df_list, ignore_index=True)
    # embryo_metadata_df = embryo_metadata_df.iloc[:, 1:]
    track_path = (os.path.join(metadata_path, "embryo_metadata_df_tracked.csv"))
    embryo_metadata_df.to_csv(track_path)

    return {}


def compile_embryo_stats(root, overwrite_flag=False, ld_rat_thresh=0.9, qc_scale_um=150):

    metadata_path = os.path.join(root, 'metadata', "combined_metadata_files", '')
    # segmentation_path = os.path.join(root, 'built_image_data', 'segmentation', '')

    track_path = (os.path.join(metadata_path, "embryo_metadata_df_tracked.csv"))
    embryo_metadata_df = pd.read_csv(track_path, index_col=0)

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

    # check for existing embryo metadata
    if os.path.isfile(os.path.join(metadata_path, "embryo_metadata_df.csv")) and not overwrite_flag:
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



    # print("Extracting embryo stats")
    # if False: #par_flag:
    #     emb_df_list = pmap(get_embryo_stats, indices_to_process,
    #                             (embryo_metadata_df, qc_scale_um, ld_rat_thresh), rP=0.75)
    # else:
    emb_df_list = []
    for index in tqdm(indices_to_process, "Extracting embryo stats..."):
        df_temp = get_embryo_stats(index, root, embryo_metadata_df, qc_scale_um, ld_rat_thresh)
        emb_df_list.append(df_temp)

    # updating data frame
    for i, ind in enumerate(indices_to_process):
        row_df = emb_df_list[i]
        for nc in new_cols:
            embryo_metadata_df.loc[ind, nc] = row_df[nc].values


    # make master flag
    embryo_metadata_df["use_embryo_flag"] = ~(
            embryo_metadata_df["bubble_flag"].values.astype(bool) | embryo_metadata_df["focus_flag"].values.astype(bool) |
            embryo_metadata_df["frame_flag"].values.astype(bool) | embryo_metadata_df["dead_flag"].values.astype(bool) |
            embryo_metadata_df["no_yolk_flag"].values.astype(bool) | (embryo_metadata_df["well_qc_flag"].values==1).astype(bool))

    embryo_metadata_df.to_csv(os.path.join(metadata_path, "embryo_metadata_df.csv"))

    print("phew")

def extract_embryo_snips(root, outscale=5.66, overwrite_flag=False, par_flag=False, outshape=None, dl_rad_um=75):

    if outshape == None:
        outshape = [576, 256]

    # read in metadata
    metadata_path = os.path.join(root, 'metadata', "combined_metadata_files", '')
    embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df.csv"), index_col=0)

    # make directory for embryo snips
    im_snip_dir = os.path.join(root, 'training_data', 'bf_embryo_snips', '')
    if not os.path.isdir(im_snip_dir):
        os.makedirs(im_snip_dir)
    mask_snip_dir = os.path.join(root, 'training_data', 'bf_embryo_masks', '')
    if not os.path.isdir(mask_snip_dir):
        os.makedirs(mask_snip_dir)

    # make stable embryo ID
    embryo_metadata_df["snip_id"] = ''
    for r in range(embryo_metadata_df.shape[0]):
        embryo_id = embryo_metadata_df["embryo_id"].iloc[r]
        time_id = embryo_metadata_df["time_int"].iloc[r]
        embryo_metadata_df.loc[r, "snip_id"] = embryo_id + f'_t{time_id:04}'

    #embryo_metadata_df["embryo_id"] + "_" + embryo_metadata_df["time_int"].astype(str)

    export_indices = range(embryo_metadata_df.shape[0])

    # draw random sample to estimate background
    # print("Estimating background...")
    px_mean, px_std = estimate_image_background(root, embryo_metadata_df, bkg_seed=309, n_bkg_samples=100)

    embryo_metadata_df["out_of_frame_flag"] = False
    embryo_metadata_df["snip_um_per_pixel"] = outscale

    # extract snips
    out_of_frame_flags = []

    # if not par_flag:
    update_indices = []
    if par_flag:
        n_workers = np.ceil(os.cpu_count() / 2).astype(int)
        out_of_frame_flags = process_map(partial(export_embryo_snips, root=root, embryo_metadata_df=embryo_metadata_df,
                                      dl_rad_um=dl_rad_um, outscale=outscale, outshape=outshape,
                                      px_mean=0.1*px_mean, px_std=0.1*px_std, overwrite_flag=overwrite_flag),
                    range(len(export_indices)), max_workers=n_workers, chunksize=10)

        update_indices = np.where(np.asarray(out_of_frame_flags) > -1)
        out_of_frame_flags = np.asarray(out_of_frame_flags)[update_indices]
    else:
        for r in tqdm(export_indices, "Exporting snips..."):
            oof = export_embryo_snips(r, root=root, embryo_metadata_df=embryo_metadata_df,
                                      dl_rad_um=dl_rad_um, outscale=outscale, outshape=outshape,
                                      px_mean=0.1*px_mean, px_std=0.1*px_std, overwrite_flag=overwrite_flag)

            if oof > -1:
                out_of_frame_flags.append(oof)
                update_indices.append(r)

        update_indices = np.asarray(update_indices)

    # add oof flag
    if update_indices[0].any():
        embryo_metadata_df["out_of_frame_flag"].iloc[update_indices] = out_of_frame_flags
        embryo_metadata_df["use_embryo_flag"] = embryo_metadata_df["use_embryo_flag"] & ~embryo_metadata_df["out_of_frame_flag"]

    # save
    embryo_metadata_df.to_csv(os.path.join(metadata_path, "embryo_metadata_df.csv"))


if __name__ == "__main__":

    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
    
    # print('Compiling well metadata...')
    build_well_metadata_master(root)
    # #
    # # print('Compiling embryo metadata...')
    segment_wells(root, par_flag=False, overwrite_well_stats=True)

    compile_embryo_stats(root, overwrite_flag=True)

    # print('Extracting embryo snips...')
    extract_embryo_snips(root, par_flag=False, outscale=6.5, dl_rad_um=50, overwrite_flag=False)