from aicsimageio import AICSImage
import os
import glob2 as glob
from skimage.morphology import (erosion, dilation, opening, closing, white_tophat)
from skimage import data
from skimage import filters
from skimage.measure import label, regionprops
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
import skimage
import scipy
import numpy as np
from alphashape import alphashape
from scipy.interpolate import LinearNDInterpolator

# set path to segmentation
seg_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/20230412/seg_24hpf_classifier/"

# set save path
save_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/20230412/3D_mesh_objects2/"

# make save directory
if not os.path.isdir(save_path):
    os.makedirs(save_path)


# get list of segmentation files
seg_list = sorted(glob.glob(seg_path + "*.tif"))

# set time points
n_time_points = len(seg_list)

# first, loaf final image to obtain a normalization constant
t_list = list(reversed(range(n_time_points)))
t_list[35] = 36
# t_list.remove(35)
for time_ind in t_list:

    # load im object
    labelObject = AICSImage(seg_list[time_ind])

    # extract raw image
    labelData = np.squeeze(labelObject.data)

    # hard coding these for now...
    pixel_size_z = 20  # res_array[1]
    pixel_size_x = 1  # res_array[2]
    pixel_size_y = 1  # res_array[3]

    # resize image such that voxels are isotropic
    z_rs_factor = pixel_size_z / pixel_size_x
    ds_factor = 4
    pixel_size_new = pixel_size_x / ds_factor
    labelData_rs = scipy.ndimage.zoom(labelData, [z_rs_factor / ds_factor, 1 / ds_factor, 1 / ds_factor])

    labelDataBin = labelData_rs > 0

    z_grid3, y_grid3, x_grid3 = np.meshgrid(range(0, labelDataBin.shape[0]),
                                            range(0, labelDataBin.shape[1]),
                                            range(0, labelDataBin.shape[2]),
                                            indexing="ij")
    z_grid3 = z_grid3.astype(float)
    ############
    # use max projection to condense image data
    max_z_project = np.max(labelDataBin, axis=0).astype(float)
    max_z_plane = np.divide(np.sum(np.multiply(labelDataBin, z_grid3), axis=0), np.sum(labelDataBin, axis=0))
    max_z_plane = max_z_plane.astype(float)
    max_z_plane[np.where(max_z_project == 0)] = np.nan

    ############
    # generate 2D mask to help remove outliers and identify interior pixels that need to be inerpolated
    fish_mask = (np.max(labelDataBin, axis=0) == 1) * 1

    # use morphological closure operation to fill in shape
    footprint = disk(13)
    fp_small = disk(2)
    fish_closed = closing(fish_mask > 0, footprint)
    fish_clean = skimage.morphology.remove_small_objects(label(fish_closed), min_size=64)
    fish_eroded = skimage.morphology.binary_erosion(fish_clean, fp_small)

    #############
    # Interpolate interior pixels

    y_grid2, x_grid2 = np.meshgrid(
        range(0, max_z_plane.shape[0]),
        range(0, max_z_plane.shape[1]),
        indexing="ij")

    x_flat = x_grid2.flatten()
    y_flat = y_grid2.flatten()

    # generate diplacement vectors
    max_z_plane_vals = max_z_plane.flatten()
    mask_flat = (fish_clean.flatten() > 0) * 1
    # mask_er = (fish_eroded.flatten() > 0) * 1
    #
    # ref_x = x_flat[np.where((~np.isnan(max_z_plane_vals)) & (mask_flat == 1))]
    # ref_y = y_flat[np.where((~np.isnan(max_z_plane_vals)) & (mask_flat == 1))]
    # ref_z = max_z_plane_vals[np.where((~np.isnan(max_z_plane_vals)) & (mask_flat == 1))]
    #
    # query_x = x_flat[np.where(mask_flat == 1)]
    # query_y = y_flat[np.where(mask_flat == 1)]
    # query_x_er = x_flat[np.where(mask_er == 1)]
    # query_y_er = y_flat[np.where(mask_er == 1)]

    # interpolate
    # interp_z = LinearNDInterpolator(list(zip(ref_x, ref_y)), ref_z)
    # query_z = interp_z(query_x, query_y)

    ##################
    # Calculate actual 3D mesh

    z_flat = max_z_plane.flatten()
    plot_x = x_flat[np.where((~np.isnan(max_z_plane_vals) & (mask_flat > 0)))]
    plot_y = y_flat[np.where((~np.isnan(max_z_plane_vals) & (mask_flat > 0)))]
    plot_z = z_flat[np.where((~np.isnan(max_z_plane_vals) & (mask_flat > 0)))]

    # surf_flat = surf_mask_array.flatten()
    # keep_indices = np.where(surf_mask_array > 0)

    # generate smooth bottom
    base_array = np.empty((len(plot_x), 3))
    base_array[:, 0] = plot_x
    base_array[:, 1] = plot_y
    base_array[:, 2] = plot_z + 20

    xyz_array3 = np.concatenate((plot_x[:, np.newaxis], plot_y[:, np.newaxis], plot_z[:, np.newaxis]), axis=1)
    xyz_array3 = np.concatenate((xyz_array3, base_array), axis=0)
    xyz_array3_norm = xyz_array3.copy()
    if time_ind == (n_time_points-1):
        norm_factor = np.max(xyz_array3_norm)
    xyz_array3_norm = xyz_array3_norm / norm_factor

    alpha_fish = alphashape(xyz_array3_norm, alpha=65)
    # alpha_fish.show()
    outPath = save_path + '3D_mesh_' + f'T{time_ind:03}.stl'
    alpha_fish.export(outPath)