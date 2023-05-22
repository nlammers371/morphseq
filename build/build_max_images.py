import glob
import re
from aicsimageio import AICSImage
import numpy as np
import os
from tifffile import imsave
from skimage.morphology import convex_hull_image
import pickle
from matplotlib import pyplot as plt
import skimage
import scipy
from skimage.morphology import disk
from skimage.measure import label, regionprops
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage import filters
from PIL import Image
import pandas as pd
import pathlib

def parse_exp_metadata(sheet_df, n_wells, n_time_points):
    sheet_shape = sheet_df.shape
    read_df = pd.DataFrame(index=range(n_wells), columns=["start_i", "stop_i"])
    total_frames = 0
    if len(sheet_shape) == 1:
        iter_i = 0
        for i in range(sheet_shape[0]):
            for j in range(sheet_shape[1] - 1):
                read_df["start_i"].iloc[iter_i] = 0
                read_df["stop_i"].iloc[iter_i] = n_time_points - 1
                iter_i += 1
                total_frames += read_df["stop_i"].iloc[iter_i] - read_df["start_i"].iloc[iter_i]

    elif sheet_shape[0]*(sheet_shape[1]-1) == n_wells:
        iter_i = 0
        for col in range(sheet_shape[1]-1):
            for row in range(sheet_shape[0]):
                well_string = sheet_df.iloc[row, col+1]
                # print(str(row) + ',' + str(col+1))
                # check for core codes
                if "EMPTY" in well_string: # exclude all empty frames
                    read_df["start_i"].iloc[iter_i] = -1
                    read_df["stop_i"].iloc[iter_i] = -1
                elif "DEAD" in well_string: # take only up to time of death + 1
                    read_df["start_i"].iloc[iter_i] = 0
                    # find time of death
                    ind1 = well_string.find("(T") + 2
                    ind2 = well_string.find(")")
                    tod = int(well_string[ind1:ind2])
                    read_df["stop_i"].iloc[iter_i] = np.min([tod+1, n_time_points-1])
                elif "OUT OF FRAME" in well_string: # take only up to time of death + 1
                    read_df["start_i"].iloc[iter_i] = 0
                    # find time of death
                    ind1 = well_string.find("(T") + 2
                    ind2 = well_string.find(")")
                    toof = int(well_string[ind1:ind2])
                    read_df["stop_i"].iloc[iter_i] = np.min([toof+1, n_time_points-1])
                elif ("GOOD" in well_string) | ("FAIR" in well_string) | ("POOR" in well_string):
                    read_df["start_i"].iloc[iter_i] = 0
                    read_df["stop_i"].iloc[iter_i] = n_time_points-1
                else:
                    read_df["start_i"].iloc[iter_i] = -1
                    read_df["stop_i"].iloc[iter_i] = -1
                total_frames += read_df["stop_i"].iloc[iter_i] - read_df["start_i"].iloc[iter_i]
                iter_i += 1

    else:
        iter_i = 0
        for i in range(sheet_shape[0]):
            for j in range(sheet_shape[1] - 1):
                read_df["start_i"].iloc[iter_i] = 0
                read_df["stop_i"].iloc[iter_i] = n_time_points - 1
                iter_i += 1
                total_frames += read_df["stop_i"].iloc[iter_i] - read_df["start_i"].iloc[iter_i]

    return read_df #, total_frames

overwrite_flag = True

# set key parameters for image generation
size_x = 256 #512  # output x size (pixels)
size_y = 128 #256  # output y size (pixels)
target_res = 7*512/size_x  # target isotropic pixel res (um)

# set path to excel doc with metadata
db_path = "E:/Nick/Dropbox (Cole Trapnell's Lab)/"
excel_path = db_path + "Nick/morphSeq/data/embryo_metadata.xlsx"

# set path to save images
out_dir = db_path + "Nick/morphSeq/data/vae_20230522/"
depth_dir = out_dir + "depth_images" + f'_res{np.round(target_res).astype(int):03}/'
max_dir = out_dir + "max_images" + f'_res{np.round(target_res).astype(int):03}/'

if not os.path.isdir(depth_dir):
    os.makedirs(depth_dir)

if not os.path.isdir(max_dir):
    os.makedirs(max_dir)

# set input directories
image_dir_list = [db_path + "Nick/morphSeq/data/20230517/"]

metadata_dict = {}

# compile metadata dictionary and calculate how many images, total, we have
print("Compiling metadata...")
total_frames = 0
for d, im_dir in enumerate(image_dir_list):
    # d = 0
    # im_dir = image_dir_list[d]
    im_list = sorted(glob.glob(im_dir + "*.nd2"))

    for i, im in enumerate(im_list):

        # set path to image
        imObject = AICSImage(im)

        n_time_points = imObject.dims["T"][0]
        # get list of scenes (well positions)
        well_list = imObject.scenes
        n_wells = len(well_list)

        # store key metadata
        im_dict = {"directory": im_dir, "filename": im, "metadata": imObject.metadata, "num_wells": n_wells, "num_frames": n_time_points}

        metadata_dict[im] = im_dict

        total_frames += n_wells*n_time_points

# create a binary pickle file
f = open(out_dir + "metadata_dict.pkl", "wb")
# write the python object (dict) to pickle file
pickle.dump(metadata_dict, f)
# close file
f.close()

# read in excel file
xl = pd.ExcelFile(excel_path)
excel_sheet_list = xl.sheet_names  # see all sheet names


iter_i = 0
print("Generating depth images...")
# set working index
for d, im_dir in enumerate(image_dir_list):   # enumerate([image_dir_list[0]]):
    # d = 0
    # im_dir = image_dir_list[d]
    im_list = sorted(glob.glob(im_dir + "*.nd2"))

    for i, im in enumerate(im_list):   # enumerate([im_list[0]]):
        im = pathlib.PureWindowsPath(im)
        im = im.as_posix()
        # set path to image
        imObject = AICSImage(im)

        # get list of scenes (well positions)
        well_list = imObject.scenes
        n_wells = len(well_list)

        # shift to desired position
        imObject.set_scene(well_list[0])
        # check how many time points we have
        n_time_points = imObject.dims["T"][0]

        # check to see if we have an excel sheet that corresponds to this experiment
        slash_inds = [m.start() for m in re.finditer("[\\|/]", im_dir)]
        im_name = im.replace(im_dir, "")
        sheet_name = im_dir[slash_inds[-2]+1:slash_inds[-1]] + "_" + im_name[:-4]

        # if there is a sheet, pull it and use it
        try:
            sheet_ind = excel_sheet_list.index(sheet_name)
            sheet_df = pd.read_excel(excel_path, sheet_name=sheet_ind)
        except:
            sheet_df = np.asarray([])

        read_df = parse_exp_metadata(sheet_df, n_wells, n_time_points)

        # iterate through each well position
        for w, well in enumerate(well_list): # enumerate([well_list[0]]):

            # shift to desired position
            imObject.set_scene(well)

            # get pixel resolution
            res_array = np.asarray(imObject.physical_pixel_sizes)

            # generate vectors to use for resampling
            rs_vector = res_array / target_res
            rs_inv = np.floor(rs_vector ** (-1)).astype(int)
            rs_inv[0] = 1
            rs_vector2 = np.multiply(rs_inv, rs_vector)

            # check how many time points we have
            n_time_points = imObject.dims["T"][0]

            # extract start and stop times
            start_i = read_df["start_i"].iloc[w]
            stop_i = read_df["stop_i"].iloc[w]

            # iterate through each time point
            for t in range(start_i, stop_i):

                # check to see if files already exist
                outName = depth_dir + f'im_depth_Source{i:03}_' + f'Embryo{w:03}_' + f'T{t:03}.tiff'
                outNameMax = max_dir + f'im_max_source{i:03}_' + f'embryo{w:03}_' + f't{t:03}.tif'

                if (not os.path.isfile(outName)) or (not os.path.isfile(outNameMax)) or overwrite_flag:

                    # extract image
                    imData = np.squeeze(imObject.get_image_data("CZYX", T=t))

                    #######################
                    # generate z projection

                    # Step 1: block reduction. Essentially max pooling
                    imData_block = skimage.measure.block_reduce(imData, (rs_inv[0], rs_inv[1], rs_inv[2]), np.max)

                    # Step 2: rescale so that voxels are isotropic
                    imData_rs = scipy.ndimage.zoom(imData_block, rs_vector2)

                    # Step 3: find brightest Z pixels at each xy coordinate
                    max_z_b = np.argmax(imData_rs, 0)
                    max_b_z = np.max(imData_rs, 0)  # and brightest pixel values

                    # Step 4: mask
                    threshold_sa = filters.threshold_sauvola(max_b_z, window_size=9)
                    fish_mask = (max_b_z < threshold_sa) * 1

                    # empirically, I find that different filter sizes are needed for 4x and 10x
                    # if res_array[2] > 2:
                    #     fp = disk(8)
                    #     fp_small = disk(3)
                    # else:
                    #     fp = disk(10)
                    #     fp_small = disk(5)
                    fish_strip = skimage.morphology.remove_small_objects(label(fish_mask),
                                                                         min_size=1000)  # remove small objects
                    fish_clean = convex_hull_image(fish_strip)
                    # fish_closed = closing(fish_mask, fp)  # morphological closing
                    # fish_strip = skimage.morphology.remove_small_objects(label(fish_closed), min_size=600)  # remove small objects
                    # fish_clean = skimage.morphology.binary_erosion(fish_strip, fp_small)  # clean up edges
                    # fish_clean = scipy.ndimage.binary_fill_holes(fish_clean)
                    # fish_clean = skimage.morphology.remove_small_objects(label(fish_clean), min_size=600)

                    # step 5: Normalize
                    mean_z = np.mean(max_z_b)  # [np.where(fish_clean > 0)])
                    std_z = np.std(max_z_b)  # [np.where(fish_clean > 0)])

                    im_norm = -(max_z_b - mean_z) / std_z

                    # get background stats
                    mean_z_depth_b = np.mean(im_norm[np.where(fish_clean == 0)])
                    std_z_depth_b = np.std(im_norm[np.where(fish_clean == 0)])

                    # step 6: use mask to replace background pixels with white noise
                    # im_norm[np.where(fish_clean == 0)] = np.random.normal(loc=0, scale=.1, size=(np.sum(fish_clean == 0),))

                    # Step 7: resize and center image
                    regions = regionprops(label(fish_clean))
                    try:
                        im_center = regions[0].centroid # by construction there is only 1 binary region at this point
                    except:
                        im_center = [size_y / 2 + 0.5, size_x / 2 + 0.5]

                    im_centroid = np.round(im_center).astype(int)

                    im_array_depth = np.random.normal(loc=mean_z_depth_b, scale=std_z_depth_b,
                                                      size=(size_y, size_x))  # initialize array

                    c_diff_y = np.floor(size_y / 2) - im_centroid[0]
                    c_diff_x = np.floor(size_x / 2) - im_centroid[1]

                    xmin = max(im_centroid[1] - (size_x / 2), 0)
                    xmax = min(im_centroid[1] + size_x / 2, fish_clean.shape[1])
                    from_x = np.arange(xmin, xmax).astype(int)

                    ymin = max(im_centroid[0] - (size_y / 2), 0)
                    ymax = min(im_centroid[0] + size_y / 2, fish_clean.shape[0])
                    from_y = np.arange(ymin, ymax).astype(int)

                    to_y = np.round(from_y + c_diff_y).astype(int)
                    to_x = np.round(from_x + c_diff_x).astype(int)

                    im_array_depth[to_y[0]:to_y[-1], to_x[0]:to_x[-1]] = \
                        im_norm[from_y[0]:from_y[-1], from_x[0]:from_x[-1]]

                    im_array_depth[np.where(np.abs(im_array_depth) > 1)] = \
                        1 * np.sign(im_array_depth[np.where(np.abs(im_array_depth) > 1)])

                    # convert to 8-bit format
                    im_array_depth_256 = im_array_depth - np.min(im_array_depth)
                    im_array_depth_256 = np.uint8(im_array_depth_256 / np.max(im_array_depth_256) * 255)
                    # plt.imshow(im_array_depth)
                    # im_depth_out = im_array_depth * 0.5 + 1
                    # im_depth_out = im_depth_out.astype(int)
                    im_depth_out = Image.fromarray(im_array_depth_256, "L")
                    # Save depth-encoded image
                    im_depth_out.save(outName)

                    ##############
                    # Generate normalized max projection images
                    mean_z_max = np.mean(max_b_z)
                    std_z_max = np.std(max_b_z)

                    im_norm_max = (max_b_z - mean_z_max) / std_z_max

                    # get background stats
                    mean_z_max_b = np.mean(im_norm_max[np.where(fish_clean == 0)])
                    std_z_max_b = np.std(im_norm_max[np.where(fish_clean == 0)])

                    im_array_max = np.random.normal(loc=mean_z_max_b, scale=std_z_max_b, size=(
                    size_y, size_x))  # np.zeros((size_y, size_x))  # initialize array

                    im_array_max[to_y[0]:to_y[-1], to_x[0]:to_x[-1]] = \
                        im_norm_max[from_y[0]:from_y[-1], from_x[0]:from_x[-1]]

                    prc99 = np.percentile(np.abs(im_array_max), 99.5)
                    im_array_max[np.where(np.abs(im_array_max) > prc99)] = \
                        prc99 * np.sign(im_array_max[np.where(np.abs(im_array_max) > prc99)])

                    # convert to 8-bit format
                    im_array_max_256 = im_array_max - np.min(im_array_max)
                    im_array_max_256 = np.uint8(im_array_max_256 / np.max(im_array_max_256) * 255)
                    # plt.imshow(im_array_depth)
                    # im_depth_out = im_array_depth * 0.5 + 1
                    # im_depth_out = im_depth_out.astype(int)
                    im_max_out = Image.fromarray(im_array_max_256, "L")

                    # Save depth-encoded image
                    outNameMax = max_dir + f'im_max_dir{d:03}' + f'source{i:03}_' + f'embryo{w:03}_' + f't{t:03}.tif'
                    im_max_out.save(outNameMax)
                else:
                    print("File exists. Skipping...")

                iter_i += 1

                print(f'{iter_i} of' + f' {total_frames} (' + f'{round(100*iter_i/total_frames,1)}%)')


# if __name__ == '__main__':
#     napari.run()