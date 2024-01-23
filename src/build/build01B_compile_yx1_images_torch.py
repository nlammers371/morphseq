# script to define functions_folder for loading and standardizing fish movies
import os
import numpy as np
from skimage import io
import glob2 as glob
import torchvision
import torch
import torch.nn.functional as F
from src.functions.utilities import path_leaf
from tqdm import tqdm
import pandas as pd
import time
import nd2
import cv2
from sklearn.cluster import KMeans

def set_inputs_to_device(input_tensor, device):

    inputs_on_device = input_tensor

    if device == "cuda":
        cuda_inputs = input_tensor

        # for key in inputs.keys():
        #     if torch.is_tensor(inputs[key]):
        #         cuda_inputs[key] = inputs[key].cuda()

        #     else:
        #         cuda_inputs[key] = inputs[key]
        cuda_inputs = input_tensor.cuda()
        inputs_on_device = cuda_inputs

    return inputs_on_device


def findnth(haystack, needle, n):
    parts = haystack.split(needle, n+1)
    if len(parts)<=n+1:
        return -1
    return len(haystack)-len(parts[-1])-len(needle)

def trim_image(im, out_shape):
    im_shape = im.shape
    im_diffs = im_shape - out_shape

    pad_width = -im_diffs
    pad_width[np.where(pad_width < 0)] = 0
    im_out = np.pad(im.copy(), ((0, pad_width[0]), (0, pad_width[1])), mode='constant').astype('uint8')

    im_diffs[np.where(im_diffs < 0)] = 0
    sv = np.floor(im_diffs / 2).astype(int)
    if np.all(sv>0):
        im_out = im_out[sv[0]:-(im_diffs[0] - sv[0]), sv[1]:-(im_diffs[1] - sv[1])]
    elif sv[0]==0:
        im_out = im_out[:, sv[1]:-(im_diffs[1] - sv[1])]
    elif sv[1]==0:
        im_out = im_out[sv[0]:-(im_diffs[0] - sv[0]), :]

    return im_out



def process_frame(w, im_data_dask, well_name_list, well_time_list, well_ind_list, ff_dir, device, rs_dims_yx=None, rs_res_yx=None, overwrite_flag=False, n_z_keep=10, ch_to_use=0):

    # set scene
    well_name_conv = well_name_list[w]
    time_int = well_time_list[w]
    well_int = well_ind_list[w]

    # get data
    # start = time.time()
    n_z_slices = im_data_dask.shape[2]
    buffer = np.max([int((n_z_slices - n_z_keep)/2), 0])
    data_zyx = im_data_dask[time_int, well_int, buffer:-buffer, :, :].compute()
    # print(time.time() - start)

    # generate save names
    ff_out_name = 'ff_' + well_name_conv + f'_t{time_int:04}_' + f'ch{ch_to_use:02}_stitch'

    # calculate filter size
    filter_size = 3 #(np.round(5.66/rs_res_yx[0]*3) // 2 * 2 + 1).astype(int)
    if os.path.isfile(os.path.join(ff_dir, ff_out_name + ".png")) and not overwrite_flag:
        print(f"Skipping time point {time_int} for well {well_name_conv}.")

    else:
        data_tensor_raw = torch.tensor(data_zyx.astype(np.float64))
        data_tensor_raw = set_inputs_to_device(data_tensor_raw, device)

        # resize image
        if rs_dims_yx is not None:
            # print("Resizing...")
            data_zyx_rs = torchvision.transforms.functional.resize(data_tensor_raw, tuple([int(rs_dims_yx[0]), int(rs_dims_yx[1])]), antialias=True)
        else:
            data_zyx_rs = data_tensor_raw

        px99 = torch.tensor(np.percentile(data_zyx, 99))
        data_zyx_rs = data_zyx_rs / px99
        data_zyx_rs[data_zyx_rs>1] = 1
        data_zyx_rs = data_zyx_rs * 65535
        # data_zyx_rs = data_zyx_rs.astype(data_zyx.dtype)

        # get laplacian and gaussian filters
        ind = filter_size // 2 + 1
        lpf = np.zeros((2*filter_size+1, 2*filter_size+1))
        lpf[filter_size,filter_size] = 1
        lpf = cv2.Laplacian(lpf, cv2.CV_64F, ksize=filter_size)
        lpf = lpf[ind:-ind, ind:-ind]
        lpf_tensor = set_inputs_to_device(torch.reshape(torch.tensor(lpf), (1, 1, filter_size, filter_size)), device)

        # get Gaussian filter
        gf = np.zeros((2*filter_size+1, 2*filter_size+1))
        gf[filter_size, filter_size] = 1
        gf = cv2.GaussianBlur(gf, (filter_size, filter_size), 0)
        gf = gf[ind:-ind, ind:-ind]
        gf_tensor = set_inputs_to_device(torch.reshape(torch.tensor(gf), (1, 1, filter_size, filter_size)), device)

        # convert image to tensor 
        data_tensor = torch.reshape(data_zyx_rs, (data_zyx_rs.shape[0], 1, data_zyx_rs.shape[1], data_zyx_rs.shape[2]))

        # get Gaussian Blur
        GB = F.conv2d(input=data_tensor, weight=gf_tensor, padding="same")
        LoG = F.conv2d(input=GB, weight=lpf_tensor, padding="same")

        # calculate FF image
        # laps = np.zeros(data_zyx_rs.shape, dtype=np.float64)
        # for i in range(data_zyx_rs.shape[0]):
        #     laps[i, :, :] = doLap(data_zyx_rs[i, :, :], lap_size=filter_size, blur_size=filter_size)
            # laps_d.append(doLap(data_zyx_rs[i, :, :], lap_size=7, blur_size=7))  # I've found that depth stacking works better with larger filters

        # laps = np.asarray(laps)
        abs_laps = torch.abs(torch.squeeze(LoG))

        # laps_d = np.asarray(laps_d)
        # abs_laps_d = np.absolute(laps_d)

        # calculate full-focus and depth images
        # ff_image = np.zeros(shape=data_zyx_rs[0].shape, dtype=data_zyx[0].dtype)
        # depth_image = np.argmax(abs_laps, axis=0)
        maxima = torch.max(abs_laps, axis=0)
        bool_mask = abs_laps == maxima.values
        # mask = bool_mask.astype(np.uint8)
        # for i in range(len(data_zyx_rs)):
        #     ff_image[np.where(mask[i] == 1)] = data_zyx_rs[i][np.where(mask[i] == 1)]
        ff_image = np.asarray(torch.max(torch.multiply(bool_mask, data_zyx_rs), axis=0).values.cpu()).astype(np.uint16)
        # take the negative
        ff_image = 65535 - ff_image 

        # save images
        ff_out_name = 'ff_' + well_name_conv + f'_t{time_int:04}_' + f'ch{ch_to_use:02}_stitch'
        # depth_out_name = 'depth_' + well_name_conv + f'_t{time_int:04}_' + f'ch{ch_to_use:02}_stitch'

        # convet depth image to 8 bit
        # max_z = abs_laps.shape[0]
        # depth_image_int8 = np.round(depth_image / max_z * 255).astype('uint8')

        io.imsave(os.path.join(ff_dir, ff_out_name + ".png"), ff_image)
        # io.imsave(os.path.join(depth_dir, depth_out_name + ".png"), depth_image_int8)

        # well_dict_out = dict({well_name_conv: well_dict})

    return {}


def build_ff_from_yx1(data_root, overwrite_flag=False, ch_to_use=0, dir_list=None, write_dir=None, rs_res=None, metadata_only_flag=False):

    read_dir_root = os.path.join(data_root, 'raw_image_data', 'YX1') 
    if write_dir is None:
        write_dir = os.path.join(data_root, 'built_image_data', 'YX1') 
        
    # handle paths
    if dir_list is None:
        # Get a list of directories
        dir_list_raw = sorted(glob.glob(read_dir_root + "*"))
        dir_list = []
        for dd in dir_list_raw:
            if os.path.isdir(dd):
                dir_list.append(path_leaf(dd))

    if rs_res is None:
        rs_res = np.asarray([3.2, 3.2])

    # filter for desired directories
    dir_indices = [d for d in range(len(dir_list)) if "ignore" not in dir_list[d]]

    for d in dir_indices:

        # initialize dictionary to metadata
        sub_name = dir_list[d]
        dir_path = os.path.join(read_dir_root, sub_name, "")

        # depth_dir = os.path.join(write_dir, "stitched_depth_images", sub_name)
        ff_dir = os.path.join(write_dir, "stitched_FF_images",  sub_name)

        # if not os.path.isdir(depth_dir):
        #     os.makedirs(depth_dir)
        if not os.path.isdir(ff_dir):
            os.makedirs(ff_dir)

        # Read in  metadata object
        image_list = sorted(glob.glob(dir_path + "*.nd2")) 
        if len(image_list) > 1:
            raise Exception("Multiple nd2 files found in " + sub_name + ". Please move extra nd2 files to a subfolder." )
        elif len(image_list) == 0:
            raise Exception("No nd2 files found in " + sub_name)

        # Read in experiment metadata 
        print(f"Processing {sub_name}...")

        imObject= nd2.ND2File(image_list[0])
        im_shape = imObject.shape
        n_time_points = im_shape[0]
        n_wells = im_shape[1]
        n_z_slices = im_shape[2]

        # pull dask array
        im_array_dask = imObject.to_dask()
        # use first 10 frames to infer time resolution

        # extract frame times
        n_frames_total = imObject.frame_metadata(0).contents.frameCount
        frame_time_vec = [imObject.frame_metadata(i).channels[0].time.relativeTimeMs / 1000 for i in
                        range(0, n_frames_total, im_shape[2])]
        # check for common nd2 artifact where time stamps jump midway through
        dt_frame_approx = (imObject.frame_metadata(n_z_slices).channels[0].time.relativeTimeMs -
                        imObject.frame_metadata(0).channels[0].time.relativeTimeMs) / 1000
        jump_ind = np.where(np.diff(frame_time_vec) > 2*dt_frame_approx)[0] # typically it is multiple orders of magnitude to large
        if len(jump_ind) > 0:
            jump_ind = jump_ind[0]
            # prior to this point we will just use the time stamps. We will extrapolate to get subsequent time points
            nf = jump_ind - 1 - int(jump_ind/2)
            dt_frame_est = (frame_time_vec[jump_ind-1] - frame_time_vec[int(jump_ind/2)]) / nf
            base_time = frame_time_vec[jump_ind-1]
            for f in range(jump_ind, len(frame_time_vec)):
                frame_time_vec[f] = base_time + dt_frame_est*(f - jump_ind)
        frame_time_vec = np.asarray(frame_time_vec)

        # get image resolution
        voxel_size = imObject.voxel_size()
        voxel_yx = np.asarray([voxel_size[1], voxel_size[0]])
        rs_factor = np.divide(voxel_yx, rs_res)

        rs_dims_yx = np.round(np.multiply(np.asarray(im_shape[3:]), rs_factor)).astype(int)
        # resample images to a standardized resolution


        # # initialize metadata data frame
        # well_df = pd.DataFrame([], columns=['well', 'nd2_series_num', 'microscope', 'time_int', 'Height (um)', 'Width (um)', 'Height (px)', 'Width (px)', 'Objective', 'Time (s)'])

        # read in plate map
        plate_map_xl = pd.ExcelFile(dir_path + sub_name + "_plate_map.xlsx")
        series_map = plate_map_xl.parse("series_number_map").iloc[:8, 1:13]

        well_name_list = []
        well_ind_list = []
        col_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        row_letter_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
        for c in range(len(col_id_list)):
            for r in range(len(row_letter_list)):
                ind_float = series_map.iloc[r, c]
                if ~np.isnan(ind_float):
                    well_name = row_letter_list[r] + f"{col_id_list[c]:02}"
                    well_name_list.append(well_name)
                    well_ind_list.append(int(ind_float))

        si = np.argsort(well_ind_list)
        well_name_list_sorted = np.asarray(well_name_list)[si].tolist()
        
        # generate longform vectors
        well_name_list_long = np.repeat(well_name_list_sorted, n_time_points)
        well_ind_list_long = np.repeat(np.asarray(well_ind_list)[si], n_time_points)

        # check that assigned well IDs match recorded stage positions
        stage_xyz_array = np.empty((n_wells*n_time_points, 3))
        well_id_array = np.empty((n_wells*n_time_points,))
        time_id_array = np.empty((n_wells*n_time_points,))
        iter_i = 0
        for w in range(n_wells):
            for t in range(n_time_points):
                base_ind = t*n_wells + w
                slice_ind = base_ind*n_z_slices
                
                stage_xyz_array[iter_i, :] = np.asarray(imObject.frame_metadata(slice_ind).channels[0].position.stagePositionUm)
                well_id_array[iter_i] = w
                time_id_array[iter_i] = t
                iter_i += 1


        # use clustering to double check well assignments
        row_letter_vec = np.asarray([id[0] for id in well_name_list_long])
        col_num_vec = np.asarray([int(id[1:]) for id in well_name_list_long])
        row_index = np.unique(row_letter_vec)
        col_index = np.unique(col_num_vec)

        # Check rows
        row_clusters = KMeans(n_init="auto", n_clusters=len(row_index)).fit(stage_xyz_array[:, 1].reshape(-1, 1))
        row_si = np.argsort(np.argsort(row_clusters.cluster_centers_.ravel()))
        row_ind_pd = row_si[row_clusters.labels_]
        row_letter_pd = row_index[row_ind_pd]
        assert np.all(row_letter_pd==row_letter_vec)

        col_clusters = KMeans(n_init="auto", n_clusters=len(col_index)).fit(stage_xyz_array[:, 0].reshape(-1, 1))
        col_si = np.argsort(np.argsort(col_clusters.cluster_centers_.ravel()))
        col_ind_pd = col_si[col_clusters.labels_]
        col_num_pd = col_index[len(col_index)-col_ind_pd-1]
        assert np.all(col_num_pd==col_num_vec)

        # generate metadata dataframe
        well_df = pd.DataFrame(well_name_list_long[:, np.newaxis], columns=["well"])
        well_df["nd2_series_num"] = well_ind_list_long
        well_df["microscope"] = "YX1"
        time_int_list = np.tile(np.arange(0, n_time_points), n_wells)
        well_df["time_int"] = time_int_list
        well_df["Height (um)"] = im_shape[3]*voxel_size[1]
        well_df["Width (um)"] = im_shape[4]*voxel_size[0]
        well_df["Height (px)"] = im_shape[3]
        well_df["Width (px)"] = im_shape[4]
        well_df["Channel"] = imObject.frame_metadata(0).channels[0].channel.name
        well_df["Objective"] = imObject.frame_metadata(0).channels[0].microscope.objectiveName
        time_ind_vec = []
        for n in range(n_wells):
            time_ind_vec += np.arange(n, n_wells*n_time_points, n_wells).tolist()
        well_df["Time (s)"] = frame_time_vec[time_ind_vec]

        # print(f'Building full-focus images in directory {d+1:01} of ' + f'{len(dir_indices)}')
        # temp = pmap(process_frame, range(n_wells*n_time_points), 
        #                         (im_array_dask, well_name_list_long, time_int_list, well_int_list, ff_dir, depth_dir, overwrite_flag))

        # get device
        device = (
                "cuda"
                if torch.cuda.is_available() 
                else "cpu"
            )

        # for indexing dask array
        well_int_list = np.repeat(np.arange(0, n_wells), n_time_points)

        # call FF function
        if not metadata_only_flag:
            for w in tqdm(range(n_wells*n_time_points)):
                process_frame(w, im_array_dask, well_name_list_long, time_int_list, well_int_list, ff_dir, device=device, 
                                overwrite_flag=overwrite_flag, rs_dims_yx=rs_dims_yx, rs_res_yx=rs_res)
        
        
        first_time = np.min(well_df['Time (s)'].copy())
        well_df['Time Rel (s)'] = well_df['Time (s)'] - first_time
        

        # load previous metadata
        metadata_path = os.path.join(ff_dir, 'metadata.csv')
        well_df.to_csv(metadata_path)

        imObject.close()
        # with open(os.path.join(ff_dir, 'metadata.pickle'), 'wb') as handle:
        #     pickle.dump(metadata_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print('Done.')



if __name__ == "__main__":

    overwrite_flag = True
    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    dir_list = ["20231206"]
    # build FF images
    # build_ff_from_keyence(data_root, write_dir=write_dir, overwrite_flag=True, dir_list=dir_list, ch_to_use=4)
    # stitch FF images
    build_ff_from_yx1(data_root=data_root, dir_list=dir_list, overwrite_flag=overwrite_flag)