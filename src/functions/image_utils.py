import cv2
import numpy as np
import torch
import skimage
from skimage.morphology import label
import torch.nn.functional as F
from src.functions.dataset_utils import set_inputs_to_device
from skimage.morphology import disk, binary_closing, remove_small_objects
from skimage.measure import label, regionprops, find_contours
import scipy


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


def process_masks(im_mask, im_yolk, row, close_radius=15):

    im_mask = np.round(im_mask / 255 * 2 - 1).astype(np.uint8) # this is confusing, but effectively, any pixel > 0.75*255=1, else 0
    im_mask_lb = label(im_mask)
    
    im_yolk = np.round(im_yolk / 255 * 2 - 1).astype(np.uint8)
    if np.any(im_yolk == 1):
        im_yolk = skimage.morphology.remove_small_objects(im_yolk.astype(bool), min_size=75).astype(int)  # remove small stuff

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

    return im_mask_ft, im_yolk


def crop_embryo_image(im_ff_rotated, emb_mask_rotated, im_yolk_rotated, outshape):

    y_indices = np.where(np.max(emb_mask_rotated, axis=1) == 1)[0]
    x_indices = np.where(np.max(emb_mask_rotated, axis=0) == 1)[0]
    y_mean = int(np.mean(y_indices))
    x_mean = int(np.mean(x_indices))

    fromshape = emb_mask_rotated.shape
    raw_range_y = [y_mean - int(outshape[0] / 2), y_mean + int(outshape[0] / 2)]
    from_range_y = np.asarray([np.max([raw_range_y[0], 0]), np.min([raw_range_y[1], fromshape[0]])])
    to_range_y = [0 + (from_range_y[0] - raw_range_y[0]), outshape[0] + (from_range_y[1] - raw_range_y[1])]

    raw_range_x = [x_mean - int(outshape[1] / 2), x_mean + int(outshape[1] / 2)]
    from_range_x = np.asarray([np.max([raw_range_x[0], 0]), np.min([raw_range_x[1], fromshape[1]])])
    to_range_x = [0 + (from_range_x[0] - raw_range_x[0]), outshape[1] + (from_range_x[1] - raw_range_x[1])]

    if len(im_ff_rotated.shape) == 2:
        im_cropped = np.zeros(outshape).astype(np.uint8)
        im_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
            im_ff_rotated[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]
    else:
        im_cropped = np.zeros((im_ff_rotated.shape[0], outshape[0], outshape[1]), dtype=im_ff_rotated.dtype)
        im_cropped[:, to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
            im_ff_rotated[:, from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]
        
    emb_mask_cropped = np.zeros(outshape).astype(np.uint8)
    emb_mask_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
        emb_mask_rotated[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

    yolk_mask_cropped = np.zeros(outshape).astype(np.uint8)
    yolk_mask_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
        im_yolk_rotated[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

    return im_cropped, emb_mask_cropped, yolk_mask_cropped

    
def get_embryo_angle(mask_emb_rs, mask_yolk_rs):
    rp = regionprops(mask_emb_rs)
    angle = rp[0].orientation

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

    return angle_to_use


def doLap(image, lap_size=3, blur_size=3):

    # YOU SHOULD TUNE THESE VALUES TO SUIT YOUR NEEDS
#     kernel_size = 5  # Size of the laplacian window
#     blur_size = 5  # How big of a kernal to use for the gaussian blur
    # Generally, keeping these two values the same or very close works well
    # Also, odd numbers, please...

    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    return cv2.Laplacian(blurred, cv2.CV_64F, ksize=lap_size)

# Based off of a focus-stacking package in FIJI:
# https://github.com/fiji/Time_Lapse/blob/Time_Lapse-2.1.1/src/main/java/sc/fiji/timelapse/Gaussian_Stack_Focuser.java
def gaussian_focus_stacker(data_zyx, filter_size, device):

    # dummy indexer
    ind = filter_size // 2 + 1

    # get Gaussian filter
    gf = np.zeros((2*filter_size+1, 2*filter_size+1))
    gf[filter_size, filter_size] = 1
    gf = cv2.GaussianBlur(gf, (filter_size, filter_size), 0)
    gf = gf[ind:-ind, ind:-ind]
    gf_tensor = set_inputs_to_device(torch.reshape(torch.tensor(gf), (1, 1, filter_size, filter_size)), device)

    # convert image to tensor 
    data_tensor = torch.reshape(data_zyx, (data_zyx.shape[0], 1, data_zyx.shape[1], data_zyx.shape[2]))
    data_tensor = set_inputs_to_device(data_tensor, device)

    # get Gaussian Blur
    GB = F.conv2d(input=data_tensor, weight=gf_tensor, padding="same")

    # calculate difference from raw image. This is used to quantify how focused each pixel is
    weights = torch.abs(GB - data_tensor) + 0.1 # add pseudocounts to avoid dvision by zero

    # calculate focus-weighted projection
    data_weighted = torch.multiply(weights, data_tensor)
    data_FF = torch.squeeze(torch.divide(torch.sum(data_weighted, axis=0), torch.sum(weights, axis=0)))

    return data_FF



def LoG_focus_stacker(data_zyx, filter_size, device):

    # dummy indexer
    ind = filter_size // 2 + 1
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
    data_tensor = torch.reshape(data_zyx, (data_zyx.shape[0], 1, data_zyx.shape[1], data_zyx.shape[2]))
    data_tensor = set_inputs_to_device(data_tensor, device)

    # get Gaussian Blur
    GB = F.conv2d(input=data_tensor, weight=gf_tensor, padding="same")
    LoG = F.conv2d(input=GB, weight=lpf_tensor, padding="same")

    abs_laps = torch.abs(torch.squeeze(LoG))
    maxima = torch.max(abs_laps, axis=0)
    bool_mask = abs_laps == maxima.values
    data_FF = torch.max(torch.multiply(bool_mask, data_zyx), axis=0).values

    # # calculate difference from raw image. This is used to quantify how focused each pixel is
    # weights = torch.abs(GB - data_tensor) + 0.1 # add pseudocounts to avoid dvision by zero

    # # calculate focus-weighted projection
    # data_weighted = torch.multiply(weights, data_tensor)
    # data_FF = torch.squeeze(torch.divide(torch.sum(data_weighted, axis=0), torch.sum(weights, axis=0)))

    return data_FF, abs_laps