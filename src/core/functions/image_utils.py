import cv2
import numpy as np
import torch
import skimage
from skimage.morphology import label
import torch.nn.functional as F
# Simple replacement for set_inputs_to_device to avoid pythae dependency
def set_inputs_to_device(input_tensor, device):
    inputs_on_device = input_tensor
    if device == "cuda":
        inputs_on_device = input_tensor.cuda()
    return inputs_on_device
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

    """
    Normalize embryo and yolk masks to clean binary arrays and select the
    correct embryo instance when given an integer-labeled mask.

    Accepts any of:
    - Integer-labeled embryo mask where pixel values are embryo IDs (SAM2)
    - Binary embryo mask in {0,1}
    - Binary embryo mask in {0,255}

    Yolk mask may be absent (all zeros), {0,1}, or {0,255}.
    """

    # --- Embryo mask normalization ---
    em = np.asarray(im_mask)
    em_max = int(em.max()) if em.size else 0

    if em_max > 1:
        # Integer-labeled mask (SAM2 style): pick the requested region_label
        lbi = int(row["region_label"]) if "region_label" in row else None
        if not lbi or lbi == 0:
            # Fallback: treat any non-zero as embryo
            em_bin = (em > 0).astype(int)
        else:
            em_bin = (em == lbi).astype(int)
    else:
        # Binary in {0,1} possibly floats
        em_bin = (em > 0).astype(int)

    # Some legacy callers may pass {0,255}; handle that as well
    if em_max >= 255:
        em_bin = (em > 127).astype(int)

    # Morph-close to fill small holes
    if em_bin.any():
        i_disk = disk(close_radius)
        em_bin = binary_closing(em_bin.astype(bool), i_disk).astype(int)

    # --- Yolk mask normalization (optional) ---
    yk = np.asarray(im_yolk)
    yk_max = int(yk.max()) if yk.size else 0
    if yk_max == 0:
        yolk_bin = np.zeros_like(em_bin)
    elif yk_max > 1 and yk_max < 255:
        # assume labels; treat any non-zero as yolk
        yolk_bin = (yk > 0).astype(int)
    else:
        # {0,255}
        yolk_bin = (yk > 127).astype(int)

    # Keep only yolk connected to embryo ROI
    if em_bin.any() and yolk_bin.any():
        intersect = (yolk_bin & em_bin).astype(int)
        if intersect.sum() < 10:
            yolk_bin = np.zeros_like(yolk_bin)
        else:
            y_lb = label(yolk_bin)
            i_lb = label(intersect)
            if i_lb.max() <= 1:
                # Single contact component
                keep_label = int(np.unique(y_lb[intersect == 1])[-1])
                yolk_bin = (y_lb == keep_label).astype(int)
            else:
                # Choose yolk label overlapping with largest contact component
                rgi = regionprops(i_lb)
                a_vec = [r.area for r in rgi]
                i_max = int(np.argmax(a_vec)) + 1
                keep_label = int(np.unique(y_lb[i_lb == i_max])[-1])
                yolk_bin = (y_lb == keep_label).astype(int)

    return em_bin.astype(int), yolk_bin.astype(int)


def crop_embryo_image(im_ff_rotated, emb_mask_rotated, im_yolk_rotated, outshape, return_metrics=False):

    if np.sum(emb_mask_rotated) == 0:
        if return_metrics:
            return np.zeros(outshape), np.zeros(outshape), np.zeros(outshape), True
        return np.zeros(outshape), np.zeros(outshape), np.zeros(outshape)

    y_indices = np.where(np.max(emb_mask_rotated, axis=1) > 0.5)[0]
    x_indices = np.where(np.max(emb_mask_rotated, axis=0) > 0.5)[0]
    if y_indices.size == 0 or x_indices.size == 0:
        # Degenerate mask after interpolation/rotation; treat as empty crop to keep pipeline moving.
        if return_metrics:
            return np.zeros(outshape), np.zeros(outshape), np.zeros(outshape), True
        return np.zeros(outshape), np.zeros(outshape), np.zeros(outshape)
    y_mean = int(np.mean(y_indices))
    x_mean = int(np.mean(x_indices))

    fromshape = emb_mask_rotated.shape
    raw_range_y = [y_mean - int(outshape[0] / 2), y_mean + int(outshape[0] / 2)]
    from_range_y = np.asarray([np.max([raw_range_y[0], 0]), np.min([raw_range_y[1], fromshape[0]])])
    to_range_y = [0 + (from_range_y[0] - raw_range_y[0]), outshape[0] + (from_range_y[1] - raw_range_y[1])]

    raw_range_x = [x_mean - int(outshape[1] / 2), x_mean + int(outshape[1] / 2)]
    from_range_x = np.asarray([np.max([raw_range_x[0], 0]), np.min([raw_range_x[1], fromshape[1]])])
    to_range_x = [0 + (from_range_x[0] - raw_range_x[0]), outshape[1] + (from_range_x[1] - raw_range_x[1])]

    # Calculate mask area before and after cropping for out_of_frame detection
    mask_area_before = np.sum(emb_mask_rotated > 0.5)

    if len(im_ff_rotated.shape) == 2:
        im_cropped = np.zeros(outshape).astype(np.uint8)
        im_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
            im_ff_rotated[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]
    else:
        im_cropped = np.zeros((im_ff_rotated.shape[0], outshape[0], outshape[1]), dtype=im_ff_rotated.dtype)
        im_cropped[:, to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
            im_ff_rotated[:, from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

    emb_mask_cropped = np.zeros(outshape)#.astype(np.uint8)
    emb_mask_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
        emb_mask_rotated[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

    yolk_mask_cropped = np.zeros(outshape)#.astype(np.uint8)
    yolk_mask_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
        im_yolk_rotated[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

    # Calculate out_of_frame flag based on mask area retention
    if return_metrics:
        mask_area_after = np.sum(emb_mask_cropped > 0.5)
        area_retained = mask_area_after / mask_area_before if mask_area_before > 0 else 0.0
        # Flag if less than 98% of mask area is retained
        out_of_frame_flag = area_retained < 0.98
        return im_cropped, emb_mask_cropped, yolk_mask_cropped, out_of_frame_flag

    return im_cropped, emb_mask_cropped, yolk_mask_cropped

    
def get_embryo_angle(mask_emb_rs, mask_yolk_rs):
    rp = regionprops(mask_emb_rs)
    if not rp:
        return 0.0  # Return a default angle if no regions are found
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
