import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.functions.dataset_utils import set_inputs_to_device

def doLap(image, lap_size=3, blur_size=3):

    # YOU SHOULD TUNE THESE VALUES TO SUIT YOUR NEEDS
#     kernel_size = 5  # Size of the laplacian window
#     blur_size = 5  # How big of a kernal to use for the gaussian blur
    # Generally, keeping these two values the same or very close works well
    # Also, odd numbers, please...

    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    return cv2.Laplacian(blurred, cv2.CV_64F, ksize=lap_size)


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
    data_tensor = torch.reshape(torch.tensor(data_zyx), (data_zyx.shape[0], 1, data_zyx.shape[1], data_zyx.shape[2]))
    data_tensor = set_inputs_to_device(data_tensor, device)

    # get Gaussian Blur
    GB = F.conv2d(input=data_tensor, weight=gf_tensor, padding="same")

    # calculate difference from raw image. This is used to quantify how focused each pixel is
    weights = np.abs(GB - data_tensor)

    # calculate focus-weighted projection
    data_weighted = torch.multiply(weights, data_tensor)
    data_FF = torch.squeeze(torch.divide(torch.sum(data_weighted, axis=0), torch.sum(weights, axis=0)))

    return data_FF
    # calculate FF image
    # laps = np.zeros(data_zyx.shape, dtype=np.float64)
    # for i in range(data_zyx.shape[0]):
    #     laps[i, :, :] = doLap(data_zyx[i, :, :], lap_size=filter_size, blur_size=filter_size)
        # laps_d.append(doLap(data_zyx[i, :, :], lap_size=7, blur_size=7))  # I've found that depth stacking works better with larger filters

    # laps = np.asarray(laps)
    abs_laps = torch.abs(torch.squeeze(LoG))