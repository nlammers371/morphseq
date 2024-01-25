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

    return data_FF