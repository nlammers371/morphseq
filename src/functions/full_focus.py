import numpy as np
import cv2

def doLap(image):

    # YOU SHOULD TUNE THESE VALUES TO SUIT YOUR NEEDS
    kernel_size = 5  # Size of the laplacian window
    blur_size = 5  # How big of a kernal to use for the gaussian blur
    # Generally, keeping these two values the same or very close works well
    # Also, odd numbers, please...

    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    return cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)


#
#   This routine finds the points of best focus in all images and produces a merged result...
#
def focus_stack(images):
    # print
    # "Computing the laplacian of the blurred images"
    laps = []
    for i in range(len(images)):
        # print
        # "Lap {}".format(i)
        laps.append(doLap(images[i]))

    laps = np.asarray(laps)
    # print
    # "Shape of array of laplacians = {}".format(laps.shape)

    output = np.zeros(shape=images[0].shape, dtype=images[0].dtype)

    abs_laps = np.absolute(laps)
    maxima = abs_laps.max(axis=0)
    bool_mask = abs_laps == maxima
    mask = bool_mask.astype(np.uint8)
    for i in range(0, len(images)):
        output = cv2.bitwise_not(images[i], output, mask=mask[i])

    return 255 - output