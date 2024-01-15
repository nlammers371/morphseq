import cv2

def doLap(image, lap_size=3, blur_size=3):

    # YOU SHOULD TUNE THESE VALUES TO SUIT YOUR NEEDS
#     kernel_size = 5  # Size of the laplacian window
#     blur_size = 5  # How big of a kernal to use for the gaussian blur
    # Generally, keeping these two values the same or very close works well
    # Also, odd numbers, please...

    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    return cv2.Laplacian(blurred, cv2.CV_64F, ksize=lap_size)