from skimage import filters
import numpy as np
import scipy.ndimage as nd
def generate_feature_stack(image):
    # set scale(s) for feature extraction
    sigma_list = [1, 3, 5, 10, 15]

    # initialize array to store features
    feature_stack = np.empty((image.size, len(sigma_list)*4 + 1))
    feature_stack[:, 0] = image.ravel()

    for s in range(len(sigma_list)):
        ind_base = s*4 + 1
        # determine features
        gauss = filters.gaussian(image, sigma=sigma_list[s])
        feature_stack[:, ind_base] = gauss.ravel()
        feature_stack[:, ind_base+1] = filters.difference_of_gaussians(image, low_sigma=sigma_list[s])
        feature_stack[:, ind_base+2] = nd.gaussian_laplace(image, sigma_list[s])
        feature_stack[:, ind_base+3] = filters.sobel(gauss)

    # return stack as numpy-array
    return feature_stack
