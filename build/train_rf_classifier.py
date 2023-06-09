from skimage import filters
import numpy as np
import scipy.ndimage as nd
from sklearn.ensemble import RandomForestClassifier
import os
import glob2 as glob
from aicsimageio import AICSImage

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
        feature_stack[:, ind_base+1] = filters.difference_of_gaussians(image, low_sigma=sigma_list[s]).ravel()
        feature_stack[:, ind_base+2] = nd.gaussian_laplace(image, sigma_list[s]).ravel()
        feature_stack[:, ind_base+3] = filters.sobel(gauss).ravel()

    # return stack as numpy-array
    return feature_stack

def format_data(feature_stack, annotation):
    # reformat the data to match what scikit-learn expects
    # transpose the feature stack
    X = feature_stack
    # make the annotation 1-dimensional
    y = annotation.ravel()

    # remove all pixels from the feature and annotations which have not been annotated
    mask = y > 0
    X = X[mask]
    y = y[mask]

    return X, y

if __name__ == '__main__':
    # project_name = '20230525'
    db_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/built_keyence_data/"

    # set random seed for reproducibility
    seed = 126
    np.random.seed(seed)

    # make write paths
    image_path = os.path.join(db_path, 'RF_training', str(seed), 'images', '')
    label_path = os.path.join(db_path, 'RF_training', str(seed), 'labels', '')

    image_list = sorted(glob.glob(image_path + '*.tif'))
    label_list = sorted(glob.glob(label_path + '*.tif'))

    # build training dataset
    for i in [0]:#range(len(image_list)):
        # load data
        imObject = AICSImage(image_list[i])
        image = np.squeeze(imObject.data)

        lbObject = AICSImage(label_list[i])
        annotation = np.squeeze(lbObject.data)

        # generate features
        feature_stack = generate_feature_stack(image[ :, :, 0])

        # reformat
        X, y = format_data(feature_stack, annotation)

    classifier = RandomForestClassifier(max_depth=10, random_state=0)
    classifier.fit(X, y)

    test_probs = classifier.predict_proba(feature_stack)
    test_seg = classifier.predict(feature_stack)
    im_prob_test = np.reshape(test_probs[:, 1], image[:, :, 0].shape)
    im_seg_test = np.reshape(test_seg, image[:, :, 0].shape)