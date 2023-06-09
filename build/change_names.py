import numpy as np
import napari
import os
import glob2 as glob
import cv2
from aicsimageio import AICSImage
import ntpath

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# project_name = '20230525'
db_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/built_keyence_data/"
path_to_images = os.path.join(db_path, 'stitched_ff_images', '202305*')
project_list = glob.glob(path_to_images)
# db_path = "E:/Nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/built_keyence_data/"
n_im = 1000
# im_dims = [641, 1158]
overwrite_flag = False
skip_labeled_flag = True
# set random seed for reproducibility
seed = 126
np.random.seed(seed)

# make write paths
image_path = os.path.join(db_path, 'UNET_training', str(seed), 'images', '')
label_path = os.path.join(db_path, 'UNET_training', str(seed), 'labels', '')
if not os.path.isdir(image_path):
    os.makedirs(image_path)
if not os.path.isdir(label_path):
    os.makedirs(label_path)

# get list of existing labels (if any)
existing_labels = sorted(glob.glob(label_path + "*tif"))

# set path to images
# path_to_images = os.path.join(db_path, 'stitched_FF_images', project_name, '')

# select subset of images to label
im_list = []
project_index = []
project_ind_long = []
for ind, p in enumerate(project_list):
    im_list_temp = glob.glob(os.path.join(p, '*.tif'))
    im_list += im_list_temp
    _, tail = ntpath.split(p)
    project_index.append(tail)
    project_ind_long += [ind]*len(im_list_temp)


im_lb_indices = np.random.choice(range(len(im_list)), n_im, replace=False)

for image_i in range(len(im_lb_indices)):
    prefix = project_index[project_ind_long[im_lb_indices[image_i]]]

    # load image
    im_path = im_list[im_lb_indices[image_i]]
    _, im_name = ntpath.split(im_path)

    # open labels if they exist (and we want to keep them)
    lb_path_new = label_path + prefix + '_' + im_name
    im_path_new = image_path + prefix + '_' + im_name
    lb_path_old = label_path + im_name
    im_path_old = image_path + im_name

    if lb_path_old in existing_labels:
        os.rename(im_path_old, im_path_new)
        os.rename(lb_path_old, lb_path_new)
