import numpy as np
import os
import glob2 as glob
import ntpath
import cv2

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# project_name = '20230525'
# db_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/built_keyence_data_v2/"
db_path = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\built_keyence_data\\"
path_to_images = os.path.join(db_path, 'stitched_ff_images', '*')
project_list = glob.glob(path_to_images)
# db_path = "E:/Nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/built_keyence_data/"
n_im = 500

# set random seed for reproducibility
# seed_str = str(126) # points to training subfolder I've been useing
seed = 671
np.random.seed(seed)

# make write paths
test_image_path = os.path.join(db_path, 'focus_UNET_training', str(seed) + '_test_node', 'images', '')
if not os.path.isdir(test_image_path):
    os.makedirs(test_image_path)


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


im_indices = np.random.choice(range(len(im_list)), n_im, replace=False)

for image_i in range(len(im_indices)):
    prefix = project_index[project_ind_long[im_indices[image_i]]]

    # load image
    im_path = im_list[im_indices[image_i]]
    _, im_name = ntpath.split(im_path)

    # open labels if they exist (and we want to keep them)
    im_path_new = test_image_path + prefix + '_' + im_name

    im_temp = cv2.imread(im_path)
    cv2.imwrite(im_path_new, im_temp)

