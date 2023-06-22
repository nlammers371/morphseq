import numpy as np
import os
import glob2 as glob
import cv2
from aicsimageio import AICSImage
import ntpath
# from PIL import Image

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# project_name = '20230525'
# db_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/built_keyence_data_v2/"
db_path = "E:/Nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/built_keyence_data_v2/"
path_to_images = os.path.join(db_path, 'stitched_ff_images', '*')
project_list = glob.glob(path_to_images)
#
n_im = 1000
image_i = 202
# set starting point
# im_dims = [641, 1158]
overwrite_flag = False
skip_labeled_flag = False
if overwrite_flag:
    skip_labeled_flag = False

# set random seed for reproducibility
seed = 126
suffix = "_live_dead_bubble"
np.random.seed(seed)

# make write paths
label_path = os.path.join(db_path, 'UNET_training', str(seed) + suffix, 'annotations', '')


# get list of existing labels (if any)
existing_labels = sorted(glob.glob(label_path + "*tif"))


for image_i in range(len(existing_labels)):

    # open labels if they exist (and we want to keep them)
    lb_path_full = existing_labels[image_i]

    lbObject = AICSImage(lb_path_full)
    lb_temp = np.squeeze(lbObject.data)
    lb_temp[np.where((lb_temp == 3) | (lb_temp == 5))] = 0
    lb_temp[np.where(lb_temp == 4)] = 3

    AICSImage(lb_temp.astype(np.uint8)).save(lb_path_full)