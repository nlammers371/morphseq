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

morph_label_flag = False
focus_label_flag = False
# db_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/built_keyence_data_v2/"
db_path = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\built_keyence_data\\" # "D:\\Nick\\morphseq\\built_keyence_data\\"   #
path_to_images = os.path.join(db_path, 'stitched_ff_images', '*')
project_list = glob.glob(path_to_images)

# set starting point
# im_dims = [641, 1158]
overwrite_flag = False
skip_labeled_flag = False
if overwrite_flag:
    skip_labeled_flag = False

# set random seed for reproducibility
# seed = 678
# suffix = "_focus"
seed = 932
suffix = "_yolk_head_tail"


# make write paths
image_path_in = image_path = os.path.join(db_path, 'UNET_Training_morph', str(seed) + suffix, 'images', '')
label_path_in = os.path.join(db_path, 'UNET_Training_morph', str(seed) + suffix, 'annotations', '')

image_path_head = os.path.join(db_path, 'UNET_training_head', str(seed) + suffix, 'images', '')
label_path_head = os.path.join(db_path, 'UNET_training_head', str(seed) + suffix, 'annotations', '')

image_path_yolk = os.path.join(db_path, 'UNET_training_yolk', str(seed) + suffix, 'images', '')
label_path_yolk = os.path.join(db_path, 'UNET_training_yolk', str(seed) + suffix, 'annotations', '')

image_path_tail = os.path.join(db_path, 'UNET_training_tail', str(seed) + suffix, 'images', '')
label_path_tail = os.path.join(db_path, 'UNET_training_tail', str(seed) + suffix, 'annotations', '')

if not os.path.isdir(image_path_head):
    os.makedirs(image_path_head)
if not os.path.isdir(label_path_head):
    os.makedirs(label_path_head)
if not os.path.isdir(image_path_yolk):
    os.makedirs(image_path_yolk)
if not os.path.isdir(label_path_yolk):
    os.makedirs(label_path_yolk)
if not os.path.isdir(image_path_tail):
    os.makedirs(image_path_tail)
if not os.path.isdir(label_path_tail):
    os.makedirs(label_path_tail)

# get list of existing labels (if any)
existing_labels = sorted(glob.glob(label_path_in + "*tif"))
existing_label_names = []
for ex in existing_labels:
    _, im_name = ntpath.split(ex)
    existing_label_names.append(im_name)

existing_images = sorted(glob.glob(image_path_in + "*tif"))
existing_image_names = []
for ex in existing_images:
    _, im_name = ntpath.split(ex)
    existing_image_names.append(im_name)

transfer_name_list = [i for i in existing_image_names if i in existing_label_names]

np.random.seed(334)
toggle = True
for image_i in range(len(transfer_name_list)):
    t_name = transfer_name_list[image_i]

    # read in image
    im_path = os.path.join(image_path_in, t_name)
    t_image = cv2.imread(im_path)

    # read label
    lb_path = os.path.join(label_path_in, t_name)
    t_label = cv2.imread(lb_path)

    # make an yolk-only label version
    t_label_yolk = t_label.copy()
    t_label_yolk[np.where(t_label_yolk != 1)] = 0

    # make a head-only label version
    t_label_head = t_label.copy()
    t_label_head[np.where(t_label_head != 2)] = 0  # zero out embryo_labels
    t_label_head[np.where(t_label_head == 2)] = 1

    # make a tail-only label version
    t_label_tail = t_label.copy()
    t_label_tail[np.where(t_label_tail != 3)] = 0  # zero out embryo_labels
    t_label_tail[np.where(t_label_tail == 3)] = 1

    # write embryo image and labels to file

    cv2.imwrite(os.path.join(image_path_yolk, t_name), t_image)
    cv2.imwrite(os.path.join(label_path_yolk, t_name), t_label_yolk)

    cv2.imwrite(os.path.join(image_path_head, t_name), t_image)
    cv2.imwrite(os.path.join(label_path_head, t_name), t_label_head)

    cv2.imwrite(os.path.join(image_path_tail, t_name), t_image)
    cv2.imwrite(os.path.join(label_path_tail, t_name), t_label_tail)




# load label file if it exists
# if os.path.isfile(image_path + lb_name):
#     lb_object = AICSImage(image_path + lb_name)
#     lb_data = np.squeeze(lb_object.data)
#     labels_layer = viewer.add_labels(lb_data, name='annotation')

# if __name__ == '__main__':
#     napari.run()