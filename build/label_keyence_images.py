import numpy as np
import napari
import os
import glob2 as glob
import cv2
from aicsimageio import AICSImage
import ntpath
# from PIL import Image
from qtpy.QtCore import QTimer
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# project_name = '20230525'
db_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/built_keyence_data/"
path_to_images = os.path.join(db_path, 'stitched_ff_images', '*')
project_list = glob.glob(path_to_images)
# db_path = "E:/Nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/built_keyence_data/"
n_im = 100
# im_dims = [641, 1158]
overwrite_flag = False

# set random seed for reproducibility
seed = 126
np.random.seed(seed)

# make write paths
image_path = os.path.join(db_path, 'RF_training', str(seed), 'images', '')
label_path = os.path.join(db_path, 'RF_training', str(seed), 'labels', '')
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
for p in project_list:
    im_list_temp = glob.glob(os.path.join(p , '*.tif'))
    im_list += im_list_temp
im_lb_indices = np.random.choice(range(len(im_list)), n_im, replace=False)

# #
i = 6
# check to see if we have label file for this image
im_path = im_list[im_lb_indices[i]]
im_name = im_path.replace(path_to_images, '')
lb_name = 'labels_' + im_name

# load
# initialize viewer
image_i = 6
while image_i < len(im_lb_indices)-1:
    # load image
    im_path = im_list[im_lb_indices[image_i]]
    im_temp = cv2.imread(im_path)
    # save
    _, im_name = ntpath.split(im_path)
    cv2.imwrite(image_path + im_name, im_temp)

    # with napari.gui_qt() as app:

    # open viewer
    viewer = napari.view_image(im_temp[:, :, 0],  colormap="gray")

    # open labels if they exist (and we want to keep them)
    lb_path_full = label_path + im_name
    if not overwrite_flag and (lb_path_full in existing_labels):
        lbObject = AICSImage(lb_path_full)
        lb_temp = np.squeeze(lbObject.data)

    else:
        lbObject = AICSImage(os.path.join(db_path, 'well_mask.tif'))
        lb_temp = np.squeeze(lbObject.data)

    viewer.add_labels(lb_temp, name='Labels')
    napari.run()

    # save new  label layer
    lb_layer = viewer.layers["Labels"]
    AICSImage(lb_layer.data.astype(np.uint8)).save(lb_path_full)

        # time_in_msec = 1000
        # QTimer().singleShot(time_in_msec, app.quit)

    # viewer.close()
    # cv2.imwrite(label_path + im_name, lb_layer.data)

    wait = input("Press Enter to continue to next image. \nPress 'x' then Enter to exit. \nType a digit then Enter to jump to a specific image")
    if wait == 'x':
        break
    elif isinstance(wait, int):
        image_i = wait
    else:
        image_i += 1
        print(image_i)

# load label file if it exists
# if os.path.isfile(image_path + lb_name):
#     lb_object = AICSImage(image_path + lb_name)
#     lb_data = np.squeeze(lb_object.data)
#     labels_layer = viewer.add_labels(lb_data, name='annotation')

# if __name__ == '__main__':
#     napari.run()