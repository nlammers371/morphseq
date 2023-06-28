import numpy as np
import napari
import os
import glob2 as glob
import cv2
from aicsimageio import AICSImage
import ntpath
# from PIL import Image

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

morph_label_flag = True
# db_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/built_keyence_data_v2/"
db_path = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\built_keyence_data\\" # "D:\\Nick\\morphseq\\built_keyence_data\\"   #
path_to_images = os.path.join(db_path, 'stitched_ff_images', '*')
project_list = glob.glob(path_to_images)
#
n_im = 1000
image_i = 26

# set starting point
# im_dims = [641, 1158]
overwrite_flag = False
skip_labeled_flag = False
if overwrite_flag:
    skip_labeled_flag = False

# set random seed for reproducibility
seed = 932
suffix = "_yolk_head_tail"
np.random.seed(seed)

# make write paths
if not morph_label_flag:
    image_path = os.path.join(db_path, 'UNET_training', str(seed) + suffix, 'images', '')
    label_path = os.path.join(db_path, 'UNET_training', str(seed) + suffix, 'annotations', '')
else:
    image_path = os.path.join(db_path, 'morph_UNET_training', str(seed) + suffix, 'images', '')
    label_path = os.path.join(db_path, 'morph_UNET_training', str(seed) + suffix, 'annotations', '')

if not os.path.isdir(image_path):
    os.makedirs(image_path)
if not os.path.isdir(label_path):
    os.makedirs(label_path)

# get list of existing labels (if any)
existing_labels = sorted(glob.glob(label_path + "*tif"))
existing_label_names = []
for ex in existing_labels:
    _, im_name = ntpath.split(ex)
    existing_label_names.append(im_name)

existing_images = sorted(glob.glob(image_path + "*tif"))
existing_image_names = []
for ex in existing_images:
    _, im_name = ntpath.split(ex)
    existing_image_names.append(im_name)


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
im_lb_indices_raw = np.random.choice(range(len(im_list)), n_im, replace=False)

# ensure that existing labels are accounted for in the new list. They should always appear, but I swapped between
# random seed early in the labeling process, so there are some inconsistencies
im_name_list = []
for i in range(len(im_list)):
    _, im_name = ntpath.split(im_list[i])
    prefix = project_index[project_ind_long[i]]
    # lb_path_full = label_path + prefix + '_' + im_name
    im_name_list.append(prefix + '_' + im_name)

# concatenate existing labels to randomly drawn list
existing_indices = [im_name_list.index(ex) for ex in existing_label_names]
existing_im_indices = [im_name_list.index(ex) for ex in existing_image_names if ex not in existing_label_names]
full_list = existing_im_indices + existing_indices + im_lb_indices_raw.tolist()
new_indices = np.unique(full_list, return_index=True)[1]
im_lb_indices = [full_list[index] for index in sorted(new_indices)]
im_lb_indices = im_lb_indices[0:n_im]
# initialize viewer

while image_i < len(im_lb_indices)-1:
    prefix = project_index[project_ind_long[im_lb_indices[image_i]]]
    # load image
    im_path = im_list[im_lb_indices[image_i]]
    _, im_name = ntpath.split(im_path)

    # open labels if they exist (and we want to keep them)
    lb_path_full = label_path + prefix + '_' + im_name
    if (lb_path_full not in existing_labels) or (not skip_labeled_flag):

        im_temp = cv2.imread(im_path)

        # open viewer
        viewer = napari.view_image(im_temp[:, :, 0], colormap="gray")

        if not overwrite_flag and (lb_path_full in existing_labels):
            lbObject = AICSImage(lb_path_full)
            lb_temp = np.squeeze(lbObject.data)
            viewer.add_labels(lb_temp, name='Labels')
        # else:
        #     lbObject = AICSImage(os.path.join(db_path, 'well_mask.tif'))
        #     lb_temp = np.squeeze(lbObject.data)


        napari.run()

        # save new  label layer
        try:
            lb_layer = viewer.layers["Labels"]
        except:
            lb_layer = viewer.layers["SAM labels"]
        AICSImage(lb_layer.data.astype(np.uint8)).save(lb_path_full)
        # save
        cv2.imwrite(image_path + prefix + '_' + im_name, im_temp)
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

    else:
        image_i += 1


# load label file if it exists
# if os.path.isfile(image_path + lb_name):
#     lb_object = AICSImage(image_path + lb_name)
#     lb_data = np.squeeze(lb_object.data)
#     labels_layer = viewer.add_labels(lb_data, name='annotation')

# if __name__ == '__main__':
#     napari.run()