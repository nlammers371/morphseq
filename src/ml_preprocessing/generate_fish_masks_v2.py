import numpy as np
import napari
import os
import glob2 as glob
import skimage.io as io
from src.functions.utilities import path_leaf



yolk_label_flag = True
focus_label_flag = False
# root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/built_keyence_data_v2/"
root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/built_image_data/" #"E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\built_keyence_data\\" # "D:\\Nick\\morphseq\\built_keyence_data\\"   #
path_to_images = os.path.join(root, 'stitched_FF_images', '*')
project_list = sorted(glob.glob(path_to_images))

n_im = 1000
image_i = 0

# set starting point
# im_dims = [641, 1158]
overwrite_flag = False
skip_labeled_flag = False
if overwrite_flag:
    skip_labeled_flag = False

# set random seed for reproducibility
seed = 932 #126
suffix = "" #"_v2"
np.random.seed(seed)

# make write paths
if yolk_label_flag:
    image_path = os.path.join(root, "unet_training", 'UNET_training_yolk', str(seed) + suffix, 'images', '')
    label_path = os.path.join(root, "unet_training", 'UNET_training_yolk', str(seed) + suffix, 'annotations', '')
elif focus_label_flag:
    image_path = os.path.join(root, "unet_training", 'UNET_training_focus', str(seed) + suffix, 'images', '')
    label_path = os.path.join(root, "unet_training", 'UNET_training_focus', str(seed) + suffix, 'annotations', '')
else:
    image_path = os.path.join(root, "unet_training", 'UNET_training_emb', str(seed) + suffix, 'images', '')
    label_path = os.path.join(root, "unet_training", 'UNET_training_emb', str(seed) + suffix, 'annotations', '')


if not os.path.isdir(image_path):
    os.makedirs(image_path)
if not os.path.isdir(label_path):
    os.makedirs(label_path)

# get list of existing labels (if any)
existing_labels = sorted(glob.glob(label_path + "*tif") +glob.glob(label_path + "*png") + glob.glob(label_path + "*jpg"))
existing_label_names = []
for ex in existing_labels:
    im_name = path_leaf(ex)
    existing_label_names.append(im_name)

existing_images = sorted(glob.glob(image_path + "*tif") + glob.glob(image_path + "*png") + glob.glob(image_path + "*jpg"))
existing_image_names = []
for ex in existing_images:
    im_name = path_leaf(ex)
    existing_image_names.append(im_name)

# select subset of images to label
im_list = []
project_index = []
project_ind_long = []
for ind, p in enumerate(project_list):
    im_list_temp = glob.glob(os.path.join(p, '*tif')) + glob.glob(os.path.join(p, '*png')) + glob.glob(os.path.join(p, '*jpg'))
    im_list += im_list_temp
    tail = path_leaf(p)
    project_index.append(tail)
    project_ind_long += [ind]*len(im_list_temp)
im_lb_indices_raw = np.random.choice(range(len(im_list)), n_im, replace=False)

# ensure that existing labels are accounted for in the new list. They should always appear, but I swapped between
# random seed early in the labeling process, so there are some inconsistencies
im_name_list = []
for i in range(len(im_list)):
    im_name = path_leaf(im_list[i])
    prefix = project_index[project_ind_long[i]]
    # lb_path_full = label_path + prefix + '_' + im_name
    im_name_list.append(prefix + '_' + im_name)

# concatenate existing labels to randomly drawn list
# existing_indices = [im_name_list.index(ex) for ex in existing_label_names]
existing_im_indices = [im_name_list.index(ex) for ex in existing_image_names if ex not in existing_label_names]
full_list = existing_im_indices # + existing_indices + im_lb_indices_raw.tolist()
new_indices = np.unique(full_list, return_index=True)[1]
im_lb_indices = [full_list[index] for index in sorted(new_indices)]
im_lb_indices = im_lb_indices[0:n_im]

# initialize viewer
while image_i <= len(im_lb_indices)-1:
    prefix = project_index[project_ind_long[im_lb_indices[image_i]]]
    # load image
    im_path = im_list[im_lb_indices[image_i]]
    im_name = path_leaf(im_path)

    # open labels if they exist (and we want to keep them)
    lb_path_full = label_path + prefix + '_' + im_name
    if (lb_path_full not in existing_labels) or (not skip_labeled_flag):

        im_temp = io.imread(im_path)

        # open viewer
        viewer = napari.view_image(im_temp, colormap="gray")

        if not overwrite_flag and (lb_path_full in existing_labels):
            lb_temp = io.imread(lb_path_full)
            viewer.add_labels(lb_temp, name='Labels')
        # else:
        #     lbObject = AICSImage(os.path.join(root, 'well_mask.tif'))
        #     lb_temp = np.squeeze(lbObject.data)


        napari.run()

        # save new  label layer
        try:
            lb_layer = viewer.layers["Labels"]
        except:
            lb_layer = viewer.layers["SAM labels"]
        io.imsave(lb_path_full, lb_layer.data.astype(np.uint8), check_contrast=False)
        # save
        io.imsave(image_path + prefix + '_' + im_name, im_temp, check_contrast=False)
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