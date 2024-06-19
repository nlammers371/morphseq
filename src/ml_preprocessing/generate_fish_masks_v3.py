import numpy as np
import os
import glob2 as glob
import ntpath
import skimage.io as io
import napari
from src.functions.utilities import path_leaf

root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
path_to_images = os.path.join(root, "built_image_data", 'stitched_FF_images', '')
project_list = glob.glob(path_to_images)
label_type = "focus"  # "via" "bubble" "focus"

image_path = os.path.join(root, "built_image_data", "unet_training", "UNET_training_" + label_type, "training", 'images', '')
label_path = os.path.join(root, "built_image_data", "unet_training", "UNET_training_" + label_type, "training", 'annotations', '')

if not os.path.isdir(image_path):
    os.makedirs(image_path)
if not os.path.isdir(label_path):
    os.makedirs(label_path)

# get list of existing labels (if any)
existing_labels = sorted(glob.glob(label_path + "*tif") + glob.glob(label_path + "*png") + glob.glob(label_path + "*jpg"))
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
image_path_list = glob.glob(path_to_images + "*tif") + glob.glob(path_to_images + "*png") + glob.glob(path_to_images + "*jpg")
image_name_list = [path_leaf(im_name) for im_name in image_path_list]
keep_indices = [i for i in range(len(image_name_list)) if image_name_list[i] not in existing_image_names]

# randomize
sample_indices = np.random.choice(keep_indices, len(keep_indices), replace=False)
image_path_list = [image_path_list[i] for i in sample_indices]
image_name_list = [image_name_list[i] for i in sample_indices]

image_i = 0

# initialize viewer
while image_i <= len(image_name_list)-1:

    # load image
    im_path = image_path_list[image_i]
    im_name = path_leaf(im_path)

    # open labels if they exist (a
    #nd we want to keep them)
    # lb_path_full = label_path + prefix + '_' + im_name
    # if (lb_path_full not in existing_labels) or (not skip_labeled_flag):

    im_temp = io.imread(im_path)
    # open viewer
    viewer = napari.view_image(im_temp, colormap="gray")
    viewer.window.add_plugin_dock_widget(plugin_name='napari-segment-anything')

    napari.run()

    # save new  label layer
    try:
        lb_layer = viewer.layers["Labels"]
    except:
        lb_layer = viewer.layers["SAM labels"]

    # save
    io.imsave(os.path.join(label_path, im_name), lb_layer.data.astype(np.uint8), check_contrast=False)
    io.imsave(os.path.join(image_path, im_name), im_temp, check_contrast=False)


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