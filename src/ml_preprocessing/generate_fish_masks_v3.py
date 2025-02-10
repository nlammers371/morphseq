import numpy as np
import os
import glob2 as glob
import ntpath
import skimage.io as io
import napari
from src.functions.utilities import path_leaf


def label_embryo_images(root, label_type, overwrite_flag=False, start_i=None):


    path_to_images = os.path.join(root, "built_image_data", 'stitched_FF_images', '')
    date_folders = glob.glob(path_to_images + '/*')

    image_path = os.path.join(root, "built_image_data", "unet_training", "UNET_training_" + label_type, "training", 'images', '')
    label_path = os.path.join(root, "built_image_data", "unet_training", "UNET_training_" + label_type, "training", 'annotations', '')

    if not os.path.isdir(image_path):
        os.makedirs(image_path)
    if not os.path.isdir(label_path):
        os.makedirs(label_path)

    # get list of existing labels (if any)
    existing_label_paths = sorted(glob.glob(label_path + "*tif") + glob.glob(label_path + "*png") + glob.glob(label_path + "*jpg"))
    existing_label_names = []
    for ex in existing_label_paths:
        im_name = path_leaf(ex)
        existing_label_names.append(im_name)

    existing_image_paths = sorted(glob.glob(image_path + "*tif") + glob.glob(image_path + "*png") + glob.glob(image_path + "*jpg"))
    existing_image_names = []
    for ex in existing_image_paths:
        im_name = path_leaf(ex)
        existing_image_names.append(im_name[:18])

    # select subset of images to label
    image_path_list = []
    image_name_list = []
    for date_dir in date_folders:
        date = path_leaf(date_dir)
        im_list = glob.glob(os.path.join(date_dir, "") + "*tif") + glob.glob(os.path.join(date_dir, "") + "*png") + glob.glob(os.path.join(date_dir, "") + "*jpg")
        im_names = [date + "_" + path_leaf(im) for im in im_list]

        image_path_list += im_list
        image_name_list += im_names

    # image_name_list = [path_leaf(im_name) for im_name in image_path_list]
    keep_indices = [i for i in range(len(image_name_list)) if image_name_list[i][:18] not in existing_image_names]

    # randomize
    sample_indices = np.random.choice(keep_indices, len(keep_indices), replace=False)
    image_path_list = [image_path_list[i] for i in sample_indices]
    image_name_list = [image_name_list[i] for i in sample_indices]

    # check to see if there are any images in the training folder that do not yet have labels
    # if so, prioritize these
    label_stubs = [lb[:18] for lb in existing_label_names]
    priority_image_indices = [i for i in range(len(existing_image_names)) if existing_image_names[i][:18] not in label_stubs]
    other_image_indices = [i for i in range(len(existing_image_names)) if i not in priority_image_indices]

    priority_image_names = [existing_image_names[i] for i in priority_image_indices]
    other_image_names = [existing_image_names[i] for i in other_image_indices]

    priority_image_paths = [existing_image_paths[i] for i in priority_image_indices]
    other_image_paths = [existing_image_paths[i] for i in other_image_indices]

    # combine
    image_name_list = other_image_names + priority_image_names + image_name_list
    image_path_list = other_image_paths + priority_image_paths + image_path_list

    # temporary thing to increase prevalence of certain sets
    # keep_indices = [i for i in range(len(image_name_list)) if ("20240404" in image_name_list[i]) or ("20240411" in image_name_list[i])]
    # image_name_list = [image_name_list[i] for i in keep_indices]
    # image_path_list = [image_path_list[i] for i in keep_indices]

    if start_i is None:
        start_i = len(other_image_names)

    image_i = start_i

    # initialize viewer
    while image_i <= len(image_name_list)-1:

        # load image
        im_path = image_path_list[image_i]
        # im_name = image_name_list[image_i]
        im_name = path_leaf(im_path)
        # open labels if they exist (a
        #nd we want to keep them)
        # lb_path_full = label_path + prefix + '_' + im_name
        # if (lb_path_full not in existing_label_paths) or (not skip_labeled_flag):

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

if __name__ == '__main__':

    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
    label_type = "mask"  # "via" "bubble" "focus"
    start_i = None

    label_embryo_images(root, label_type, overwrite_flag=False, start_i=start_i)


#     napari.run()