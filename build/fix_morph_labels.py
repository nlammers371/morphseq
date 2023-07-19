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

# db_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/built_keyence_data_v2/"
db_path = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\built_keyence_data\\" # "D:\\Nick\\morphseq\\built_keyence_data\\"   #

# set random seed for reproducibility
# seed = 678
# suffix = "_focus"
seed = 932
suffix = "_yolk_head_tail"

image_path_in = os.path.join(db_path, 'morph_UNET_training', str(seed) + suffix, 'images', '')
label_path_in = os.path.join(db_path, 'morph_UNET_training', str(seed) + suffix, 'annotations', '')

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

transfer_name_list = [i for i in existing_image_names if i in existing_image_names]

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

    # make an embryo-only label version
    t_label_emb = t_label.copy()
    t_label_emb[np.where(t_label_emb > 2)] = 0
    if np.sum(t_label_emb) <= 30:  # zero out bubble labels
        t_label_emb[:] = 0
    # make a bubble-only label version
    t_label_bubble = t_label.copy()
    t_label_bubble[np.where(t_label_bubble != 4)] = 0  # zero out embryo_labels

    # write embryo image and labels to file
    cv2.imwrite(os.path.join(image_path_emb, t_name), t_image)
    cv2.imwrite(os.path.join(label_path_emb, t_name), t_label_emb)

    # write bubble
    if np.any(t_label_bubble):
        cv2.imwrite(os.path.join(image_path_bubble, t_name), t_image)
        cv2.imwrite(os.path.join(label_path_bubble, t_name), t_label_bubble)
    elif toggle:
        cv2.imwrite(os.path.join(image_path_bubble, t_name), t_image)
        cv2.imwrite(os.path.join(label_path_bubble, t_name), t_label_bubble)
        toggle = False
    else:
        toggle = True




# load label file if it exists
# if os.path.isfile(image_path + lb_name):
#     lb_object = AICSImage(image_path + lb_name)
#     lb_data = np.squeeze(lb_object.data)
#     labels_layer = viewer.add_labels(lb_data, name='annotation')

# if __name__ == '__main__':
#     napari.run()