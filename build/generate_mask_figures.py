import numpy as np
import napari
import os
import glob2 as glob
import cv2
from matplotlib import pyplot as plt
from aicsimageio import AICSImage
import ntpath
# from PIL import Image
from qtpy.QtCore import QTimer
from matplotlib.cm import ScalarMappable
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# project_name = '20230525'
db_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/built_keyence_data_v2/"

# set random seed for reproducibility
seed = 126
np.random.seed(seed)

# make write paths
image_path = os.path.join(db_path, 'UNET_training', str(seed), 'images', '')
label_path = os.path.join(db_path, 'UNET_training', str(seed), 'annotations', '')
figure_path = os.path.join(db_path, 'UNET_training', str(seed), 'figures', '')

path_to_images = os.path.join(db_path, 'stitched_ff_images', '*')

if not os.path.isdir(figure_path):
    os.makedirs(figure_path)
# get list of existing labels (if any)
existing_labels = sorted(glob.glob(label_path + "*tif"))

# for e in range(len(existing_labels)):
#     lb_path = existing_labels[e]
#     _, im_name = ntpath.split(lb_path)
#     prefix = im_name[0:8]
#     suffix = im_name[9:]
#
#     im_path = os.path.join(path_to_images[:-2], prefix, suffix)
#     im_temp = cv2.imread(im_path)
#     cv2.imwrite(os.path.join(image_path, im_name), im_temp)
#
#
for e in range(len(existing_labels)):
    lb_path = existing_labels[e]
    _, im_name = ntpath.split(lb_path)
    im_path = image_path + im_name

    # lbObject = AICSImage(lb_path)
    lb_temp = cv2.imread(lb_path)
    # lb_temp[0, 0:10, :] = 100
    # imObject = AICSImage(im_path)
    im_temp = cv2.imread(im_path)

    s = np.asarray(im_temp[:, :, 0].shape)

    plt.figure(dpi=300)
    y, x = np.mgrid[0:s[0], 0:s[1]]
    # plt.axes().set_aspect('equal', 'datalim')
    plt.set_cmap(plt.gray())

    # plt.pcolormesh(x, y, Image2_mask, cmap='jet')
    plt.imshow(im_temp)
    plt.imshow(lb_temp[:, :, 0], cmap='viridis', alpha=0.3, vmin=0, vmax=2)
    # plt.axis([x.min(), x.max(), y.min(), y.max()])

    plt.xlim([x.min(), x.max()])
    plt.ylim([y.min(), y.max()])
    plt.colorbar(
        ticks=[0, 1, 2]
    )
    # plt.show()
    # plt.colorbar()
    plt.savefig(os.path.join(figure_path, im_name))
    plt.close()
    #

# load label file if it exists
# if os.path.isfile(image_path + lb_name):
#     lb_object = AICSImage(image_path + lb_name)
#     lb_data = np.squeeze(lb_object.data)
#     labels_layer = viewer.add_labels(lb_data, name='annotation')

# if __name__ == '__main__':
#     napari.run()