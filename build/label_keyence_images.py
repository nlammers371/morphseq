import numpy as np
import napari
import os
import glob2 as glob
import cv2
# from aicsimageio import AICSImage
# from PIL import Image


project_name = '20230525_bf_timeseries_stack1000_pitch040'
# db_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/built_data/"
db_path = "E:/Nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/built_data/"
n_im = 10
im_dims = [641, 1158]

# set random seed for reproducibility
seed = 124
np.random.seed(seed)

# make write path
out_path = os.path.join(db_path, 'RF_training', project_name, str(seed), '')
if not os.path.isdir(out_path):
    os.makedirs(out_path)

# set path to imags
path_to_images = os.path.join(db_path, 'stitched_FF_images', project_name, '')

# select subset of images to label
im_list = glob.glob(path_to_images + '*.tif')
im_lb_indices = np.random.choice(range(len(im_list)), n_im, replace=False)

images = []
for n in range(n_im):
    im_path = im_list[im_lb_indices[n]]
    im_temp = cv2.imread(im_path)
    im_rs = cv2.resize(im_temp, im_dims)
    im_name = im_path.replace(path_to_images, '')
    cv2.imwrite(out_path + im_name, im_rs)

    images.append(im_rs[:, :, 0])

image_array = np.asarray(images)

# # imData = np.squeeze(imObject.data)
# res_raw = imObject.physical_pixel_sizes
# res_array = np.asarray(res_raw)
#
# lbObject = AICSImage(readPathLabels)
# label_data = lbObject.data
# #
# # # with open("/Users/nick/RNA300_GFP_10x_wholeEmbryo.npy", 'wb') as f:
# # # np.save("/Users/nick/RNA300_GFP_10x_wholeEmbryo.npy", imData)
# #
i = 0
# check to see if we have label file for this image
im_path = im_list[im_lb_indices[i]]
im_name = im_path.replace(path_to_images, '')
lb_name = 'labels_' + im_name

# initialize viewer
viewer = napari.view_image(image_array[0],  colormap="gray")
# load label file if it exists
# if os.path.isfile(out_path + lb_name):
#     lb_object = AICSImage(out_path + lb_name)
#     lb_data = np.squeeze(lb_object.data)
#     labels_layer = viewer.add_labels(lb_data, name='annotation')

if __name__ == '__main__':
    napari.run()