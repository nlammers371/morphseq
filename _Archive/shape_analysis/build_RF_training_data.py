from aicsimageio import AICSImage
import numpy as np
import glob2 as glob
import os

###########
# training images first
image_folder_train = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphMap/data/yx1_samples/segmentation/"
image_list_train1 = glob.glob(image_folder_train + "training_images_raw/*.ome.tif")
image_list_train2 = glob.glob(image_folder_train + "training_images_raw/*.nd2")
image_list_train = image_list_train1 + image_list_train2

training_dir = image_folder_train + "training_images_np_ds/"
if os.path.isdir(training_dir)==False:
    os.makedirs(training_dir)

for im in range(len(image_list_train)):
    im_path = image_list_train[im]
    folder_ind = im_path.find("_raw")
    dot_ind = im_path.find(".")
    im_name = im_path[folder_ind+5:dot_ind]

    # check that file does not already exist
    outName = training_dir + im_name + ".npy"
    if os.path.isfile(outName)==False:
        imObject = AICSImage(im_path)
        imData = np.squeeze(imObject.data)

        with open(training_dir + im_name + ".npy", 'wb') as f:
            np.save(f, imData)


####################
## Now test images

image_folder_test = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphMap/data/yx1_samples/segmentation/"
image_list_test1 = glob.glob(image_folder_test + "testing_images_raw/*.ome.tif")
image_list_test2 = glob.glob(image_folder_test + "testing_images_raw/*.nd2")
image_list_test = image_list_test1 + image_list_test2

testing_dir = image_folder_test + "testing_images_np_ds/"
if os.path.isdir(testing_dir)==False:
    os.makedirs(testing_dir)

for im in range(len(image_list_test)):
    im_path = image_list_test[im]
    folder_ind = im_path.find("_raw")
    dot_ind = im_path.find(".")
    im_name = im_path[folder_ind+5:dot_ind]

    # check that file does not already exist
    outName = testing_dir + im_name + ".npy"
    if os.path.isfile(outName)==False:
        imObject = AICSImage(im_path)
        imData = np.squeeze(imObject.data)

        with open(testing_dir + im_name + ".npy", 'wb') as f:
            np.save(f, imData)