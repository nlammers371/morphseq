# script to define functions for loading and standardizing fish images
import os
from PIL import Image
import glob2 as glob

imageDir = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphMap/data/gap16_bf_images_clean/"
# get list of files
folderList = os.listdir(imageDir)
# filter
#folderList = [s for s in folderList if "Export" in s]
# load images in each subdirectory
for f in range(1):
    subDir = os.path.join(imageDir,folderList[f])
    print(subDir)
    imList = os.listdir(subDir)
    imList = [s for s in imList if "tif" in s]
    print(imList)
    for i in range(1):#range(len(im_list)):
        im_path = os.path.join(subDir,imList[i])
        print(im_path)
        im = Image.open(im_path)
        im.show()