# script to define functions for loading and standardizing fish movies
import os
import numpy as np
from PIL import Image
from skimage import (exposure, feature, filters, io, measure,
                      morphology, restoration, segmentation, transform,
                      util)
import matplotlib
import cv2

from tqdm import trange
import glob
import tifffile
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
io.use_plugin('matplotlib')

import glob2 as glob

# this could be useful: https://scikit-image.org/skimage-tutorials/lectures/three_dimensional_image_processing.html

def build_ff_movies(im_dims, channelIncludeVec, dateString, readFolderName, overwriteFlag, rootDir, embList = []):

    # handle paths
    writeFolderName = "in_focus_movies/"
    imageRootDir = rootDir + dateString + "/"
    imageReadDir = imageRootDir + readFolderName
    imageWriteDir = imageRootDir + writeFolderName

    # get list of files. Each should be a folder containing imaging data for a single embryo
    #folderList = sorted(os.listdir(imageReadDir))
    FPath = imageReadDir + "W*"
    folderList = sorted(glob.glob(imageReadDir + "W*"))
    iterList = range(len(folderList))
    if len(embList)!=0:
        iterList = embList

    for f in iterList:
        # next layer is always the same
        subdir = 'P00001'
        subPath = os.path.join(imageReadDir,folderList[f],subdir)

        # get list of sub(sub)folders (should be time points)
        if os.path.isdir(subPath):
            timePointFolders = sorted(os.listdir(subPath))
            outDir = os.path.join(imageWriteDir,folderList[f])

            if (os.path.isdir(outDir)!=True) | (overwriteFlag==True):

                if (os.path.isdir(outDir) != True):
                    os.makedirs(outDir)

                for t in range(len(timePointFolders)):

                    # get lit of image files
                    for c in range(len(channelIncludeVec)):
                        chString = "CH" + str(c+1)
                        imPath = os.path.join(subPath,timePointFolders[t])

                        if os.path.isdir(imPath):
                            imageList = sorted(glob.glob(imPath + "/*" + chString + "*"))
                            #print(imPath + "*" + chString + "*")
                            im_dims_temp = im_dims
                            im_dims_temp.append(len(imageList))
                            imArray = np.zeros(im_dims_temp)
                            lp_array = np.zeros(im_dims_temp)
                            focusList = []

                            for i in range(len(imageList)):
                                im = Image.open(os.path.join(imPath,imageList[i])).convert('L')
                                imArray[:,:,i] = im
                                #viewer = napari.view_image(im)
                                fm_full = cv2.Laplacian(np.asarray(im), cv2.CV_64F)
                                #fm_sm = cv2.GaussianBlur(fm_full,  (10, 10), cv2.BORDER_DEFAULT)

                                #lp_array[:,:,i] = fm_sm

                                fm = fm_full.var()
                                focusList.append(fm)

                            # find in-focus frame
                            i_focus = np.argmax(focusList)
                            #argmax_indices = np.argmax(lp_array, axis=2)
                            i_focus_vec = [i_focus]
                            if (i_focus > 0) & (i_focus < len(imageList)-1):
                                i_focus_vec = [i_focus-1, i_focus, i_focus+1]
                            elif (i_focus > 0):
                                i_focus_vec = [i_focus - 1, i_focus]
                            else:
                                i_focus_vec = [i_focus, i_focus + 1]

                            imFocus = Image.fromarray(np.mean(imArray[:,:,i_focus_vec],axis=2))#Image.fromarray(imArray[:,:,6],'L')
                            # save
                            tempName = 'emb_' + folderList[f][-4:] + '_t' + str(t).zfill(4) + '_' + chString + '.tif'
                            imName = os.path.join(outDir,tempName)

                            imFocus.save(imName)
                            im_dims = im_dims[:-1]
                            #imFocus.show()


if __name__ == "__main__":
    readFolderName = "timelapse_test_lm_agarose_uplate/"
    im_dims = [480, 640]
    overwriteFlag = True
    channelIncludeVec = [1]  # ,2,3]
    dateString = "2022.11.01"
    rootDir = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphMap/data/"

    build_ff_movies(im_dims, channelIncludeVec, dateString, readFolderName, overwriteFlag, rootDir, embList=[0])