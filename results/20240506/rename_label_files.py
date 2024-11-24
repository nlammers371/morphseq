import os
import torch
import numpy as np
from src.functions.core_utils_segmentation import Dataset, FishModel
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import glob
import ntpath
from tqdm import tqdm
from src.functions.utilities import path_leaf
import skimage.io as io

def rename_label_files(root, model_name, segment_list=None):


    """
    :param root:
    :param model_name:
    :param n_classes:
    :param overwrite_flag:
    :param im_dims:
    :param batch_size:
    :param n_workers:
    :return:
    """

    # generate directory for model predictions
    path_to_labels = os.path.join(root, 'segmentation', model_name + '_predictions', '')


    # get list of images to classify
    path_to_images = os.path.join(root, 'stitched_FF_images', '*')
    if segment_list is None:
        project_list = sorted(glob.glob(path_to_images))
        project_list = [p for p in project_list if "ignore" not in p]
        project_list = [p for p in project_list if os.path.isdir(p)]
    else:
        project_list = [os.path.join(root, 'stitched_FF_images', p) for p in segment_list]

    # select subset of images to label
    # image_path_list = []
    # label_path_list = []
    # exist_flags = []
    for ind, p in enumerate(tqdm(project_list, "Renaming label files...")):
        im_list_temp = glob.glob(os.path.join(p, '*.png')) + glob.glob(os.path.join(p, '*.tif')) + glob.glob(os.path.join(p, '*.jpg'))
        # image_path_list += im_list_temp

        _, project_name = ntpath.split(p)
        label_path_root = os.path.join(path_to_labels, project_name)
        if not os.path.isdir(label_path_root):
            os.makedirs(label_path_root)

        for imp in im_list_temp:
            _, tail = ntpath.split(imp)
            label_path = os.path.join(label_path_root, tail)
            label_path = label_path + ".jpg"
            label_path_new = label_path.replace(".png", "")#.replace(".png", ".jpg")
            label_path_new = label_path_new.replace(".jpg.jpg", ".jpg")
            if os.path.isfile(label_path):
                os.rename(label_path, label_path_new)

if __name__ == '__main__':
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/built_image_data/"

    rename_label_files(root, model_name="unet_yolk_v1_0050")
    rename_label_files(root, model_name="unet_emb_v5_0025")