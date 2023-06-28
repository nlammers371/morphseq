import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from functions.core_utils import Dataset, FishModel
from torch.utils.data import DataLoader
import glob
import ntpath
import cv2
from tqdm import tqdm

def apply_unet(root, model_name, n_classes, write_path, overwrite_flag=False):
    # extract key info about computational resources
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_cpu = os.cpu_count()
    gpu_flag = device.type == 'cuda'

    np.random.seed(345)  # use random seed for consistency

    im_dims = [576, 320] # NL: can I read this off of the loaded model object?

    # generate directory for model predictions
    path_to_labels = os.path.join(root, model_name + '_predictions', '')

    # get list of images to classify
    path_to_images = os.path.join(root, 'stitched_ff_images', '*')
    project_list = glob.glob(path_to_images)

    # select subset of images to label
    image_path_list = []
    label_path_list = []
    exist_flags = []
    for ind, p in enumerate(project_list):
        im_list_temp = glob.glob(os.path.join(p, '*.tif'))
        image_path_list += im_list_temp

        _, project_name = ntpath.split(p)
        label_path_root = os.path.join(path_to_labels, project_name)
        if not os.path.isdir(label_path_root):
            os.makedirs(label_path_root)

        for imp in im_list_temp:
            _, tail = ntpath.split(imp)
            label_path = os.path.join(label_path_root, tail)
            label_path_list.append(label_path)
            exist_flags.append(os.path.isfile(label_path))

    # remove images with previously existing labels if overwrite_flag=False
    if not overwrite_flag:
        image_path_list = [image_path_list[e] for e in range(len(image_path_list)) if not exist_flags[e]]
        label_path_list = [label_path_list[e] for e in range(len(image_path_list)) if not exist_flags[e]]
        n_ex = np.sum(np.asarray(exist_flags) == 1)
        if n_ex > 0:
            print('Skipping ' + str(n_ex) + ' previously segmented images. Set overwrite_flag=True to overwrite')

    # generate data loader object
    im_dataset = Dataset(root, image_path_list, im_dims, num_classes=n_classes, predict_only_flag=True)
    im_dataloader = DataLoader(im_dataset, batch_size=1, shuffle=False)

    # initialize instance of model
    model = FishModel("FPN",
                      "resnet34",
                      in_channels=3,
                      out_classes=n_classes)

    # load trained model weights
    model.load_state_dict(torch.load(os.path.join(root, model_name), map_location=device))
    model.eval()

    # get predictions
    with torch.no_grad():
        data_list = list(im_dataloader)
        print("Classifying images...")
        for idx in tqdm(range(len(data_list))):
            batch = data_list[idx]
            im = batch["image"]
            logits = model(im)
            pr_masks = np.squeeze(np.asarray(logits.sigmoid()))

            lb_predicted = np.zeros((pr_masks.shape[1], pr_masks.shape[2]))
            for c in range(n_classes):
                lb_predicted[np.where(pr_masks[c, :, :] >= 0.5)] = c+1

            lb_predicted.astype(np.uint8) # convert to integer

            # write to file
            cv2.imwrite(label_path_list[idx], lb_predicted)


if __name__ == "__main__":
    root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\built_keyence_data"
    model_name = "unet_yh_v2_0003"  #'unet_ldb_v4_0050'  # 'unet_live_dead_0030'
    n_classes = 3
    pd_only_flag = False
    type_string = "morph_UNET_training"
    # Set path do data
    db_path = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\built_keyence_data" #"D:\\Nick\\morphseq\\built_keyence_data\\"
    # data_path = os.path.join(db_path, "UNET_training", '')
    data_path = os.path.join(db_path, type_string, '')
