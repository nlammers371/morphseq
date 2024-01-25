import os
import torch
import numpy as np
from src.functions.core_utils_segmentation import Dataset, FishModel
from torch.utils.data import DataLoader
import glob
import ntpath
from tqdm import tqdm
import skimage.io as io

def apply_unet(root, model_name, n_classes, overwrite_flag=False, im_dims=None, batch_size=64, n_workers=4):

    # extract key info about computational resources
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # n_cpu = os.cpu_count()
    # gpu_flag = device.type == 'cuda'

    np.random.seed(345)  # use random seed for consistency
    if im_dims == None:
        im_dims = [576, 320]  # NL: can I read this off of the loaded model object?

    # generate directory for model predictions
    path_to_labels = os.path.join(root, 'segmentation', model_name + '_predictions', '')

    # get list of images to classify
    path_to_images = os.path.join(root, 'stitched_ff_images', '*')
    project_list = glob.glob(path_to_images)
    project_list = [p for p in project_list if "ignore" not in p]

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
        label_path_list = [label_path_list[e] for e in range(len(label_path_list)) if not exist_flags[e]]
        n_ex = np.sum(np.asarray(exist_flags) == 1)
        if n_ex > 0:
            print('Skipping ' + str(n_ex) + " previously segmented images. Set 'overwrite_flag=True' to overwrite")

    # generate data loader object
    im_dataset = Dataset(root, image_path_list, im_dims, num_classes=n_classes, predict_only_flag=True)
    im_dataloader = DataLoader(im_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=n_workers)

    # initialize instance of model
    model = FishModel("FPN",
                      "resnet34",
                      in_channels=3,
                      out_classes=n_classes)

    # load trained model weights
    model.load_state_dict(torch.load(os.path.join(root, 'segmentation', model_name), map_location=device))
    model.eval()

    # get predictions
    print("Classifying images...")
    with torch.no_grad():
        # data_list = list(im_dataloader)
        it = iter(im_dataloader)
        for idx in tqdm(range(len(im_dataloader))):
            batch = next(it)
            im = batch["image"]
            # print('im')
            logits = model(im)
            # print('logits')
            pr_masks = np.squeeze(np.asarray(logits.sigmoid()))

            if n_classes==1:
                pr_masks = pr_masks[np.newaxis, :, :]

            lb_predicted = np.ones((pr_masks.shape[1], pr_masks.shape[2]))
            for c in range(n_classes):
                lb_predicted[np.where(pr_masks[c, :, :] >= 0.5)] = c+2
            lb_predicted = lb_predicted / (n_classes+1) * 255
            lb_predicted = lb_predicted.astype(np.uint8)  # convert to integer

            # write to file
            io.imsave(label_path_list[idx], lb_predicted)



if __name__ == "__main__":
    root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\built_keyence_data"

    # first, apply classifier to identify living and dead embryos, as well bubbles
    n_classes = 2
    model_name = "unet_emb_v4_0050"
    apply_unet(root, model_name, n_classes)

    # now apply bubble classifier
    n_classes = 1
    model_name = "unet_bubble_v0_0050"
    apply_unet(root, model_name, n_classes)

    # now apply yolk classifier
    n_classes = 1
    model_name = "unet_yolk_v0_0050"
    apply_unet(root, model_name, n_classes)

    # now apply classifier to flag out-of-focus embryos
    n_classes = 1
    model_name = "unet_focus_v2_0050"
    apply_unet(root, model_name, n_classes)
