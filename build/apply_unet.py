import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from functions.core_utils import Dataset, FishModel
from pprint import pprint
from torch.utils.data import DataLoader
from torchvision import transforms
import glob
import ntpath
import random
from aicsimageio import AICSImage
import skimage.transform as st


if __name__ == "__main__":
    model_name = 'unet_live_dead_0030'

    # Set path do data
    db_path = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphSeq\data\\built_keyence_data_v2\\"
    data_path = os.path.join(db_path, "UNET_training", '')

    seed_str = "1294_test_node"

    # make write paths
    figure_path = os.path.join(db_path, 'UNET_training', seed_str, model_name + '_predictions', '')

    if not os.path.isdir(figure_path):
        os.makedirs(figure_path)

    root = os.path.join(data_path, seed_str)

    # extract key info about computational resources
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_cpu = os.cpu_count()
    gpu_flag = device.type == 'cuda'
    # kwargs = {'num_workers': 1, 'pin_memory': True} if gpu_flag else {'num_workers': n_cpu, 'pin_memory': False}


    ttv_split = [0.8, 0.0, 0.2]
    im_dims = [576, 320]

    image_list = glob.glob(os.path.join(data_path, seed_str, 'images', '*.tif'))
    n_samples_total = len(image_list)

    im_list = []
    for n in range(n_samples_total):
        _, im_name = ntpath.split(image_list[n])
        im_list.append(im_name.replace('.tif', ''))


    # split into train, test, and validation sets
    n_train = n_samples_total #np.round(ttv_split[0] * n_samples_total).astype(int)
    # n_validate = n_samples_total - n_train

    np.random.seed(345)  # use random seed for consistency
    random_indices = np.random.choice(np.arange(n_samples_total), n_samples_total, replace=False)
    # train_files = [im_list[r] for r in random_indices[0:n_train]]
    valid_files = im_list#[im_list[r] for r in random_indices]#[n_train:]]

    # train_dataset = Dataset(root, train_files, im_dims, num_classes=2) #, transform=transforms)
    valid_dataset = Dataset(root, valid_files, im_dims, num_classes=2, predict_only_flag=True) #, transform=transforms)

    # train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=n_cpu)#**kwargs) # I think it is OK for Dataloader to use CPU workers
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)#**kwargs)

    model = FishModel("FPN",
                      "resnet34",
                      in_channels=3,
                      out_classes=2)

    n_devices = 1 if gpu_flag else n_cpu
    n_plot = len(valid_files)
    #############
    # load model
    model.load_state_dict(torch.load(os.path.join(root, model_name), map_location=device))
    model.eval()

    with torch.no_grad():
        data_list = list(valid_dataloader)
        for idx in range(len(data_list)):
            batch = data_list[idx]
            im = batch["image"]
            logits = model(im)
            pr_masks = np.squeeze(np.asarray(logits.sigmoid()))
            im_name = im_list[idx]
            # print(pr_masks.shape)
            # print(pr_masks)
            # print(pr_masks.shape)
            # print(im.shape)

            lb_predicted = np.zeros((pr_masks.shape[1], pr_masks.shape[2]))
            lb_predicted[np.where(pr_masks[0, :, :] >= 0.5)] = 1
            lb_predicted[np.where(pr_masks[1, :, :] >= 0.5)] = 2
            im_plot = np.squeeze(im.numpy()).transpose(1, 2, 0).astype(int)

            s = im_plot.shape

            plt.figure(figsize=(10, 5))
            #
            plt.subplot(1, 2, 1)
            plt.imshow(np.flipud(im_plot))

            plt.subplot(1, 2, 2)
            y, x = np.mgrid[0:s[0], 0:s[1]]
            # plt.axes().set_aspect('equal', 'datalim')
            plt.set_cmap(plt.gray())

            # plt.pcolormesh(x, y, Image2_mask, cmap='jet')
            plt.imshow(im_plot)
            plt.imshow(lb_predicted, cmap='viridis', alpha=0.3, vmin=0, vmax=2)
            # plt.axis([x.min(), x.max(), y.min(), y.max()])

            plt.xlim([x.min(), x.max()])
            plt.ylim([y.min(), y.max()])
            plt.colorbar(
                ticks=[0, 1, 2]
            )
            # plt.show()
            # plt.colorbar()
            plt.savefig(os.path.join(figure_path, im_name + f'_{idx:003}_prediction.tif'))
            plt.close()
    # for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pr_masks):
    #     plt.figure(figsize=(10, 5))
    #
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
    #     plt.title("Image")
    #     plt.axis("off")
    #
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(gt_mask.numpy().squeeze())  # just squeeze classes dim, because we have only one class
    #     plt.title("Ground truth")
    #     plt.axis("off")
    #
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(pr_mask.numpy().squeeze())  # just squeeze classes dim, because we have only one class
    #     plt.title("Prediction")
    #     plt.axis("off")
    #
    #     plt.show()