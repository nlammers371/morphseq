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

from aicsimageio import AICSImage
import skimage.transform as st


if __name__ == "__main__":

    # Set path do data
    data_path = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphSeq\data\\built_keyence_data_v2\\UNET_training\\"
    seed_str = str(126)  # specify random seed that points to specific set of labeled training images
    suffix = "v2"
    root = os.path.join(data_path, seed_str)

    # extract key info about computational resources
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_cpu = os.cpu_count()
    gpu_flag = device.type == 'cuda'
    # kwargs = {'num_workers': 1, 'pin_memory': True} if gpu_flag else {'num_workers': n_cpu, 'pin_memory': False}

    # designate transformations to apply to augment data
    # transform = torch.nn.Sequential(
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip()
    # )


    ttv_split = [0.85, 0.0, 0.15]
    im_dims = [576, 320]

    image_list = glob.glob(os.path.join(data_path, seed_str + '_' + suffix, 'images', '*.tif'))
    mask_list = glob.glob(os.path.join(data_path, seed_str + '_' + suffix, 'annotations', '*.tif'))
    n_samples_total = len(mask_list)

    im_list = []
    for n in range(n_samples_total):
        _, im_name = ntpath.split(image_list[n])
        im_list.append(im_name.replace('.tif', ''))


    # split into train, test, and validation sets
    n_train = np.round(ttv_split[0] * n_samples_total).astype(int)
    n_test = np.round(ttv_split[1] * n_samples_total).astype(int)
    n_validate = n_samples_total - n_train - n_test

    np.random.seed(345)
    random_indices = np.random.choice(np.arange(n_samples_total), n_samples_total, replace=False)
    train_files = [im_list[r] for r in random_indices[0:n_train]]
    # test_files = [im_list[r] for r in random_indices[n_train:n_train + n_test]]
    valid_files = [im_list[r] for r in random_indices[n_train + n_test:]]

    train_dataset = Dataset(root, train_files, im_dims, num_classes=5)#, transform=transforms)
    # test_dataset = Dataset(root, test_files, im_dims)
    valid_dataset = Dataset(root, valid_files, im_dims, num_classes=5)#, transform=transforms)

    # It is a good practice to check datasets don`t intersects with each other
    assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    # print(f"Test size: {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=n_cpu)#**kwargs) # I think it is OK for Dataloader to use CPU workers
    valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=n_cpu)#**kwargs)
    # test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=n_cpu)

    model = FishModel("FPN",
                      "resnet34",
                      in_channels=3,
                      out_classes=2)
                      # classes=['live', 'dead'])

    n_devices = 1 if gpu_flag else n_cpu
    n_epoch = 30
    trainer = pl.Trainer(
        gpus=1,
        # devices=n_devices,
        accelerator='auto',
        max_epochs=n_epoch,
        row_log_interval=1000 # NL I'm hoping this will reduce the size of the log files
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )

    # save
    torch.save(model.state_dict(), os.path.join(root, 'unet_live_dead_' + f'{n_epoch:04}'))

    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    pprint(valid_metrics)

    # test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    # pprint(test_metrics)

    # batch = next(iter(valid_dataloader))
    # with torch.no_grad():
    #     model.eval()
    #     logits = model(batch["image"])
    # pr_masks = logits.sigmoid()
    # print(pr_masks.shape)
    #
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