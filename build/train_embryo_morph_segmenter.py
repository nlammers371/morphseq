# this script abd supporting function draw heavily from:
import os
import torch
import numpy as np
import pytorch_lightning as pl
from functions.core_utils import Dataset, FishModel
from pprint import pprint
from torch.utils.data import DataLoader
import glob
import ntpath
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers


if __name__ == "__main__":

    n_classes = 2
    n_epoch = 50
    model_name = 'unet_emb_v5_'
    # n_classes = 3
    # n_epoch = 5
    # model_name = 'unet_morph_v0_'

    # Set path do data
    # data_path = "D:\\Nick\morphseq\\built_keyence_data\\UNET_training\\"
    # data_path = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\built_keyence_data\\morph_UNET_training\\"
    data_path = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\built_keyence_data\\UNET_training_emb\\"
    seed_str = str(126) + "_v2"  # '_v2' # specify random seed that points to specific set of labeled training images
    root = os.path.join(data_path, seed_str)

    # extract key info about computational resources
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_cpu = os.cpu_count()
    gpu_flag = device.type == 'cuda'

    ttv_split = [0.85, 0.0, 0.15]
    im_dims = [576, 320]

    image_list = glob.glob(os.path.join(data_path, seed_str, 'images', '*.tif'))
    mask_list = glob.glob(os.path.join(data_path, seed_str, 'annotations', '*.tif'))
    n_samples_total = len(mask_list)

    im_list = []
    for n in range(n_samples_total):
        _, im_name = ntpath.split(image_list[n])
        im_list.append(im_name.replace('.tif', ''))

    # split into train, test, and validation sets
    n_train = np.round(ttv_split[0] * n_samples_total).astype(int)
    n_test = np.round(ttv_split[1] * n_samples_total).astype(int)
    n_validate = n_samples_total - n_train - n_test

    # np.random.seed(345)
    random_indices = np.random.choice(np.arange(n_samples_total), n_samples_total, replace=False)
    train_files = [im_list[r] for r in random_indices[0:n_train]]
    # test_files = [im_list[r] for r in random_indices[n_train:n_train + n_test]]
    valid_files = [im_list[r] for r in random_indices[n_train + n_test:]]

    train_dataset = Dataset(root, train_files, im_dims, num_classes=n_classes)#, transform=transforms)
    # test_dataset = Dataset(root, test_files, im_dims)
    valid_dataset = Dataset(root, valid_files, im_dims, num_classes=n_classes)#, transform=transforms)

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
                      out_classes=n_classes)
                      # classes=['live', 'dead'])

    n_devices = 1 if gpu_flag else n_cpu

    # instrcut pytorch to track model performance and save the best-performing versions
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(data_path, seed_str, model_name + 'checkpoints', ''),
                                          save_top_k=5,
                                          monitor='valid_per_image_iou',
                                          mode='max')

    # initialize custom logger so I can control where the (rather large) training logs are stored
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(data_path, seed_str, model_name + "logs", ''))
    trainer = pl.Trainer(
        gpus=1,
        # devices=n_devices,
        accelerator='auto',
        max_epochs=n_epoch,
        callbacks=[checkpoint_callback],
        logger=tb_logger
        # row_log_interval=1000 # NL I'm hoping this will reduce the size of the log files
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )

    # save
    torch.save(model.state_dict(), os.path.join(root, model_name + f'{n_epoch:04}'))

    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    pprint(valid_metrics)

    best_model_path = checkpoint_callback.best_model_path
    best_model_score = checkpoint_callback.best_model_score
    print(best_model_path)
    print(best_model_score)

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