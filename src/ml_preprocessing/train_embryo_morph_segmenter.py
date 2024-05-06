# this script abd supporting function draw heavily from:
import os
import torch
import numpy as np
import pytorch_lightning as pl
from src.functions.core_utils_segmentation import Dataset, FishModel
from src.functions.utilities import path_leaf
from pprint import pprint
from torch.utils.data import DataLoader
import glob
import ntpath
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers


def train_unet_classifier(image_root, model_type, model_name, seed_str, n_epoch=50, pretrained_model=None):

    if model_type=="emb":
        n_classes=2
    else:
        n_classes=1

    data_path = os.path.join(image_root, "unet_training", "UNET_training_" + model_type)

    root = os.path.join(data_path, seed_str)

    # extract key info about computational resources
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_cpu = os.cpu_count()
    gpu_flag = device.type == 'cuda'

    ttv_split = [0.85, 0.0, 0.15]
    im_dims = [576, 320]

    image_list = sorted(glob.glob(os.path.join(data_path, seed_str, 'images', '*.tif')) +
                        glob.glob(os.path.join(data_path, seed_str, 'images', '*.png')) +
                        glob.glob(os.path.join(data_path, seed_str, 'images', '*.jpg')))

    mask_list = sorted(glob.glob(os.path.join(data_path, seed_str, 'annotations', '*.tif')) +
                       glob.glob(os.path.join(data_path, seed_str, 'annotations', '*.png')) +
                       glob.glob(os.path.join(data_path, seed_str, 'annotations', '*.jpg')))
    n_samples_total = len(mask_list)

    im_list = []
    for n in range(n_samples_total):
        im_name = path_leaf(image_list[n])
        im_list.append(im_name)

    # split into train, test, and validation sets
    n_train = np.round(ttv_split[0] * n_samples_total).astype(int)
    n_test = np.round(ttv_split[1] * n_samples_total).astype(int)
    n_validate = n_samples_total - n_train - n_test

    # np.random.seed(345)
    random_indices = np.random.choice(np.arange(n_samples_total), n_samples_total, replace=False)
    train_files = [im_list[r] for r in random_indices[0:n_train]]
    # test_files = [im_list[r] for r in random_indices[n_train:n_train + n_test]]
    valid_files = [im_list[r] for r in random_indices[n_train + n_test:]]

    train_dataset = Dataset(root, train_files, im_dims, num_classes=n_classes)  # , transform=transforms)
    # test_dataset = Dataset(root, test_files, im_dims)
    valid_dataset = Dataset(root, valid_files, im_dims, num_classes=n_classes)  # , transform=transforms)

    # It is a good practice to check datasets don`t intersects with each other
    assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    # print(f"Test size: {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                  num_workers=n_cpu)  # **kwargs) # I think it is OK for Dataloader to use CPU workers
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=n_cpu)  # **kwargs)
    # test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=n_cpu)

    model = FishModel("FPN",
                      "resnet34",
                      in_channels=3,
                      out_classes=n_classes)
    # classes=['live', 'dead'])

    if pretrained_model is not None:
        model.load_state_dict(
            torch.load(pretrained_model, map_location=device))

    # n_devices = 1 if gpu_flag else n_cpu

    # instrcut pytorch to track model performance and save the best-performing versions
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(data_path, seed_str, model_name + 'checkpoints', ''),
                                          save_top_k=5,
                                          monitor='valid_per_image_iou',
                                          mode='max')

    # initialize custom logger so I can control where the (rather large) training logs are stored
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(data_path, seed_str, model_name + "logs", ''))
    trainer = pl.Trainer(
        gpus=1,
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

if __name__ == "__main__":


    n_epoch = 50
    model_name = 'unet_emb_v5_'

    # option to load previously trained model
    pretrained_model = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/built_image_data/segmentation_models/20240507/unet_emb_v40050"
    n_epochs = 25
    # Set path to data
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/built_image_data/"

    seed_str = str(932)


