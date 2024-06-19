import glob
import skimage.transform as st
import skimage
import skimage.io as io
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from src.functions.dataset_utils import set_inputs_to_device
import torch
import os
import torch
import shutil
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
from tqdm import tqdm
from aicsimageio import AICSImage
# from urllib3.request import urlretrieve

def load_split_train_test(datadir, valid_size=.2, batch_size=8):
    # train_transforms = transforms.Compose([transforms.Resize(224),
    #                                    transforms.ToTensor(),
    #                                    ])
    # test_transforms = transforms.Compose([transforms.Resize(224),
    #                                   transforms.ToTensor(),
    #                                   ])
    train_data = datasets.ImageFolder(datadir, transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                                             transforms.ToTensor()]))
    test_data = datasets.ImageFolder(datadir, transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                                            transforms.ToTensor()]))
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return trainloader, testloader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, filenames, out_dims, num_classes=1, predict_only_flag=False, mode="train", transform=None):

        # assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.out_dims = out_dims
        self.predict_only_flag = predict_only_flag
        self.num_classes = num_classes
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations")

        self.filenames = filenames   #self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        if ".tif" in filename:
            # filename = filename.replace(".tif", "")  # get rid of tif suffix if it exists
            image_path = os.path.join(self.images_directory, filename)
        elif ".png" in filename:
            # filename = filename.replace(".png", "")  # get rid of tif suffix if it exists
            image_path = os.path.join(self.images_directory, filename)
        elif ".jpg" in filename:
            # filename = filename.replace(".jpg", "")  # get rid of tif suffix if it exists
            image_path = os.path.join(self.images_directory, filename)
        else:
            raise Exception("File type not supported")

        stub_name = filename[:-4].replace("_stitch", "")
        stub_name = stub_name.replace("_ch01", "")
        mask_path = glob.glob(os.path.join(self.masks_directory, stub_name + "*"))

        if len(mask_path) > 1:
            raise Exception("Multiple masks found")
        mask_path = mask_path[0]
        # image = np.array(Image.open(image_path).convert("RGB"))

        # trimap = np.array(Image.open(mask_path))
        # mask = self._preprocess_mask(trimap)

        # load and resize mask
        if not self.predict_only_flag:
            lbObject = AICSImage(mask_path)
            lb_temp = np.squeeze(lbObject.data)
            if len(lb_temp.shape) == 3:
                lb_temp = lb_temp[:, :, 0]

            if False: #self.num_classes == 1:
                mask = st.resize(lb_temp, self.out_dims, order=0, preserve_range=True, anti_aliasing=False)
            else:
                mask_temp = st.resize(lb_temp, self.out_dims, order=0, preserve_range=True, anti_aliasing=False)
                mask = np.zeros((self.num_classes, self.out_dims[0], self.out_dims[1]))
                for c in range(self.num_classes):
                    mask[c, :, :] = (mask_temp == (c+1))*1.0
        else:
            if self.num_classes == 1:
                mask = np.zeros(self.out_dims)
            else:
                mask = np.zeros((self.num_classes, self.out_dims[0], self.out_dims[1]))

        # load and resize image
        im_temp = io.imread(image_path)
        if len(im_temp.shape) == 3:
            im_temp = im_temp[:, :, 0]

        # reformat to 8bit if image is natively 16 bit
        if im_temp.dtype != "uint8":
            im_temp = skimage.util.img_as_ubyte(im_temp)

        if im_temp.shape[0] < im_temp.shape[1]:
            im_temp = im_temp.transpose(1,0)
            
        image = st.resize(im_temp, self.out_dims, order=0, preserve_range=True, anti_aliasing=False)
        image = np.repeat(image[np.newaxis, :, :], 3, axis=0)

    
        # image = st.resize(im_temp, self.out_dims, order=0, preserve_range=True, anti_aliasing=False)
        # if False: #self.num_classes == 1:
        # sample = dict(image=image.astype(np.float32), mask=mask[np.newaxis, :, :].astype(np.float32))  #, trimap=trimap)
        # else:
        if np.any(np.isnan(mask)):
            raise Exception("wft")

        sample = dict(image=image.astype(np.float32), mask=mask.astype(np.float32), path=filename)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

class FishModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
    # @staticmethod
    # def _preprocess_mask(mask):
    #     mask = mask.astype(np.float32)
    #     mask[mask == 2.0] = 0.0
    #     mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    #     return mask

    # def _read_split(self):
    #     # split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
    #     # split_filepath = os.path.join(self.root, "annotations", split_filename)
    #     # with open(split_filepath) as f:
    #     #     split_data = f.read().strip("\n").split("\n")
    #     # filenames = [x.split(" ")[0] for x in split_data]
    #     filenames = glob.glob(os.path.join(self.images_directory, '*.tif'))
    #     if self.mode == "train":  # 90% for train
    #         filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
    #     elif self.mode == "valid":  # 10% for validation
    #         filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
    #     return filenames


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

# def download_url(url, filepath):
#     directory = os.path.dirname(os.path.abspath(filepath))
#     os.makedirs(directory, exist_ok=True)
#     if os.path.exists(filepath):
#         return

#     with TqdmUpTo(
#         unit="B",
#         unit_scale=True,
#         unit_divisor=1024,
#         miniters=1,
#         desc=os.path.basename(filepath),
#     ) as t:
#         urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
#         t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)