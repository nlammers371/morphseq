from pythae.data.datasets import DatasetOutput
from torchvision import datasets, transforms
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from pythae.data.datasets import collate_dataset_output


def contrastive_transform(target_size=None):  # (size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(brightness=0.3)
    data_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                          transforms.RandomAffine(degrees=15, scale=tuple([0.7, 1.3])),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          # transforms.RandomGrayscale(p=0.2),
                                          # GaussianBlur(kernel_size=5),
                                          transforms.ToTensor()])
    return data_transforms


def basic_transform(target_size=None):
    if target_size is not None:
        data_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((target_size[0], target_size[1])),
            transforms.ToTensor(),
        ])
    else:
        data_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
    return data_transform