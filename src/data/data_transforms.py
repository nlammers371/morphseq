from pythae.data.datasets import DatasetOutput
from torchvision import datasets, transforms
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from pythae.data.datasets import collate_dataset_output

def make_dynamic_rs_transform():#im_dims):
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        # transforms.Resize((im_dims[0], im_dims[1])),
        transforms.ToTensor(),
    ])
    return data_transform