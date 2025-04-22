from pythae.data.datasets import DatasetOutput
from torchvision import datasets, transforms
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from pythae.data.datasets import collate_dataset_output


class BasicDataset(datasets.ImageFolder):

    def __init__(self, root, return_name=False, transform=None, target_transform=None,
                 load_batch_size=128):
        self.return_name = return_name
        # self.train_config = training_config
        self.root = root
        # self.use_cache = training_config.cache_data
        super().__init__(root=root, transform=transform, target_transform=target_transform)


    def __getitem__(self, index):
        # if not self.use_cache:
        X, _ = super().__getitem__(index)

        if not self.return_name:
            return DatasetOutput(
                data=X
            )
        else:
            return DatasetOutput(data=X, label=self.samples[index], index=index)

