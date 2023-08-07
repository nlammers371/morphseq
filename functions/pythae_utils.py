from pythae.data.datasets import DatasetOutput
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from torchvision.utils import _log_api_usage_once
import torch
from torch import Tensor

# define transforms
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor() # the data must be tensors
])




class MyCustomDataset(datasets.ImageFolder):

    def __init__(self, root, transform=None, target_transform=None, standardize=False):
        self.standardize = standardize
        super().__init__(root=root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        X, _ = super().__getitem__(index)
        X_out = X.clone()
        if self.standardize:

            for i in range(X.shape[0]):
                mean = torch.mean(X_out[i, :, :])
                std = torch.std(X_out[i, :, :])
                xt = (X_out[i, :, :] - mean) / std
                X_out[i, :, :] = xt

        return DatasetOutput(
            data=X_out
        )