# NL: this is borrowed from a pre-existing pytorch repo for simclr:
# https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py

from torchvision.transforms import transforms
from functions.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from functions.view_generator import ContrastiveLearningViewGenerator
from functions.pythae_utils import MyCustomDataset
# from exceptions.exceptions import InvalidDatasetSelection


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform():#(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(brightness=0.3)
        data_transforms = transforms.Compose([#transforms.RandomResizedCrop(size=size, scale=tuple([0.5, 1]),  ratio=ratio),
                                              transforms.RandomAffine(degrees=15, scale=tuple([0.7, 1.3])),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              #transforms.RandomGrayscale(p=0.2),
                                              #GaussianBlur(kernel_size=5),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(),#(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(6),#(96),
                                                              n_views),
                                                          download=True),

                          'custom': lambda:  MyCustomDataset(root=self.root_folder,
                                                             transform=ContrastiveLearningViewGenerator(
                                                                 self.get_simclr_pipeline_transform(),#(96),
                                                                 n_views)
                                                             )}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise Exception("Invalid data selection")
        else:
            return dataset_fn()