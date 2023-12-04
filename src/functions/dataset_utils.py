from pythae.data.datasets import DatasetOutput
from torchvision import datasets, transforms
import torch

# define transforms
# data_transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.ToTensor() # the data must be tensors
# ])

def make_dynamic_rs_transform():#im_dims):
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        # transforms.Resize((im_dims[0], im_dims[1])),
        transforms.ToTensor(),
    ])
    return data_transform

#########3
# Define a custom dataset class
class MyCustomDataset(datasets.ImageFolder):

    def __init__(self, root, return_name=False, transform=None, target_transform=None):
        self.return_name = return_name
        super().__init__(root=root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        X, Y = super().__getitem__(index)

        if not self.return_name:
            return DatasetOutput(
                data=X
            )
        else:
            # path_str = path_leaf(self.samples[index])
            # path_str = path_str[:-4]
            return DatasetOutput(data=X, label=self.samples[index])

# View generation class used for contrastive training
class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    # def __call__(self, x):
    #     return [self.base_transform(x) for i in range(self.n_views)]

    def __call__(self, x):
        temp_list = []
        for n in range(self.n_views):
            data_tr = self.base_transform(x)
            temp_list.append(torch.reshape(data_tr, (1, data_tr.shape[0], data_tr.shape[1], data_tr.shape[2])))

        return torch.cat(temp_list, dim=0)

# define custom class for contrastive data loading
class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform():#(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(brightness=0.3)
        data_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
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
                                                              self.get_simclr_pipeline_transform(),#(96),
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


