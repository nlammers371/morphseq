from pythae.data.datasets import DatasetOutput
from torchvision import datasets, transforms
import torch
from PIL import Image
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def load_split_train_test(datadir, model_type, model_config, training_config, valid_size=.2, train_frac=0.5):


    if model_type == "MetricVAE":

        # initialize contrastive data loader
        data_transform = ContrastiveLearningViewGenerator(
                                                 ContrastiveLearningDataset.get_simclr_pipeline_transform(), 2)

        # Make datasets
        image_data = MyCustomDataset(root=os.path.join(datadir),
                                        transform=data_transform,
                                        return_name=True
                                        )

        # eval_data = MyCustomDataset(root=os.path.join(datadir),
        #                                transform=data_transform,
        #                                return_name=True
        #                                )

    elif model_type == "VAE":
        # load standard VAE config

        # Standard data transform
        data_transform = make_dynamic_rs_transform()

        # Make datasets
        image_data = MyCustomDataset(root=os.path.join(datadir),
                                        transform=data_transform,
                                        return_name=True
                                        )

        # eval_data = MyCustomDataset(root=os.path.join(datadir),
        #                                transform=data_transform,
        #                                return_name=True
        #                                )

    elif model_type == "SeqVAE":

        # initialize contrastive data loader
        data_transform = ContrastiveLearningDataset.get_simclr_pipeline_transform()

        if model_config.metric_loss_type == "NT-Xent":
            # Make datasets
            image_data = SeqPairDataset(root=os.path.join(datadir),
                                           model_config=model_config,
                                           mode="train",
                                           transform=data_transform,
                                           return_name=True
                                            )

            # eval_data = SeqPairDataset(root=os.path.join(datadir),
            #                               model_config=model_config,
            #                               mode="eval",
            #                               transform=data_transform,
            #                               return_name=True
            #                                )
        elif model_config.metric_loss_type == "triplet":
            # Make datasets
            image_data = TripletPairDataset(root=os.path.join(datadir),
                                           model_config=model_config,
                                           mode="train",
                                           transform=data_transform,
                                           return_name=True
                                            )

            # eval_data = TripletPairDataset(root=os.path.join(datadir),
            #                               model_config=model_config,
            #                               mode="eval",
            #                               transform=data_transform,
            #                               return_name=True
            #                                )
    else:
        raise Exception("Unrecognized model type: " + model_type)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    train_idx, eval_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    eval_sampler = SubsetRandomSampler(eval_idx)

    trainloader = torch.utils.data.DataLoader(image_data, sampler=train_sampler, batch_size=training_config.per_device_train_batch_size)
    evalloader = torch.utils.data.DataLoader(image_data, sampler=eval_sampler, batch_size=training_config.per_device_eval_batch_size)

    return trainloader, evalloader


def set_inputs_to_device(input_tensor, device):

    inputs_on_device = input_tensor

    if device == "cuda":
        cuda_inputs = input_tensor

        # for key in inputs.keys():
        #     if torch.is_tensor(inputs[key]):
        #         cuda_inputs[key] = inputs[key].cuda()

        #     else:
        #         cuda_inputs[key] = inputs[key]
        cuda_inputs = input_tensor.cuda()
        inputs_on_device = cuda_inputs

    return inputs_on_device

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
            return DatasetOutput(data=X, label=self.samples[index], index=index)

class SeqPairDataset(datasets.ImageFolder):

    def __init__(self, root, model_config, mode, return_name=False, transform=None, target_transform=None):
        self.return_name = return_name
        self.model_config = model_config
        self.mode = mode
        super().__init__(root=root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        X = Image.open(self.samples[index][0])
        if self.transform:
            X = self.transform(X)


        key_dict = self.model_config.seq_key_dict[self.mode]

        pert_id_vec = key_dict["pert_id_vec"]
        e_id_vec = key_dict["e_id_vec"]
        age_hpf_vec = key_dict["age_hpf_vec"]

        time_window = self.model_config.time_window
        self_target = self.model_config.self_target_prob
        other_age_penalty = self.model_config.other_age_penalty

        #############3
        # Select sequential pair
        pert_id_input = pert_id_vec[index]
        e_id_input = e_id_vec[index]
        age_hpf_input = age_hpf_vec[index]

        pert_match_array = pert_id_vec == pert_id_input
        e_match_array = e_id_vec == e_id_input
        age_delta_array = np.abs(age_hpf_vec - age_hpf_input)
        age_match_array = age_delta_array <= time_window

        # positive options
        self_option_array = e_match_array & age_match_array
        other_option_array = ~e_match_array & age_match_array & pert_match_array

        if (np.random.rand() <= self_target) or (np.sum(other_option_array) == 0):
            options = np.nonzero(self_option_array)[0]
            seq_pair_index = np.random.choice(options, 1, replace=False)[0]
            weight_hpf = age_delta_array[seq_pair_index] + 1
        else:
            options = np.nonzero(other_option_array)[0]
            seq_pair_index = np.random.choice(options, 1, replace=False)[0]
            weight_hpf = age_delta_array[seq_pair_index] + 1 + other_age_penalty

        #########
        # load sequential pair
        Y = Image.open(self.samples[seq_pair_index][0])
        if self.transform:
            Y = self.transform(Y)

        X = torch.reshape(X, (1, X.shape[0], X.shape[1], X.shape[2]))
        Y = torch.reshape(Y, (1, Y.shape[0], Y.shape[1], Y.shape[2]))
        XY = torch.cat([X, Y], axis=0)

        weight_hpf = torch.ones(weight_hpf.shape)  # ignore age-based weighting for now
        # if not self.return_name:
        #     return DatasetOutput(
        #         data=X
        #     )
        # else:
        return DatasetOutput(data=XY, label=[self.samples[index][0], seq_pair_index], index=[index, seq_pair_index],
                             weight_hpf=weight_hpf,
                             self_stats=[e_id_input, age_hpf_input, pert_id_input],
                             other_stats=[e_id_vec[seq_pair_index], age_hpf_vec[seq_pair_index], pert_id_vec[seq_pair_index]])


class TripletPairDataset(datasets.ImageFolder):

    def __init__(self, root, model_config, mode, return_name=False, transform=None, target_transform=None):
        self.return_name = return_name
        self.model_config = model_config
        self.mode = mode
        super().__init__(root=root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        X = Image.open(self.samples[index][0])
        if self.transform:
            X = self.transform(X)


        # key_dict = self.model_config.seq_key_dict[self.mode]

        key_dict = self.model_config.seq_key_dict

        pert_id_vec = key_dict["pert_id_vec"]
        e_id_vec = key_dict["e_id_vec"]
        age_hpf_vec = key_dict["age_hpf_vec"]

        time_window = self.model_config.time_window
        self_target = self.model_config.self_target_prob
        other_age_penalty = self.model_config.other_age_penalty

        #############3
        # Select sequential pair
        pert_id_input = pert_id_vec[index]
        e_id_input = e_id_vec[index]
        age_hpf_input = age_hpf_vec[index]

        pert_match_array = pert_id_vec == pert_id_input
        e_match_array = e_id_vec == e_id_input
        age_delta_array = np.abs(age_hpf_vec - age_hpf_input)
        age_match_array = age_delta_array <= time_window

        # Select positive comparison
        self_option_array = e_match_array & age_match_array
        other_option_array = ~e_match_array & age_match_array & pert_match_array

        if (np.random.rand() <= self_target) or (np.sum(other_option_array) == 0):
            options = np.nonzero(self_option_array)[0]
            pos_pair_index = np.random.choice(options, 1, replace=False)[0]

        else:
            options = np.nonzero(other_option_array)[0]
            pos_pair_index = np.random.choice(options, 1, replace=False)[0]

        # Select negative comparison
        age_mismatch_array = age_delta_array >= (time_window + 1.5)
        negative_option_array = age_mismatch_array | (~pert_match_array)
        neg_options = np.nonzero(negative_option_array)[0]
        neg_pair_index = np.random.choice(neg_options, 1, replace=False)[0]

        #########
        # load positive and negative points
        YP = Image.open(self.samples[pos_pair_index][0])
        if self.transform:
            YP = self.transform(YP)

        YN = Image.open(self.samples[neg_pair_index][0])
        if self.transform:
            YN = self.transform(YN)

        X = torch.reshape(X, (1, X.shape[0], X.shape[1], X.shape[2]))
        YP = torch.reshape(YP, (1, YP.shape[0], YP.shape[1], YP.shape[2]))
        YN = torch.reshape(YN, (1, YN.shape[0], YN.shape[1], YN.shape[2]))
        XY = torch.cat([X, YP, YN], axis=0)

        # weight_hpf = torch.ones(weight_hpf.shape)  # ignore age-based weighting for now
        # if not self.return_name:
        #     return DatasetOutput(
        #         data=X
        #     )
        # else:
        return DatasetOutput(data=XY, label=[self.samples[index][0], self.samples[pos_pair_index][0], self.samples[neg_pair_index][0]],
                             index=[index, pos_pair_index, neg_pair_index])

# View generation class used for contrastive training
class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

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


