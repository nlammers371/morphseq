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


class BasicEvalDataset(datasets.ImageFolder):
    def __init__(self, root, experiments=None, return_name=False,
                 transform=None):
        self.return_name = return_name
        self.root = root
        self.transform = transform
        self.experiments = experiments  # e.g., ['date1', 'date2']

        # Let ImageFolder initialize full dataset
        super().__init__(root=root, transform=transform, target_transform=None)

        if experiments is not None:
            # Map class names to class indices
            allowed_indices = [self.class_to_idx[c] for c in experiments]

            # Filter samples and targets
            filtered_samples = [
                (path, label)
                for (path, label) in self.samples
                if label in allowed_indices
            ]
            self.samples = filtered_samples
            self.targets = [s[1] for s in self.samples]

    def __getitem__(self, index):
        X, _ = super().__getitem__(index)
        if self.transform:
            X = self.transform(X)

        if not self.return_name:
            return DatasetOutput(data=X)
        else:
            return DatasetOutput(data=X, label=self.samples[index], index=index)

class NTXentDataset(datasets.ImageFolder):

    def __init__(self, cfg, transform=None, target_transform=None,):

        from src.data.dataset_configs import NTXentDataConfig

        assert isinstance(cfg, NTXentDataConfig)
        self.cfg = cfg
        self.transform = transform
        self.target_transform = target_transform
        # self.target_transform = None

        super().__init__(root=cfg.image_path, transform=self.transform, target_transform=self.target_transform)


    def __getitem__(self, index):

        X = Image.open(self.samples[index][0])
        if self.transform:
            X = self.transform(X)

        # determine if we're in train or eval partition
        train_flag = index in self.cfg.train_indices
        if train_flag:
            group_bool_vec = self.cfg.train_bool
        else:
            group_bool_vec = self.cfg.eval_bool

        key_dict = self.cfg.seq_key_dict  # [self.cfg.mode]

        pert_id_vec = key_dict["pert_id_vec"]
        e_id_vec = key_dict["e_id_vec"]
        age_hpf_vec = key_dict["age_hpf_vec"]

        time_window = self.cfg.time_window
        self_target = self.cfg.self_target_prob
        # other_age_penalty = self.cfg.other_age_penalty

        #############3
        # Select sequential pair
        pert_id_input = pert_id_vec[index]
        e_id_input = e_id_vec[index]
        age_hpf_input = age_hpf_vec[index]

        # load metric array
        metric_array = self.cfg.metric_array
        pos_pert_ids = np.where(metric_array[pert_id_input, :] == 1)[0]
        # neg_pert_ids = np.where(metric_array[pert_id_input, :]==0)[0]

        pert_match_array = np.isin(pert_id_vec,
                                   pos_pert_ids)  # torch.tensor(np.isin(pert_id_vec, pos_pert_ids)).type(torch.bool)

        e_match_array = e_id_vec == e_id_input
        age_delta_array = np.abs(age_hpf_vec - age_hpf_input)
        age_match_array = age_delta_array <= time_window

        # positive options
        self_option_array = e_match_array & age_match_array & group_bool_vec
        other_option_array = (~e_match_array) & age_match_array & pert_match_array & group_bool_vec

        if (np.random.rand() <= self_target) or (np.sum(other_option_array) == 0):
            options = np.nonzero(self_option_array)[0]
            seq_pair_index = np.random.choice(options, 1, replace=False)[0]
            # weight_hpf = age_delta_array[seq_pair_index] + 1
        else:
            options = np.nonzero(other_option_array)[0]
            seq_pair_index = np.random.choice(options, 1, replace=False)[0]
            # weight_hpf = age_delta_array[seq_pair_index] + 1 + other_age_penalty

        #########
        # load positive comparison
        Y = Image.open(self.samples[seq_pair_index][0])
        if self.transform:
            Y = self.transform(Y)

        X = torch.reshape(X, (1, X.shape[0], X.shape[1], X.shape[2]))
        Y = torch.reshape(Y, (1, Y.shape[0], Y.shape[1], Y.shape[2]))
        XY = torch.cat([X, Y], axis=0)

        # weight_hpf = torch.ones(weight_hpf.shape)  # ignore age-based weighting for now

        return DatasetOutput(data=XY, label=[self.samples[index][0], seq_pair_index], index=[index, seq_pair_index],
                             # weight_hpf=weight_hpf,
                             self_stats=[e_id_input, age_hpf_input, pert_id_input],
                             other_stats=[e_id_vec[seq_pair_index], age_hpf_vec[seq_pair_index],
                                          pert_id_vec[seq_pair_index]])
