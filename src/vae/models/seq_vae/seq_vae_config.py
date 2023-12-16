from pydantic.dataclasses import dataclass
from ..vae import VAEConfig
import pandas as pd
from src.build.make_training_key import make_seq_key, get_sequential_pairs
import os
import numpy as np

@dataclass
class SeqVAEConfig(VAEConfig):
    """
    MetricVAE model config config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        temperature (float): Parameter dictating the temperature used in NT-Xent loss function. Default: 1
        zn_frac (float): fraction of latent dimensions to use for capturing nuisance variability
        orth_flag (bool): indicates whether or not to impose orthogonality constraint on latent dimensions
        gamma (float): weight factor that controls weight of orthogonality cost relative to rest of loss function
    """

    temperature: float = 1.0
    zn_frac: float = 0.1
    orth_flag: bool = True
    n_conv_layers: int = 5  # number of convolutional layers
    n_out_channels: int = 16  # number of layers to convolutional kernel
    distance_metric: str = "euclidean"
    beta: float = 1.0  # tunes the weight of the Gaussian prior term
    name: str = "SeqVAEConfig"
    data_root: str = ''
    train_folder: str = ''

    # set sequential hyperparameters
    time_window: float = 1.5  # max permitted age difference between sequential pairs
    self_target_prob: float = 0.5  # fraction of time to load self-pair vs. alternative comparison
    other_age_penalty: float = 1.0  # added similarity delta for cross-embryo comparisons

    def __init__(self,
                 data_root=None,
                 train_folder=None,
                 input_dim=(1, 288, 128),
                 latent_dim=100,
                 temperature=1.0,
                 zn_frac=0.2,
                 orth_flag=True,
                 beta=1.0,
                 n_conv_layers=5,  # number of convolutional layers
                 n_out_channels=16,  # number of layers to convolutional kernel
                 distance_metric="euclidean",
                 name="SeqVAEConfig",
                 uses_default_encoder=True, uses_default_decoder=True, reconstruction_loss='mse',
                 time_window=2.0, self_target_prob=0.5, other_age_penalty=2.0, **kwargs):

        self.__dict__.update(kwargs)

        self.uses_default_encoder = uses_default_encoder
        self.uses_default_decoder = uses_default_decoder
        self.reconstruction_loss = reconstruction_loss
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.temperature = temperature
        self.zn_frac = zn_frac
        self.orth_flag = orth_flag
        self.n_conv_layers = n_conv_layers
        self.n_out_channels = n_out_channels
        self.distance_metric = distance_metric
        self.name = name
        self.beta = beta
        self.data_root = data_root
        self.train_folder = train_folder
        self.time_window = time_window
        self.self_target_prob = self_target_prob
        self.other_age_penalty = other_age_penalty


    def make_dataset(self):
        """
        Load the dataset from the specified file path using pandas.
        """
        # get seq key
        seq_key = make_seq_key(self.data_root, self.train_folder)

        # use this to get dictionaries for valid pairs for each snip ID
        # seq_key_dict = get_sequential_pairs(seq_key, time_window=self.time_window,
        #                               self_target=self.self_target_prob,
        #                               other_age_penalty=self.other_age_penalty)

        self.seq_key = seq_key

        mode_vec = ["train", "eval", "test"]
        seq_key_dict = dict({})
        for m, mode in enumerate(mode_vec):
            seq_key = self.seq_key
            seq_key = seq_key.loc[seq_key["train_cat"] == mode]
            seq_key = seq_key.reset_index()

            pert_id_vec = seq_key["perturbation_id"].to_numpy()
            e_id_vec = seq_key["embryo_id_num"].to_numpy()
            age_hpf_vec = seq_key["predicted_stage_hpf"].to_numpy()

            dict_entry = dict({"pert_id_vec": pert_id_vec, "e_id_vec":e_id_vec, "age_hpf_vec": age_hpf_vec})
            seq_key_dict[mode] = dict_entry

        self.seq_key_dict = seq_key_dict
        # self.write_seq_pair_data()


    # def get_seq_pair_dict(self):
    #     mode_vec = ["train", "eval", "test"]
    #     batch_size = 128
    #     # self.seq_key_dict = seq_key_dict
    #     train_dir = os.path.join(self.data_root, self.train_folder, '')
    #
    #     for m, mode in enumerate(mode_vec):
    #         seq_key = self.seq_key
    #         seq_key = seq_key.loc[seq_key["train_cat"] == mode]
    #         seq_key = seq_key.reset_index()
    #
    #         time_window = self.time_window
    #         # self_target = self.self_target_prob
    #         other_age_penalty = self.other_age_penalty
    #
    #         pert_id_vec = seq_key["perturbation_id"].to_numpy()[:, np.newaxis]
    #         e_id_vec = seq_key["embryo_id_num"].to_numpy()[:, np.newaxis]
    #         age_hpf_vec = seq_key["predicted_stage_hpf"].to_numpy()[:, np.newaxis]
    #
    #         n_images = seq_key.shape[0]
    #         n_batches = np.ceil(n_images / batch_size)
    #         batch_start = 0
    #
    #         for n in range(n_batches):
    #
    #             if n < n_batches-1:
    #                 input_indices = np.arange(batch_start, batch_start+batch_size)
    #             else:
    #                 input_indices = np.arange(batch_start, n_images)
    #
    #             pert_id_input = pert_id_vec[input_indices]
    #             e_id_input = e_id_vec[input_indices]
    #             age_hpf_input = age_hpf_vec[input_indices]
    #
    #             pert_match_array = pert_id_vec == pert_id_input.T
    #             e_match_array = e_id_vec == e_id_input.T
    #             age_delta_array = age_hpf_vec - age_hpf_input.T
    #             age_match_array = np.abs(age_delta_array) <= time_window
    #
    #             self_option_array = e_match_array & age_match_array
    #             other_option_array = ~e_match_array & age_match_array & pert_match_array
    #
    #             indices_to_load = []
    #             pair_time_deltas = []
    #             for i, ind in enumerate(input_indices):
    #
    #                 out_dir_self = os.path.join(train_dir, mode, str(ind), "self")
    #                 if not os.path.isdir(out_dir_self):
    #                     os.makedirs(out_dir_self)
    #                 out_dir_other = os.path.join(train_dir, mode, str(ind), "other")
    #                 if not os.path.isdir(out_dir_other):
    #                     os.makedirs(out_dir_other)
    #
    #                 # if (np.random.rand() <= self_target) or (np.sum(other_option_array[:, i]) == 0):
    #                 self_options = np.nonzero(self_option_array[:, i])[0]
    #                 np.save(os.path.join(os.path.join(out_dir_self, "self_options.npy", self_options)))
    #                 np.save(os.path.join(os.path.join(out_dir_self, "self_weights.npy", age_delta_array[self_options, i].flatten())))
    #
    #                 # else:
    #                 other_options = np.nonzero(other_option_array[:, i])[0]
    #                 np.save(os.path.join(os.path.join(out_dir_other, "other_options.npy", other_options)))
    #                 np.save(os.path.join(
    #                     os.path.join(out_dir_other, "self_weights.npy", age_delta_array[other_options, i].flatten())))







