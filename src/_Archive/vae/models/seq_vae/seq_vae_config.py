from pydantic.dataclasses import dataclass
from src.vae import VAEConfig
import pandas as pd
from src.vae import make_seq_key, make_train_test_split
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

    temperature: float = 0.1
    metric_weight: float = 1.0  # tunes weight of contastive loss within the loss function
    margin: float = 1.0  # sets tolerance/scale for metric loss
    beta: float = 1.0  # tunes the weight of the Gaussian prior term
    zn_frac: float = 0.1

    orth_flag: bool = True
    n_conv_layers: int = 5  # number of convolutional layers
    n_out_channels: int = 16  # number of layers to convolutional kernel

    distance_metric: str = "euclidean"
    name: str = "SeqVAEConfig"
    metric_loss_type: str = "NT-Xent"
    data_root: str = ''
    train_folder: str = ''
    age_key_path: str = ''
    metric_key_path: str = ''
    pert_time_key_path: str = ''
    
    # set sequential hyperparameters
    time_window: float = 1.5  # max permitted age difference between sequential pairs
    self_target_prob: float = 0.5  # fraction of time to load self-pair vs. alternative comparison
    time_only_flag: float = 0  # binary flag. If 0, use only staging info for metric learning
    other_age_penalty: float = 1.0  # added similarity delta for cross-embryo comparisons (currently not used)

    def __init__(self,
                 data_root=None,
                 train_folder=None,
                 age_key_path=None,
                 pert_time_key_path=None,
                 metric_key_path=None,
                 train_indices=None,
                 eval_indices=None,
                 test_indices=None,
                 metric_loss_type="NT-Xent",
                 input_dim=(1, 288, 128),
                 latent_dim=100,
                 temperature=1.0,
                 zn_frac=0.2,
                 orth_flag=True,
                 beta=1.0,
                 margin=1.0,
                 metric_weight=1.0,
                 n_conv_layers=5,  # number of convolutional layers
                 n_out_channels=16,  # number of layers to convolutional kernel
                 distance_metric="euclidean",
                 name="SeqVAEConfig",
                 uses_default_encoder=True, uses_default_decoder=True, reconstruction_loss='mse',
                 time_window=2.0, self_target_prob=0.5, time_only_flag=0, other_age_penalty=2.0, **kwargs):

        self.__dict__.update(kwargs)

        self.uses_default_encoder = uses_default_encoder
        self.uses_default_decoder = uses_default_decoder
        self.reconstruction_loss = reconstruction_loss
        self.train_indices = train_indices
        self.eval_indices = eval_indices
        self.test_indices = test_indices
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.temperature = temperature
        self.margin = margin
        self.zn_frac = zn_frac
        self.orth_flag = orth_flag
        self.n_conv_layers = n_conv_layers
        self.n_out_channels = n_out_channels
        self.distance_metric = distance_metric
        self.name = name
        self.metric_loss_type = metric_loss_type
        self.beta = beta
        self.metric_weight = metric_weight
        self.data_root = data_root
        self.train_folder = train_folder
        self.time_window = time_window
        self.age_key_path = age_key_path
        self.metric_key_path = metric_key_path
        self.pert_time_key_path = pert_time_key_path
        self.self_target_prob = self_target_prob
        self.time_only_flag = time_only_flag
        self.other_age_penalty = other_age_penalty


    def make_dataset(self):
        """
        Load the dataset from the specified file path using pandas.
        """
        # get seq key
        seq_key = make_seq_key(self.data_root, self.train_folder)

        # load metric info
        if self.metric_key_path == '':
            self.metric_key_path = os.path.join(self.data_root, "metadata", "combined_metadata_files", "curation", "perturbation_metric_key.csv")
        metric_key_df = pd.read_csv(self.metric_key_path, index_col=0)

        
        # age_key_df = age_key_df.loc[:, ["snip_id", "stage_hpf"]]
        self.metric_key = metric_key_df
        # seq_key = seq_key.merge(age_key_df, how="left", on="snip_id")

        if self.pert_time_key_path != '':
            pert_time_key = pd.read_csv(self.pert_time_key_path)
        else:
            pert_time_key = None

        seq_key, train_indices, eval_indices, test_indices = make_train_test_split(seq_key, pert_time_key=pert_time_key)

        self.seq_key = seq_key
        self.eval_indices = eval_indices
        self.test_indices = test_indices
        self.train_indices = train_indices

        seq_key = self.seq_key

        pert_id_vec = seq_key["perturbation_id"].to_numpy()
        e_id_vec = seq_key["embryo_id_num"].to_numpy()
        age_hpf_vec = seq_key["stage_hpf"].to_numpy()

        seq_key_dict = dict({"pert_id_vec": pert_id_vec, "e_id_vec":e_id_vec, "age_hpf_vec": age_hpf_vec})
        self.seq_key_dict = seq_key_dict

        # make array version of metric key
        metric_key = self.metric_key
        pert_id_key = seq_key.loc[:, ["short_pert_name", "perturbation_id"]].drop_duplicates().reset_index(drop=True)
        metric_array = metric_key.to_numpy()
        pert_skel = pd.DataFrame(metric_key.index.tolist(), columns=["short_pert_name"])
        sort_skel = pert_skel.merge(pert_id_key, how="left", on="short_pert_name")
        id_sort_vec = np.argsort(sort_skel["perturbation_id"])
        
        metric_array = metric_array[id_sort_vec, :]
        self.metric_array = metric_array[:, id_sort_vec]

        # make boolean vactors for train, eval, and test groups
        self.train_bool = np.zeros(pert_id_vec.shape, dtype=np.bool_)
        self.train_bool[self.train_indices] = True
        self.eval_bool = np.zeros(pert_id_vec.shape, dtype=np.bool_)
        self.eval_bool[self.eval_indices] = True
        self.test_bool = np.zeros(pert_id_vec.shape, dtype=np.bool_)
        self.test_bool[self.test_indices] = True

       


