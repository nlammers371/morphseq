from pydantic.dataclasses import dataclass
from ..vae import VAEConfig
import pandas as pd
from src.build.make_training_key import make_seq_key, get_sequential_pairs

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
    zn_frac: float = 0.2
    orth_flag: bool = True
    n_conv_layers: int = 5  # number of convolutional layers
    n_out_channels: int = 16  # number of layers to convolutional kernel
    distance_metric: str = "euclidean"
    beta: float = 1.0  # tunes the weight of the Gaussian prior term
    name: str = "SeqVAEConfig"
    data_root: str = ''
    train_folder: str = ''

    # set sequential hyperparameters
    time_window: float = 2.0  # max permitted age difference between sequential pairs
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
        # self.seq_key_dict = seq_key_dict






