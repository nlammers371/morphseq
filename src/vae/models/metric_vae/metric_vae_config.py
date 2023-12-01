from pydantic.dataclasses import dataclass
from ..vae import VAEConfig
import pandas as pd


@dataclass
class MetricVAEConfig(VAEConfig):
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
    orth_flag: bool = False
    n_conv_layers: int = 5  # number of convolutional layers
    n_out_channels: int = 16  # number of layers to convolutional kernel
    distance_metric: str = "cosine"
    class_key_path: str = ''
    class_key = None
    class_ignorance_flag: bool = False
    time_ignorance_flag: bool = False  # if true, we squeeze class info out of the nuisance partition
    time_similarity_threshold: float = 3  # specifies how close in age different observations need to be to be counted as positive pairs
    gamma: float = 1.0
    name: str = "MetricVAEConfig"

    def __init__(self, class_key_path=None,
                 input_dim=(1, 288, 128),
                 latent_dim=10,
                 temperature=1.0,
                 zn_frac=0.1,
                 orth_flag=False,
                 n_conv_layers=5,  # number of convolutional layers
                 n_out_channels=16,  # number of layers to convolutional kernel
                 distance_metric="cosine",
                 class_key=None,
                 class_ignorance_flag=False,
                 time_ignorance_flag=False,  # if true, we squeeze class info out of the nuisance partition
                 time_similarity_threshold=3,
                 # specifies how close in age different observations need to be to be counted as positive pairs
                 gamma=1.0,
                 name="MetricVAEConfig",
                 uses_default_encoder=True, uses_default_decoder=True, reconstruction_loss='mse'):

        # self.__dict__.update(kwargs)
        self.uses_default_encoder = uses_default_encoder
        self.uses_default_decoder = uses_default_decoder
        self.reconstruction_loss = reconstruction_loss
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.class_key_path = class_key_path
        self.temperature = temperature
        self.zn_frac = zn_frac
        self.orth_flag = orth_flag
        self.n_conv_layers = n_conv_layers
        self.n_out_channels = n_out_channels
        self.distance_metric = distance_metric
        self.class_ignorance_flag = class_ignorance_flag
        self.time_ignorance_flag = time_ignorance_flag
        self.time_similarity_threshold = time_similarity_threshold
        self.gamma = gamma  # NL: is this used?
        self.name = name

        if self.class_key_path is not None:
            self.load_dataset()

    def load_dataset(self):
        """
        Load the dataset from the specified file path using pandas.
        """
        class_key = pd.read_csv(self.class_key_path)

        if self.class_ignorance_flag & self.time_ignorance_flag:
            class_key = class_key.loc[:, ["snip_id", "predicted_stage_hpf", "perturbation_id"]]
        elif self.class_ignorance_flag:
            class_key = class_key.loc[:, ["snip_id", "perturbation_id"]]
        elif self.time_ignorance_flag:
            class_key = class_key.loc[:, ["snip_id", "predicted_stage_hpf"]]
        else:
            class_key = None

        self.class_key = class_key






