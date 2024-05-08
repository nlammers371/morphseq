from pydantic.dataclasses import dataclass
from typing_extensions import Literal
from src.vae.auxiliary_scripts.make_training_key import make_seq_key, make_train_test_split
from ..base.base_config import BaseAEConfig
import pandas as pd

@dataclass
class VAEConfig(BaseAEConfig):
    """VAE config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
    """
    orth_flag: bool = False
    n_conv_layers: int = 5  # number of convolutional layers
    n_out_channels: int = 16
    beta: float = 1.0  # tunes the weight of the KL normalization term
    reconstruction_loss: Literal["bce", "mse"] = "mse"
    data_root: str = ''
    train_folder: str = ''
    age_key_path: str = ''

    def __init__(self, data_root, train_folder, age_key_path):
        self.data_root = data_root
        self.train_folder = train_folder
        self.age_key_path = age_key_path

    def split_train_test(self):
        """
        Load the dataset from the specified file path using pandas.
        """
        # get seq key
        seq_key = make_seq_key(self.data_root, self.train_folder)

        # if self.age_key_path != '':
        #     age_key_df = pd.read_csv(self.age_key_path, index_col=0)
        #     age_key_df = age_key_df.loc[:, ["snip_id", "inferred_stage_hpf_reg"]]
        #     seq_key = seq_key.merge(age_key_df, how="left", on="snip_id")
        # else:
        #     raise Error("No age key path provided")
            # seq_key["inferred_stage_hpf_reg"] = seq_key["predicted_stage_hpf"].copy()

        seq_key, train_indices, eval_indices, test_indices = make_train_test_split(seq_key)

        self.seq_key = seq_key
        self.eval_indices = eval_indices
        self.test_indices = test_indices
        self.train_indices = train_indices


