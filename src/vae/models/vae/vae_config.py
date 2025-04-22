from dataclasses import dataclass, fields, field
from ..base.base_config import BaseAEConfig
from typing import Any, Dict, Optional
from src.run.run_utils import deep_merge

@dataclass
class VAEConfig():
    """VAE config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
    """
    name: str = "VAE"
    ddconfig: Dict[str, Any] = field(default_factory=
                                lambda: { "latent_dim": 64,
                                          "input_dim": (1, 288, 128),
                                          "n_channels_out": 16,
                                          "n_conv_layers": 5,
                                })
    # ── Dataset defaults (string path + kwargs) ────────────────────
    # dataset_cls: str = "src.data.DatasetClasses.BasicDataset"

    @classmethod
    def from_cfg(cls, model_cfg: Dict[str, Any]) -> "ModelConfig":
        # filter only the fields we know about
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        clean = {k: v for k, v in model_cfg.items() if k in valid}

        # 1) create a “base” instance with ALL your code‑defaults
        inst = cls()

        # 2) for each user key, merge or overwrite appropriately
        for key, override in clean.items():
            default = getattr(inst, key)

            # If it's a dict‐default, do a deep merge
            if isinstance(default, dict) and isinstance(override, dict):
                merged = deep_merge(default, override)
                setattr(inst, key, merged)
            else:
                setattr(inst, key, override)


        return inst

    # def split_train_test(self):
    #     """
    #     Load the dataset from the specified file path using pandas.
    #     """
    #     # get seq key
    #     seq_key = make_seq_key(self.data_root, self.train_folder)
    #
    #     if self.age_key_path != '':
    #         age_key_df = pd.read_csv(self.age_key_path, index_col=0)
    #         age_key_df = age_key_df.loc[:, ["snip_id", "inferred_stage_hpf_reg"]]
    #         seq_key = seq_key.merge(age_key_df, how="left", on="snip_id")
    #     else:
    #         # raise Warning("No age key path provided")
    #         seq_key["inferred_stage_hpf_reg"] = 1
    #
    #     if self.pert_time_key_path != '':
    #         pert_time_key = pd.read_csv(self.pert_time_key_path)
    #     else:
    #         pert_time_key = None
    #
    #     seq_key, train_indices, eval_indices, test_indices = make_train_test_split(seq_key, pert_time_key=pert_time_key)
    #
    #     self.seq_key = seq_key
    #     self.eval_indices = eval_indices
    #     self.test_indices = test_indices
    #     self.train_indices = train_indices


