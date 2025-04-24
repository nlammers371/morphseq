from torch.utils.data import Dataset
from pydantic.dataclasses import dataclass
from dataclasses import field
from typing import Literal, List, Type, Callable, Any, Dict
from src.data.dataset_utils import make_seq_key, make_train_test_split
from src.data.data_transforms import make_dynamic_rs_transform
from src.data.dataset_classes import BasicDataset
import os
import numpy as np
import pandas as pd
from pydantic   import ConfigDict

@dataclass# (config_wrapper=ConfigDict(arbitrary_types_allowed=True))
class BaseDataConfig:

    seq_key: List[Dict[str,Any]] = field(default_factory=list)

    batch_size:    int                     = 64
    num_workers:   int                     = 4
    shuffle:       bool                    = True
    wrap:          bool                    = True
    root:          str                     = "./data"

    # 1) pick by name, not by object
    target_name:   Literal["BasicDataset"] = "BasicDataset"
    # 2) a catch‐all for per‐dataset options
    target_kwargs: Dict[str,Any]        = field(default_factory=dict)

    # similarly for transform
    transform_name:   Literal["dynamic_rs"] = "dynamic_rs"
    transform_kwargs: Dict[str,Any]                = field(default_factory=dict)

    # … same primitives …
    train_indices: List[int] = field(default_factory=list)
    eval_indices: List[int] = field(default_factory=list)
    test_indices: List[int] = field(default_factory=list)

    def __post_init__(self):
        raw = self.seq_key
        self.seq_key = pd.DataFrame(raw)
        # now convert them to arrays
        self.train_indices = np.array(self.train_indices, dtype=int)
        self.eval_indices = np.array(self.eval_indices, dtype=int)
        self.test_indices = np.array(self.test_indices, dtype=int)

    @property
    def image_path(self) -> str:
        return os.path.join(self.root, "images")

    @property
    def age_key_path(self) -> str:
        return os.path.join(self.root, "metadata", "age_key.csv")

    @property
    def pert_time_key_path(self) -> str:
        return os.path.join(self.root, "metadata", "perturbation_train_key.csv")


    def split_train_test(self):
        """
        Load the dataset from the specified file path using pandas.
        """
        # get seq key
        seq_key = make_seq_key(self.root)

        if os.path.isfile(self.age_key_path):
            age_key_df = pd.read_csv(self.age_key_path)
            age_key_df = age_key_df.loc[:, ["snip_id", "inferred_stage_hpf_reg"]]
            seq_key = seq_key.merge(age_key_df, how="left", on="snip_id")
        else:
            raise Exception("Stage key provided!")

        if os.path.isfile(self.pert_time_key_path):
            pert_time_key = pd.read_csv(self.pert_time_key_path)
        else:
            # raise Exception("No perturbation-time key provided!")
            pert_time_key = None

        seq_key, train_indices, eval_indices, test_indices = make_train_test_split(seq_key, pert_time_key=pert_time_key)

        self.seq_key = seq_key
        self.eval_indices = eval_indices
        self.test_indices = test_indices
        self.train_indices = train_indices


    def create_dataset(self):
        # map names→classes/functions
        ds_map = {
            "BasicDataset": BasicDataset,
            # "OtherDataset": OtherDataset,
        }
        tf_map = {
            "dynamic_rs": make_dynamic_rs_transform,
            # "other":      another_transform,
        }

        ds_cls = ds_map[self.target_name]
        tf_fn  = tf_map[self.transform_name]

        # build the actual transform
        transform = tf_fn(**self.transform_kwargs)

        # instantiate your dataset with both fixed and configurable args
        return ds_cls(
            root=self.image_path,
            transform=transform,
            **self.target_kwargs
        )

# @dataclass
# class BasicDataConfig:
#     dataset_path: Literal["BasicDataset"] = "BasicDataset"
#     data_root:    str  = "./data"
#     batch_size:   int  = 64
#
#     def create(self):
#         cls = getattr(import_module("src.datasets"), self.dataset_path)
#         return cls(self.data_root)