from torch.utils.data import Dataset
from pydantic.dataclasses import dataclass
from dataclasses import field
from typing import Literal, Type, Callable, Any, Dict
from importlib import import_module
from src.data.data_transforms import make_dynamic_rs_transform
from src.data.dataset_classes import BasicDataset
import os
import numpy as np

@dataclass
class BaseDataConfig:
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

    ##########
    # Paths to metadata

    # an empty 1-D integer array
    age_key_path: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=str)
    )
    # an empty 1-D string array (NumPy will pick a Unicode dtype)
    pert_time_key_path: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=str)
    )

    # indices for sampling
    train_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    # an empty 1-D string array (NumPy will pick a Unicode dtype)
    eval_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )

    test_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )


    def __post_init__(self):
        up_folder = os.path.dirname(self.root)
        self.age_key_path = os.path.join(up_folder, "metadata", "age_key.csv")
        self.pert_time_key_path = os.path.join(up_folder, "metadata", "pert_time_key.csv")


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
            root=self.root,
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