from torch.utils.data import Dataset
from pydantic.dataclasses import dataclass
from dataclasses import field
from typing import Literal, List, Type, Callable, Any, Dict, Optional
from src.data.dataset_utils import make_seq_key, make_train_test_split
from src.data.data_transforms import basic_transform, contrastive_transform
from src.data.dataset_classes import BasicDataset, NTXentDataset
import os
import numpy as np
import pandas as pd
from src.data.dataset_utils import smart_read_csv
from pydantic   import ConfigDict


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class UrrDataConfig:

    seq_key: pd.DataFrame = field(default_factory=pd.DataFrame)
    train_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    eval_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    test_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    batch_size: int = 64
    num_workers: int = 4
    wrap: bool = True
    root: str = "./data"

    @property
    def image_path(self) -> str:
        return os.path.join(self.root, "images")

    @property
    def metadata_path(self) -> str:
        return os.path.join(self.root, "metadata", "")

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
        if (self.train_indices.size
                and self.eval_indices.size
                and self.test_indices.size
        ):
            pass
        else: #  overwrite if empty
            self.eval_indices = eval_indices
            self.test_indices = test_indices
            self.train_indices = train_indices


@dataclass# (config_wrapper=ConfigDict(arbitrary_types_allowed=True))
class BaseDataConfig(UrrDataConfig):

    # 1) pick by name, not by object
    target_name:   Literal["BasicDataset"] = "BasicDataset"

    # 2) a catch‐all for per‐dataset options
    target_kwargs: Dict[str,Any]        = field(default_factory=dict)

    # similarly for transform
    transform_name:   Literal["basic", "simclr"] = "simclr"
    transform_kwargs: Dict[str,Any]                = field(default_factory=dict)

    return_sample_names: bool = False

    def make_metadata(self):
        self.split_train_test()

    def create_dataset(self):
        # map names→classes/functions
        ds_map = {
            "BasicDataset": BasicDataset,
            # "OtherDataset": OtherDataset,
        }
        tf_map = {
            "basic": basic_transform,
            "simclr": contrastive_transform,
        }

        ds_cls = ds_map[self.target_name]
        tf_fn  = tf_map[self.transform_name]

        # build the actual transform
        transform = tf_fn(**self.transform_kwargs)

        # instantiate your dataset with both fixed and configurable args
        return ds_cls(
            root=self.image_path,
            return_name=self.return_sample_names,
            transform=transform,
            **self.target_kwargs
        )

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class NTXentDataConfig(UrrDataConfig):

    seq_key_dict: Dict[str, np.ndarray] = field(
        default_factory=dict
    )
    metric_key: pd.DataFrame = field(default_factory=pd.DataFrame)

    metric_array: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    train_bool: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    eval_bool: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    test_bool: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    return_name: bool = True

    target_name:   Literal["NTXentDataset"] = "NTXentDataset"

    # 2) a catch‐all for per‐dataset options
    target_kwargs: Dict[str,Any]        = field(default_factory=dict)

    # these attributes will be pulled from loss config
    time_window: Optional[int] = None
    self_target_prob: Optional[int] = None

    # similarly for transform
    transform_name:   Literal["simclr"] = "simclr"
    transform_kwargs: Dict[str,Any]                = field(default_factory=dict)


    @property
    def metric_key_path(self) -> str:
        return os.path.join(self.root, "metadata", "metric_key.csv")


    def create_dataset(self):
        # map names→classes/functions
        ds_map = {
            "NTXentDataset": NTXentDataset,
            # "OtherDataset": OtherDataset,
        }
        tf_map = {
            "basic": basic_transform(),
            "simclr":     contrastive_transform,
        }

        ds_cls = ds_map[self.target_name]
        tf_fn  = tf_map[self.transform_name]

        # build the actual transform
        transform = tf_fn(**self.transform_kwargs)

        # instantiate your dataset with both fixed and configurable args
        return ds_cls(
            cfg=self,
            transform=transform,
        )

    def make_metadata(self):
        # produces train/test/eval indices and seq_key and pert_time_key
        self.split_train_test()

        # load metric key
        metric_key = smart_read_csv(self.metric_key_path)
        self.metric_key = metric_key

        # generate some indexing/metadata vectors
        seq_key = self.seq_key

        pert_id_vec = seq_key["perturbation_id"].to_numpy()
        e_id_vec = seq_key["embryo_id_num"].to_numpy()
        age_hpf_vec = seq_key["stage_hpf"].to_numpy()

        seq_key_dict = dict({"pert_id_vec": pert_id_vec, "e_id_vec": e_id_vec, "age_hpf_vec": age_hpf_vec})
        self.seq_key_dict = seq_key_dict

        # make array version of metric key
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
