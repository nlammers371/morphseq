import numpy as np
import skimage.io as skio
from skimage import exposure
import torch
from torch.utils.data import Dataset
from typing import Tuple, Any
from src.build.export_utils import im_rescale
from collections import OrderedDict

class DatasetOutput(OrderedDict):
    """Base ModelOutput class fixing the output type from the models. This class is inspired from
    the ``ModelOutput`` class from hugginface transformers library"""

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())
    

class MultiTileZStackDataset(Dataset):
    def __init__(self, samples, ff_dtype=np.float32):
        self.samples   = samples
        self.ff_dtype  = ff_dtype

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry       = self.samples[idx]
        all_stacks  = []
        
        # 1) load each tile’s Z-stack
        for zpaths in entry["tile_zpaths"]:
            stack = np.stack([skio.imread(str(p)) for p in zpaths], axis=0)
            all_stacks.append(stack.astype(self.ff_dtype))

        # 2) compute one shared percentile‐range across _every_ voxel in every tile
        flat = np.concatenate([s.ravel() for s in all_stacks])
        lo, hi = np.percentile(flat, [.1, 99.9])

        # 3) apply the same contrast‐stretch to each tile
        stretched = [
            exposure.rescale_intensity(s, in_range=(lo, hi), out_range=(0, 1))
            for s in all_stacks
        ]

        # 4) optionally do your FF‐projection per tile here, e.g. gaussian or LoG
        #    For simplicity we’ll just return the 3D stacks
        arr = np.stack(stretched, axis=0)
        # shape == (n_tiles, Z, H, W)

        return DatasetOutput(data=torch.from_numpy(arr), path=entry)