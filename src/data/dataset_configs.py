from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset


class _SnipDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], transform: Optional[Callable] = None):
        self._paths = list(image_paths)
        self._transform = transform

    def __len__(self) -> int:  # noqa: D401
        return len(self._paths)

    def __getitem__(self, idx: int):
        p = self._paths[idx]
        img = Image.open(p).convert("L")
        if self._transform is not None:
            x = self._transform(img)
        else:
            # Default: 1×H×W float tensor in [0, 1]
            x = torch.from_numpy(torch.tensor(img, dtype=torch.uint8).numpy()).float() / 255.0
            x = x.unsqueeze(0)

        # Match the shape expected by `extract_embeddings_legacy`:
        # DataLoader default-collate turns a batch of tuples into a tuple-of-batches,
        # so `inputs["label"][0]` becomes the list of file paths.
        return {"data": x, "label": (str(p), 0)}


def _snips_root_candidates(root: Path) -> List[Path]:
    candidates = [
        root / "training_data" / "bf_embryo_snips",
        # Some callers pass repo_root instead of data_root; try the conventional playground location.
        root / "morphseq_playground" / "training_data" / "bf_embryo_snips",
    ]
    return candidates


def _collect_snip_paths(root: Path, experiments: Optional[Sequence[str]]) -> List[Path]:
    snips_root = None
    for c in _snips_root_candidates(root):
        if c.is_dir():
            snips_root = c
            break
    if snips_root is None:
        tried = "\n  - ".join(str(c) for c in _snips_root_candidates(root))
        raise FileNotFoundError(f"bf_embryo_snips not found. Tried:\n  - {tried}")

    if experiments is None:
        exp_dirs = [p for p in snips_root.iterdir() if p.is_dir()]
    else:
        exp_dirs = [snips_root / str(e) for e in experiments]

    paths: List[Path] = []
    for d in exp_dirs:
        if not d.is_dir():
            continue
        paths.extend(sorted(d.glob("*.jpg")))
        paths.extend(sorted(d.glob("*.png")))
    return paths


@dataclass
class BaseDataConfig:
    """Minimal shim for analysis scripts.

    This is not a full replacement for the original training-time data config;
    it only supports the subset used by downstream analysis utilities.
    """

    root: Union[str, Path]
    return_sample_names: bool = True
    transforms: Optional[Callable] = None
    transform_name: str = "basic"
    num_workers: int = 0

    # Optional split indices (present in some analysis flows)
    train_indices: Optional[Sequence[int]] = None
    test_indices: Optional[Sequence[int]] = None
    eval_indices: Optional[Sequence[int]] = None

    def make_metadata(self) -> None:
        # Legacy API hook; no-op for the snip-folder dataset.
        return None

    def create_dataset(self) -> Dataset:
        root_p = Path(self.root)
        paths = _collect_snip_paths(root_p, experiments=None)
        return _SnipDataset(paths, transform=self.transforms)


@dataclass
class EvalDataConfig:
    experiments: Optional[List[str]]
    root: Union[str, Path]
    return_sample_names: bool = True
    transforms: Optional[Callable] = None
    num_workers: int = 0

    def create_dataset(self) -> Dataset:
        root_p = Path(self.root)
        paths = _collect_snip_paths(root_p, experiments=self.experiments)
        return _SnipDataset(paths, transform=self.transforms)


class NTXentDataConfig(BaseDataConfig):
    """Placeholder to satisfy imports; not used by the pipeline embedding path."""

    pass

