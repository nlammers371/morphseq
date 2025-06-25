from __future__ import annotations
import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.build.build01A_compile_keyence_torch import stitch_ff_from_keyence, build_ff_from_keyence
from src.build.build01AB_stitch_keyence_z_slices import stitch_z_from_keyence
from src.build.build01B_compile_yx1_images_torch import build_ff_from_yx1
import pandas as pd
import multiprocessing
from typing import Literal, Optional, Dict, List
import torch
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import glob2 as glob
import functools
import os


def _mod_time(path: Optional[Path]) -> float:
    return path.stat().st_mtime if path and path.exists() else 0.0

def record(step: str):
    """
    Decorator that only marks `step` if the wrapped method
    completes without throwing.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(self, *args, **kwargs):
            try:
                result = fn(self, *args, **kwargs)
            except Exception:
                # the exception propagates, and we do *not* record the step
                raise
            else:
                # only record on success
                self.record_step(step)
                return result
        return wrapped
    return decorator


@dataclass
class Experiment:
    date: str
    data_root: Path | str
    n_workers: int = 1
    flags:      Dict[str,bool] = field(init=False)
    timestamps: Dict[str,str]  = field(init=False)

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.flags = {}
        self.timestamps = {}
        self._load_state()
        self._sync_with_disk()

    # # ——— public API ——————————————————————————————————————————————————

    def record_step(self, step: str):
        """Mark `step` done right now and save state."""
        now = datetime.utcnow().isoformat()
        self.flags[step] = True
        self.timestamps[step] = now
        self._save_state()

    # ——— path properties ———————————————————————————————————————————————
    @property
    def num_cpu_workers(self, prefactor: float = 0.25, min_workers: int = 1, max_workers: int = 24) -> int:
        """
        Returns a recommended number of CPU workers.
        By default uses half of all logical cores (but at least min_workers).
        You can tune `prefactor` between 0 and 1.
        """
        total = os.cpu_count() or 1
        n = min(max(min_workers, int(total * prefactor)), max_workers)
        return n
    
    @property
    def has_gpu(self) -> bool:
        """
        Returns True if PyTorch can see at least one CUDA device.
        Falls back to False if torch isn’t installed.
        """
        return torch.cuda.is_available()
    
    @property
    def gpu_names(self) -> List[str]:
        """
        Returns a list of device names, e.g. ['Tesla V100', ...].
        Empty if no GPUs or torch unavailable.
        """
        if not torch.cuda.is_available():
            return []
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    @property
    def microscope(self) -> Optional[Literal["Keyence","YX1"]]:
        key = self.data_root/"raw_image_data"/"Keyence"/self.date
        yx  = self.data_root/"raw_image_data"/"YX1"   /self.date
        if   key.exists() and not yx.exists(): return "Keyence"
        elif yx.exists() and not key.exists(): return "YX1"
        elif key.exists() and yx.exists():
            raise RuntimeError(f"Ambiguous raw data for {self.date}")
        else:
            return None

    @property
    def raw_path(self) -> Optional[Path]:
        m = self.microscope
        return (self.data_root/"raw_image_data"/m/self.date) if m else None

    @property
    def ff_path(self) -> Optional[Path]:
        p = self.data_root/"built_image_data"/"stitched_FF_images"/self.date
        return p if p.exists() else None

    @property
    def stitch_z_path(self) -> Optional[Path]:
        if self.microscope=="Keyence":
            p = self.data_root/"built_image_data"/"keyence_stitched_z"/self.date
        else:
            p = self.raw_path
        return p if p and p.exists() else None

    @property
    def segment_path(self) -> Optional[Path]:
        seg_root = self.data_root/"segmentation"
        masks = [d for d in seg_root.glob("mask*") if d.is_dir()]
        if not masks: return None
        candidate = masks[0]/self.date
        return candidate if candidate.exists() else None

    @property
    def state_file(self) -> Path:
        p = self.data_root/"metadata"/"experiments"/f"{self.date}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    
    @property
    def needs_export(self) -> bool:
        """True if raw data is newer than last-export timestamp."""
        return (_mod_time(self.raw_path) >
                datetime.fromisoformat(self.timestamps.get("export", "1970-01-01T00:00:00")).timestamp())

    @property
    def needs_build_metadata(self) -> bool:
        return (_mod_time(self.ff_path) >
                datetime.fromisoformat(self.timestamps.get("metadata", "1970-01-01T00:00:00")).timestamp())

    @property
    def needs_stitch(self) -> bool:
        return (_mod_time(self.ff_path) >
                datetime.fromisoformat(self.timestamps.get("stitch", "1970-01-01T00:00:00")).timestamp())

    @property
    def needs_segment(self) -> bool:
        return (_mod_time(self.stitch_z_path) >
                datetime.fromisoformat(self.timestamps.get("segment", "1970-01-01T00:00:00")).timestamp())

    # ——— internal sync logic —————————————————————————————————————————————

    def _sync_with_disk(self):
        """
        Check each pipeline step’s path on disk,
        update flags + timestamps if they’ve just appeared/disappeared.
        """
        steps = {
            "raw":     self.raw_path,
            "ff":      self.ff_path,
            "stitch":  self.stitch_z_path,
            "segment": self.segment_path,
        }
        changed = False
        for step, path in steps.items():
            exists = bool(path and path.exists())
            prev   = self.flags.get(step, False)

            if exists and not prev:
                # step just became available
                self.flags[step] = True
                self.timestamps[step] = datetime.utcnow().isoformat()
                changed = True
            elif not exists and prev:
                # step vanished
                self.flags[step] = False
                self.timestamps.pop(step, None)
                changed = True
            else:
                # no change: ensure the flag is recorded
                self.flags.setdefault(step, exists)

        if changed:
            self._save_state()

    # ——— call pipeline functions —————————————————————————————————————————————
    @record("ff")
    def export_images(self):

        if self.microscope == "Keyence":
            build_ff_from_keyence(data_root=self.data_root, exp_name=self.date, overwrite=True)
        else:
            build_ff_from_yx1(data_root=self.data_root, exp_name=self.date, overwrite=True)

    @record("stitch")
    def stitch_images(self):
        if self.microscope == "Keyence":
            stitch_ff_from_keyence(data_root=self.data_root, exp_name=self.date, overwrite=True, n_workers=self.num_cpu_workers)
            stitch_z_from_keyence(data_root=self.data_root, exp_name=self.date, overwrite=True, n_workers=self.num_cpu_workers)
        else:
            pass


    # ——— load/save ——————————————————————————————————————————————————

    def _load_state(self):
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            self.flags      = data.get("flags", {})
            self.timestamps = data.get("timestamps", {})

    def _save_state(self):
        payload = {"flags": self.flags, "timestamps": self.timestamps}
        self.state_file.write_text(json.dumps(payload, indent=2))




class ExperimentManager:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.exp_dir = self.root / "metadata" / "experiments"
        self.experiments: dict[str, Experiment] = {}
        self.discover_experiments()

    def discover_experiments(self):
        # scan "raw_image_data" subfolders for dates
        raw = self.root / "raw_image_data"
        for mic in raw.iterdir():
            if mic.is_dir():
                for d in (mic).iterdir():
                    date = d.name
                    if date not in self.experiments:
                        self.experiments[date] = Experiment(date, self.root)

    def export_all(self):
        for exp in self.experiments.values():
            if exp.needs_export:
                exp.export_images()

    def build_metadata_all(self):
        for exp in self.experiments.values():
            if exp.needs_build_metadata():
                exp.build_image_metadata()

    def report(self):
        for date, exp in self.experiments.items():
            print(f"{date}: raw={exp.flags['raw']}, ff={exp.flags['ff']}, stitch={exp.flags['stitch']}, segment={exp.flags['segment']}")



if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Call pipeline functions
    root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")

    # load master experiment log
    manager = ExperimentManager(root=root)
    # exp = manager.experiments["20230525"]
    # exp.export_images()
    manager.export_all()   

    print("check")
    # first, assess status of each experiment within the pipeline
    # run_export_flags = master_log["microscope"].isin(["Keyence", "YX1"])


    # next, run export scripts as needed
