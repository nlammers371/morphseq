from __future__ import annotations
import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.build.build01A_compile_keyence_images import stitch_ff_from_keyence, build_ff_from_keyence
from src.build.build01AB_stitch_keyence_z_slices import stitch_z_from_keyence
import pandas as pd
import multiprocessing
from typing import Literal, Optional, Dict
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import glob2 as glob


@dataclass
class Experiment:
    date: str
    data_root: Path | str

    flags:      Dict[str,bool] = field(init=False)
    timestamps: Dict[str,str]  = field(init=False)

    def __post_init__(self):
        # normalize data_root
        self.data_root = Path(self.data_root)
        # load prior JSON or start blank
        self.flags      = {}
        self.timestamps = {}
        self._load_state()

        # define the steps and their canonical paths
        step_paths = {
            "raw":     self.raw_path,
            "ff":      self.ff_path,
            "stitch":  self.stitch_z_path,
            "segment": self.segment_path,
        }

        # reconcile each flag against disk
        changed = False
        for step, path in step_paths.items():
            exists = bool(path and path.exists())
            prev    = self.flags.get(step, False)
            if exists and not prev:
                # folder just appeared → mark it
                self.flags[step] = True
                mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                self.timestamps[step] = mtime
                changed = True
            elif not exists and prev:
                # folder disappeared (or was never there) → clear it
                self.flags[step] = False
                self.timestamps.pop(step, None)
                changed = True
            else:
                # no change
                self.flags.setdefault(step, exists)
                # leave timestamps alone if flag==prev
        if changed:
            self._save_state()

    # ——— path properties —————————————————————————————————————————

    @property
    def microscope(self) -> Optional[Literal["Keyence","YX1"]]:
        key = self.data_root/"raw_image_data"/"Keyence"/self.date
        yx  = self.data_root/"raw_image_data"/"YX1"   /self.date
        if key.exists() and yx.exists():
            raise RuntimeError(f"Ambiguous raw data for {self.date}")
        if key.exists(): return "Keyence"
        if yx.exists():  return "YX1"
        return None

    @property
    def raw_path(self) -> Optional[Path]:
        m = self.microscope
        return (self.data_root/"raw_image_data"/m/self.date) if m else None

    @property
    def meta_path(self) -> Optional[Path]:
        p = self.data_root/"metadata"/"well_metadata"/f"{self.date}_well_metadata.xlsx"
        return p if p.exists() else None

    @property
    def ff_path(self) -> Optional[Path]:
        p = self.data_root/"built_image_data"/"stitched_FF_images"/self.date
        return p if p.exists() else None

    @property
    def stitch_z_path(self) -> Optional[Path]:
        if self.microscope == "Keyence":
            p = self.data_root/"built_image_data"/"keyence_stitched_z"/self.date
        else:
            p = self.raw_path
        return p if p.exists() else None

    @property
    def segment_path(self) -> Optional[Path]:
        seg_root= self.data_root/"segmentation"
        mask_dirs = [p for p in seg_root.glob("mask*") if p.is_dir()]
        if len(mask_dirs) == 0:
            return None
        p_dir = mask_dirs[0] / self.date
        return p_dir if p_dir.exists() else None


    @property
    def state_file(self) -> Path:
        p = (self.data_root
              /"metadata"
              /"experiments"
              /f"{self.date}.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ——— state load/save —————————————————————————————————————————

    def _load_state(self) -> None:
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            self.flags      = data.get("flags", {})
            self.timestamps = data.get("timestamps", {})

    def _save_state(self) -> None:
        out = {"flags": self.flags, "timestamps": self.timestamps}
        self.state_file.write_text(json.dumps(out, indent=2))

    # ——— marking pipeline progress ——————————————————————————————————

    # def mark(self, step: str) -> None:
    #     """
    #     Call this at the end of any stage:
    #        exp.mark("ff");  exp.mark("stitch");  etc.
    #     """
    #     now = datetime.utcnow().isoformat()
    #     self.flags[step]      = True
    #     self.timestamps[step] = now
    #     self._save_state()

    # ——— calling pipeline functions ——————————————————————————————————

    def needs_export(self) -> bool:
        raw_ts = timestamp(self.raw_path)
        last = self.timestamps.get("export", 0)
        return raw_ts > last

    def export_images(self):
        # placeholder: implement export
        # ...
        self.flags["raw"] = True
        self.timestamps["export"] = timestamp(self.raw_path)
        self.save_state()

    def needs_build_metadata(self) -> bool:
        ff_ts = timestamp(self.ff_path)
        last = self.timestamps.get("metadata", 0)
        return ff_ts > last

    def build_image_metadata(self):
        # placeholder: implement metadata build
        # ...
        self.flags["ff"] = True
        self.timestamps["metadata"] = timestamp(self.ff_path)
        self.save_state()




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
            if exp.needs_export():
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

    root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")
    date = "20230525"

    # load master experiment log
    manager = ExperimentManager(root=root)
    exp = Experiment(date=date, data_root=root)
    test = exp.segment_path
    print("check")
    # first, assess status of each experiment within the pipeline
    # run_export_flags = master_log["microscope"].isin(["Keyence", "YX1"])


    # next, run export scripts as needed
