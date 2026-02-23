"""
MorphSeq Pipeline Objects: Intelligent Experiment Management and Orchestration

This module provides the core classes for managing MorphSeq experiments with intelligent
tracking, state management, and pipeline orchestration. It enables seamless coordination
between per-experiment processing and global cohort-level operations.

Architecture Overview:
===================

Individual Experiments (Experiment class):
├── Raw data acquisition & FF image creation (Build01)
├── QC mask generation (Build02) OR SAM2 segmentation 
├── Embryo processing & df01 contribution (Build03)
└── Latent embedding generation (per-experiment)

Global Operations (ExperimentManager class):
├── df01 → QC & staging → df02 (Build04)
└── df02 + latents → final dataset → df03 (Build06)

Key Features:
============
✓ **Intelligent State Tracking**: JSON-based state files with timestamp comparison
✓ **Automatic State Sync**: Detects existing work from previous runs
✓ **Dependency Management**: Ensures correct execution order and prerequisites
✓ **Duplicate Prevention**: Avoids reprocessing data already in downstream files
✓ **Flexible Orchestration**: Supports individual steps or full end-to-end workflows

Classes:
========
- Experiment: Manages individual experiment lifecycle and per-experiment operations
- ExperimentManager: Orchestrates multiple experiments and global operations

Author: Claude Code with MorphSeq Team
Stage: 3 - Intelligent Pipeline Orchestration (Complete)
"""

from __future__ import annotations
import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust "2" if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.build.build01A_compile_keyence_torch import stitch_ff_from_keyence, build_ff_from_keyence
from src.build.build01AB_stitch_keyence_z_slices import stitch_z_from_keyence
from src.build.build01B_compile_yx1_images_torch import build_ff_from_yx1
import pandas as pd
import multiprocessing
from typing import Literal, Optional, Dict, List, Sequence, Union
import torch
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import glob2 as glob
import functools
import os
import logging
import itertools
from src.build.export_utils import PATTERNS, _match_files, has_output, newest_mtime, _mod_time
from src.run_morphseq_pipeline.paths import (
    # Global/non-generated inputs
    get_stage_ref_csv,
    get_perturbation_key_csv,
    get_well_metadata_xlsx,
    # Build01
    get_keyence_ff_dir,
    get_stitched_ff_dir,
    get_built_metadata_csv,
    # SAM2
    get_sam2_csv,
    get_sam2_masks_dir,
    get_gdino_detections_json,
    get_sam2_segmentations_json,
    get_sam2_mask_export_manifest,
    get_experiment_metadata_json,
    # Build03
    get_build03_output,
    get_snips_dir,
    # Build04
    get_build04_output,
    # Build06
    get_latents_csv,
    get_build06_output,
)
from src.build.build03A_process_images import segment_wells, compile_embryo_stats, extract_embryo_snips

# Dependency simplification notes (comments only; no behavior change):
# - glob2: used only to match files; replace with `Path.glob()` to drop glob2.
# - pandas in this file is used primarily for IO; could be minimized if needed.
# - Consider lazy imports for heavyweight modules in methods that need them (e.g. stitch/segment).


log = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)

FF_COMPLETE_MARKER = ".ff_complete"



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
    """
    Represents a single MorphSeq experiment with intelligent pipeline tracking.
    
    This class manages the complete lifecycle of a MorphSeq experiment, tracking
    the state of each pipeline step and determining what work needs to be done.
    
    Pipeline Flow:
    1. Raw data → FF images (Build01)
    2. FF images → QC masks (Build02) 
    3. FF images → SAM2 segmentation (SAM2)
    4. SAM2/QC masks → embryo processing (Build03) → contributes to df01
    5. Individual processing → latent embeddings (per-experiment)
    6. df01 → df02 (Build04, global QC)
    7. df02 + latents → df03 (Build06, global merge)
    
    State Management:
    - Tracks completion via JSON state files in metadata/experiments/
    - Uses file timestamps to detect when inputs are newer than outputs
    - Automatically syncs with existing combined metadata files (df01/df02/df03)
    - Avoids duplicate processing by checking downstream file inclusion
    
    Attributes:
        date: Historical experiment string (commonly the raw folder name)
        data_root: Path to MorphSeq data directory
        n_workers: Number of CPU workers for processing (auto-calculated if not set)
        flags: Dict tracking which pipeline steps have completed
        timestamps: Dict tracking when each step was last run
        repo_root: Path to MorphSeq repository
    """
    date: str
    data_root: Union[Path, str]
    n_workers: int = 1
    flags:      Dict[str,bool] = field(init=False)
    timestamps: Dict[str,str]  = field(init=False)
    repo_root:  Path = field(init=False)

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.flags = {}
        self.timestamps = {}

        # Determine repo path relative to this script
        script_path = Path(__file__).resolve()
        self.repo_root = script_path.parents[2]

        self._load_state()
        self._sync_with_disk()

    # # ——— public API ——————————————————————————————————————————————————
    def record_step(self, step: str):
        """Mark `step` done right now and save state."""
        now = datetime.utcnow().isoformat()
        self.flags[step] = True
        self.timestamps[step] = now
        self._save_state()

    # ——— timestamp helpers ————————————————————————————————————————————
    def _ts(self, key: str, default: float = 0.0) -> float:
        """
        Safely return a float UNIX timestamp for a recorded step.
        Accepts either float seconds or ISO-8601 string in the state file.
        """
        val = self.timestamps.get(key, None)
        if val is None:
            return default
        try:
            # common case: already a float (mtime)
            if isinstance(val, (int, float)):
                return float(val)
            # maybe an ISO string
            return datetime.fromisoformat(str(val)).timestamp()
        except Exception:
            return default

    def _safe_mtime_compare(self, file_path: Path, timestamp_key: str) -> bool:
        """Return True if file is newer than recorded timestamp for key."""
        try:
            if not file_path or not file_path.exists():
                return False
            file_mtime = file_path.stat().st_mtime
            last_run = self._ts(timestamp_key)
            return file_mtime > last_run
        except Exception:
            return False

    # ——— path properties ———————————————————————————————————————————————
    @property
    def experiment_id(self) -> str:
        """Canonical per-experiment identifier used in filenames.

        We derive this from the raw_image_data folder name when available to make
        intent explicit. Fallback to `self.date` to preserve existing behavior.
        """
        try:
            rp = self.raw_path
            if rp is not None and rp.exists():
                return rp.name
        except Exception:
            pass
        return self.date
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
    def meta_path(self) -> Optional[Path]:
        """Well metadata Excel file path."""
        try:
            p = get_well_metadata_xlsx(self.data_root, self.date)
            return p if p.exists() else None
        except Exception:
            return None
    
    @property
    def meta_path_built(self) -> Optional[Path]:
        """Built metadata CSV file path."""
        try:
            p = get_built_metadata_csv(self.data_root, self.date)
            return p if p.exists() else None
        except Exception:
            return None
    
    @property
    def meta_path_embryo(self) -> Optional[Path]:
        p = self.data_root/"metadata"/"embryo_metadata_files"/ f"{self.date}_embryo_metadata.csv"
        return p if p.exists() else None
    
    @property
    def snip_path(self) -> Optional[Path]:
        """Training snips directory path."""
        try:
            p = get_snips_dir(self.data_root, self.date)
            return p if p.exists() else None
        except Exception:
            return None

    @property
    def ff_path(self) -> Optional[Path]:
        """Path to stitched FF images - final output location for both microscope types."""
        try:
            p = get_stitched_ff_dir(self.data_root, self.date)
            return p if p.exists() else None
        except Exception:
            return None

    def _ff_complete_marker(self) -> Optional[Path]:
        if not self.ff_path:
            return None
        return self.ff_path / FF_COMPLETE_MARKER

    def _count_ff_images(self) -> int:
        if not self.ff_path:
            return 0
        patterns = ("*_stitch.jpg", "*_stitch.png")
        total = 0
        for pat in patterns:
            total += sum(1 for _ in self.ff_path.glob(pat))
        return total

    def _expected_ff_count_from_meta(self) -> Optional[int]:
        meta_csv = self.meta_path_built
        if not meta_csv or not meta_csv.exists():
            return None
        try:
            with meta_csv.open("r", encoding="utf-8", errors="ignore") as handle:
                # subtract header row if present
                count = sum(1 for _ in handle) - 1
            return max(count, 0)
        except Exception as exc:
            log.warning("Could not read metadata CSV for %s: %s", self.date, exc)
            return None

    def _ff_complete(self) -> bool:
        if not self.ff_path or not self.ff_path.exists():
            return False
        marker = self._ff_complete_marker()
        if marker and marker.exists():
            return True
        # For YX1, require full image count to avoid marking partial runs as complete.
        if self.microscope == "YX1":
            expected = self._expected_ff_count_from_meta()
            if expected is None or expected == 0:
                return False
            actual = self._count_ff_images()
            if actual >= expected:
                return True
            log.warning("YX1 FF incomplete for %s: have %d/%d images", self.date, actual, expected)
            return False
        return has_output(self.ff_path, PATTERNS["ff"])

    @property
    def stitch_ff_path(self) -> Optional[Path]:
        """Path to stitched FF images - same as ff_path, SAM2 input location."""
        return self.ff_path  # Same location for both microscope types
    
    @property
    def stitch_z_path(self) -> Optional[Path]:
        if self.microscope=="Keyence":
            p = self.data_root/"built_image_data"/"Keyence_stitched_z"/self.date
        else:
            p = self.raw_path
        return p if p and p.exists() else None

    @property
    def mask_path(self) -> Optional[Path]:
        seg_root = self.data_root/"segmentation"
        masks = [d for d in seg_root.glob("mask*") if d.is_dir()]
        if not masks: return None
        candidate = masks[0]/self.date
        return candidate if candidate.exists() else None

    # ——— new per-experiment tracking properties ————————————————
    @property
    def sam2_csv_path(self) -> Path:
        """Expected per-experiment SAM2 metadata CSV path."""
        try:
            return get_sam2_csv(self.data_root, self.date)
        except Exception:
            return Path(str(self.data_root)) / "sam2_pipeline_files" / "sam2_expr_files" / f"sam2_metadata_{self.date}.csv"

    # ——— planned per-experiment intermediate/final artifacts —————————————
    @property
    def gdino_detections_path(self) -> Path:
        """Per-experiment GroundingDINO detections JSON path."""
        try:
            p = get_gdino_detections_json(self.data_root, self.date)
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            base = Path(self.data_root) / "sam2_pipeline_files" / "detections"
            base.mkdir(parents=True, exist_ok=True)
            return base / f"gdino_detections_{self.date}.json"

    @property
    def sam2_segmentations_path(self) -> Path:
        """Per-experiment SAM2 segmentations JSON path."""
        try:
            p = get_sam2_segmentations_json(self.data_root, self.date)
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            base = Path(self.data_root) / "sam2_pipeline_files" / "segmentation"
            base.mkdir(parents=True, exist_ok=True)
            return base / f"grounded_sam_segmentations_{self.date}.json"

    @property
    def build03_path(self) -> Path:
        """Per-experiment Build03 embryo metadata CSV path."""
        try:
            p = get_build03_output(self.data_root, self.date)
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            base = Path(self.data_root) / "metadata" / "build03_output"
            base.mkdir(parents=True, exist_ok=True)
            return base / f"expr_embryo_metadata_{self.date}.csv"

    @property
    def build04_path(self) -> Path:
        """Per-experiment Build04 QC+staged CSV path (matches actual implementation)."""
        try:
            p = get_build04_output(self.data_root, self.date)
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            base = Path(self.data_root) / "metadata" / "build04_output"
            base.mkdir(parents=True, exist_ok=True)
            return base / f"qc_staged_{self.date}.csv"

    @property
    def build06_path(self) -> Path:
        """Per-experiment Build06 df03 path (matches actual implementation)."""
        try:
            p = get_build06_output(self.data_root, self.date)
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            base = Path(self.data_root) / "metadata" / "build06_output"
            base.mkdir(parents=True, exist_ok=True)
            return base / f"df03_final_output_with_latents_{self.date}.csv"

    @property
    def build06_final_path(self) -> Path:
        """Legacy property - use build06_path instead."""
        return self.build06_path

    def qc_mask_status(self) -> tuple[int, int]:
        """Return (present_count, total_count) across the 5 QC mask model outputs."""
        mask_types = [
            "mask_v0_0100",
            "yolk_v1_0050",
            "focus_v0_0100",
            "bubble_v0_0100",
            "via_v1_0100",
        ]
        present = 0
        try:
            seg_root = self.data_root / "segmentation"
            for mt in mask_types:
                if (seg_root / f"{mt}_predictions" / self.date).exists():
                    present += 1
        except Exception:
            # treat as zero present on errors
            present = 0
        return present, len(mask_types)

    @property
    def has_all_qc_masks(self) -> bool:
        p, t = self.qc_mask_status()
        return p == t and t > 0

    def get_latent_path(self, model_name: str) -> Path:
        try:
            return get_latents_csv(self.data_root, model_name, self.date)
        except Exception:
            return Path(str(self.data_root)) / "analysis" / "latent_embeddings" / "legacy" / model_name / f"morph_latents_{self.date}.csv"

    def has_latents(self, model_name: str = "20241107_ds_sweep01_optimum") -> bool:
        try:
            return self.get_latent_path(model_name).exists()
        except Exception:
            return False

    # ——— Additional SAM2 convenience paths (prep + masks) ————————————
    @property
    def sam2_masks_dir(self) -> Path:
        try:
            return get_sam2_masks_dir(self.data_root, self.date)
        except Exception:
            return Path(str(self.data_root)) / "sam2_pipeline_files" / "exported_masks" / self.date / "masks"

    @property
    def sam2_mask_export_manifest(self) -> Path:
        try:
            return get_sam2_mask_export_manifest(self.data_root, self.date)
        except Exception:
            return Path(str(self.data_root)) / "sam2_pipeline_files" / "exported_masks" / self.date / f"mask_export_manifest_{self.date}.json"

    @property
    def sam2_experiment_metadata_json(self) -> Path:
        try:
            return get_experiment_metadata_json(self.data_root, self.date)
        except Exception:
            return Path(str(self.data_root)) / "sam2_pipeline_files" / "raw_data_organized" / f"experiment_metadata_{self.date}.json"

    # ——— Stage 3: Downstream file tracking ————————————————————————————
    
    def is_in_df01(self) -> bool:
        """
        Check if this experiment exists in df01.

        Robust detection:
        - If `experiment_date` column exists, check exact match against self.date.
        - Otherwise, derive experiment ID from `embryo_id`/`video_id` by dropping the
          trailing well/embryo tokens, e.g.,
              20250529_30hpf_ctrl_atf6_A01_e01 -> 20250529_30hpf_ctrl_atf6
        """
        try:
            import pandas as pd
            df01_path = self.data_root / "metadata" / "combined_metadata_files" / "embryo_metadata_df01.csv"
            if not df01_path.exists():
                return False
            df01 = pd.read_csv(df01_path)

            # Preferred explicit column
            if 'experiment_date' in df01.columns:
                vals = df01['experiment_date'].astype(str)
                return self.date in set(vals.values)

            # Fallback: derive experiment id from embryo/video identifiers
            def derive_experiment(series):
                parts = series.astype(str).str.split('_')
                # Drop last two tokens (well, embryo index) when present
                return parts.apply(lambda p: '_'.join(p[:-2]) if len(p) >= 3 else p[0])

            for col in ("embryo_id", "EmbryoID", "embryoID", "video_id", "VideoID", "videoID"):
                if col in df01.columns:
                    exps = set(derive_experiment(df01[col]).values)
                    return self.date in exps
            
            # Final fallback: stream-scan for pattern in the CSV text
            patt = f"{self.date}_"
            try:
                with open(df01_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if patt in line or self.date in line:
                            return True
            except Exception:
                pass
            return False
        except Exception:
            # If pandas fails, do the same streaming fallback
            try:
                df01_path = self.data_root / "metadata" / "combined_metadata_files" / "embryo_metadata_df01.csv"
                if not df01_path.exists():
                    return False
                patt = f"{self.date}_"
                with open(df01_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if patt in line or self.date in line:
                            return True
            except Exception:
                return False
            return False

    def is_in_df02(self) -> bool:
        """
        Check if this experiment exists in df02.

        Preferred: `experiment_id` (present in many df02 builds).
        Fallbacks: `experiment_date`, or derive from `embryo_id`/`video_id` like df01.
        """
        try:
            import pandas as pd
            df02_path = self.data_root / "metadata" / "combined_metadata_files" / "embryo_metadata_df02.csv"
            if not df02_path.exists():
                return False
            df02 = pd.read_csv(df02_path)

            if 'experiment_id' in df02.columns:
                vals = df02['experiment_id'].astype(str)
                return self.date in set(vals.values)
            if 'experiment_date' in df02.columns:
                vals = df02['experiment_date'].astype(str)
                return self.date in set(vals.values)

            def derive_experiment(series):
                parts = series.astype(str).str.split('_')
                return parts.apply(lambda p: '_'.join(p[:-2]) if len(p) >= 3 else p[0])

            for col in ("embryo_id", "EmbryoID", "embryoID", "video_id", "VideoID", "videoID"):
                if col in df02.columns:
                    exps = set(derive_experiment(df02[col]).values)
                    return self.date in exps
            
            # Final fallback: stream-scan the CSV for a simple pattern
            patt = f"{self.date}_"
            try:
                with open(df02_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if patt in line or self.date in line:
                            return True
            except Exception:
                pass
            return False
        except Exception:
            try:
                df02_path = self.data_root / "metadata" / "combined_metadata_files" / "embryo_metadata_df02.csv"
                if not df02_path.exists():
                    return False
                patt = f"{self.date}_"
                with open(df02_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if patt in line or self.date in line:
                            return True
            except Exception:
                return False
            return False

    def is_in_df03(self) -> bool:
        """
        Check if this experiment exists in the final dataset (df03).
        
        df03 is created by Build06 and contains df02 data merged with latent embeddings.
        This is the final, analysis-ready dataset.
        
        Returns:
            bool: True if experiment_date appears in df03.csv
        """
        try:
            import pandas as pd
            df03_path = self.data_root / "metadata" / "combined_metadata_files" / "embryo_metadata_df03.csv"
            if not df03_path.exists():
                return False
            df03 = pd.read_csv(df03_path)
            return self.date in df03['experiment_date'].values if 'experiment_date' in df03.columns else False
        except Exception:
            return False

    def needs_build06_merge(self, model_name: str = "20241107_ds_sweep01_optimum") -> bool:
        """
        Determine if THIS specific experiment needs to be merged in Build06.

        This is the most precise check for Build06 requirements. An experiment needs
        merging if its latent embeddings are newer than the current df03 file.

        Logic:
        - If no df03 exists → needs merge (if has latents)
        - If latent embeddings are newer than df03 → needs merge
        - If latent embeddings are older than df03 → already merged

        This avoids unnecessary rebuilds when experiments are already current in df03.

        Args:
            model_name: Model name for latent embeddings (default: latest model)

        Returns:
            bool: True if this experiment's latents need merging into df03
        """
        try:
            df03_path = self.data_root / "metadata" / "combined_metadata_files" / "embryo_metadata_df03.csv"

            # If no df03 exists, needs merge if we have latents
            if not df03_path.exists():
                return self.has_latents(model_name)

            # Key insight: Compare latent timestamp vs df03 timestamp
            if self.has_latents(model_name):
                latent_path = self.get_latent_path(model_name)
                df03_time = df03_path.stat().st_mtime
                latent_time = latent_path.stat().st_mtime
                return latent_time > df03_time  # Latents newer = needs merge

            return False
        except Exception:
            return False

    def needs_build04(self) -> bool:
        """
        Determine if per-experiment Build04 needs to run for this experiment.

        Per-experiment Build04 processes the experiment's Build03 output through
        quality control and staging to produce a per-experiment df02 file.

        Logic:
        - If Build03 output doesn't exist → False (no input available)
        - If Build04 output doesn't exist → True (needs processing)
        - If Build03 output is newer than Build04 output → True (stale output)

        Returns:
            bool: True if per-experiment Build04 needs to run
        """
        try:
            build03_file = self.build03_path
            build04_file = self.build04_path

            # No input available
            if not build03_file.exists():
                return False

            # Output missing
            if not build04_file.exists():
                return True

            # Check if input is newer than output
            return build03_file.stat().st_mtime > build04_file.stat().st_mtime

        except Exception:
            return False

    def needs_build06_per_experiment(self, model_name: str = "20241107_ds_sweep01_optimum") -> bool:
        """Check if per-experiment Build06 needs to run."""
        try:
            # Missing per-experiment output
            if not self.build06_path.exists():
                return True

            # Build04 per-experiment newer than Build06 per-experiment
            if self.build04_path.exists() and self.build04_path.stat().st_mtime > self.build06_path.stat().st_mtime:
                return True

            # Latents newer than Build06 per-experiment
            latents_path = self.get_latent_path(model_name)
            if latents_path.exists() and latents_path.stat().st_mtime > self.build06_path.stat().st_mtime:
                return True

            return False
        except Exception:
            return False

    # ——— Stage 3: Pipeline step requirements ———————————————————————

    @property
    def needs_sam2(self) -> bool:
        """
        Determine if SAM2 segmentation needs to run for this experiment.

        Simple, robust rule for containerization:
        - Run if the SAM2 CSV is missing, OR
        - Run if stitched FF inputs are newer than the SAM2 CSV (stale outputs).

        This avoids complex dependency checks while ensuring we refresh SAM2 when
        upstream images change.

        Returns:
            bool: True if SAM2 should run
        """
        try:
            csv_path = self.sam2_csv_path
            # Missing CSV => need to run
            if not csv_path.exists():
                return True

            # Freshness check: if stitched FF (or FF) inputs are newer than CSV, rerun
            try:
                # Prefer stitched FF path; fall back to FF path
                in_base = self.stitch_ff_path or self.ff_path
                if in_base is None:
                    return False  # No inputs to compare; treat as up-to-date

                # Choose appropriate pattern key for newest_mtime
                patt_key = "stitch" if (self.stitch_ff_path and in_base == self.stitch_ff_path) else "ff"
                input_time = newest_mtime(in_base, PATTERNS[patt_key])
                csv_time = csv_path.stat().st_mtime
                if input_time > csv_time:
                    return True

                # Built metadata is injected into the SAM2 JSON/CSV outputs, so treat
                # metadata edits as stale SAM2 results and rerun when detected.
                meta_csv = self.meta_path_built
                if meta_csv is not None and meta_csv.exists():
                    meta_time = meta_csv.stat().st_mtime
                    if meta_time > csv_time:
                        return True
                return False
            except Exception:
                # On failure to compare, don't force rerun
                return False
        except Exception:
            return False

    @property
    def state_file(self) -> Path:
        p = self.data_root/"metadata"/"experiments"/f"{self.date}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    
    @property
    def needs_export(self) -> bool:
        # last_run = self.timestamps.get("export", 0)
        # newest   = newest_mtime(self.raw_path, PATTERNS["raw"])
        return not self.flags["ff"]

    @property
    def needs_metadata_export(self) -> bool:
        """Return True when the built metadata CSV is missing or stale."""
        try:
            # Missing CSV → need to rebuild
            meta_csv = self.meta_path_built
            if meta_csv is None:
                return True

            # Compare metadata timestamp to freshest inputs we have
            input_times: list[float] = []
            try:
                if self.ff_path and has_output(self.ff_path, PATTERNS["ff"]):
                    mt = newest_mtime(self.ff_path, PATTERNS["ff"]) or 0.0
                    if mt:
                        input_times.append(mt)
            except Exception:
                pass

            try:
                if self.raw_path and has_output(self.raw_path, PATTERNS["raw"]):
                    mt = newest_mtime(self.raw_path, PATTERNS["raw"]) or 0.0
                    if mt:
                        input_times.append(mt)
            except Exception:
                pass

            try:
                excel_paths: list[Path] = []
                if self.meta_path:
                    excel_paths.append(self.meta_path)

                expected_name = f"{self.date}_well_metadata.xlsx"
                for base in (
                    self.repo_root / "metadata" / "well_metadata",
                    self.repo_root / "metadata" / "plate_metadata",
                ):
                    try:
                        if not base.exists():
                            continue
                        candidate = base / expected_name
                        if candidate.exists():
                            excel_paths.append(candidate)
                            break
                        variants = sorted(base.glob(f"{self.date}_well_metadata*.xlsx"))
                        if variants:
                            excel_paths.extend(variants)
                            break
                    except Exception:
                        continue

                for excel in excel_paths:
                    try:
                        mt = excel.stat().st_mtime
                        if mt:
                            input_times.append(mt)
                    except Exception:
                        continue
            except Exception:
                pass

            if not input_times:
                return False  # No usable upstream timestamp; assume up-to-date

            return max(input_times) > meta_csv.stat().st_mtime
        except Exception:
            return False

    # @property
    # def needs_build_metadata(self) -> bool:
    #     return (_mod_time(self.ff_path) >
    #             datetime.fromisoformat(self.timestamps.get("metadata", "1970-01-01T00:00:00")).timestamp())

    @property
    def needs_stitch(self) -> bool:
        last_run = self.timestamps.get("stitch", 0)
        newest   = newest_mtime(self.raw_path, PATTERNS["raw"])
        return newest >= last_run

    @property
    def needs_stitch_z(self) -> bool:
        last_run = self.timestamps.get("stitch_z", 0)
        newest   = newest_mtime(self.raw_path, PATTERNS["raw"])
        return newest >= last_run

    @property
    def needs_segment(self) -> bool:
        # If any of the 5 QC mask outputs are missing, we need to run Build02
        try:
            present, total = self.qc_mask_status()
            if total > 0 and present < total:
                return True
        except Exception:
            # Be conservative: if we can't determine, fall back to timestamps
            pass
        # Otherwise use freshness to decide if an update is needed
        return (
            _mod_time(self.mask_path)
            > self._ts("segment", 0.0)
        )
    
    @property
    def needs_stats(self) -> bool:
        last_run = newest_mtime(self.mask_path, PATTERNS["snips"])
        newest   = newest_mtime(self.snip_path, PATTERNS["segment"])
        return newest >= last_run

    @property
    def needs_build03(self) -> bool:
        """
        Determine if Build03 needs to run based solely on per‑experiment outputs.

        Policy (df01 deprecated):
        - Inputs: SAM2 CSV if present; otherwise legacy QC masks.
        - If no inputs available → False.
        - If per‑experiment Build03 CSV exists and is newer than inputs → False.
        - Otherwise → True (missing or stale output).
        """
        try:
            # Determine inputs and their freshness timestamp
            inputs_available = False
            input_time = 0.0
            if self.sam2_csv_path.exists():
                inputs_available = True
                input_time = self.sam2_csv_path.stat().st_mtime
            elif self.has_all_qc_masks:
                inputs_available = True
                try:
                    input_time = newest_mtime(self.mask_path, PATTERNS["segment"]) or 0.0
                except Exception:
                    input_time = 0.0

            # Incorporate built metadata timestamp (genotype/treatment updates)
            try:
                meta_csv = self.meta_path_built
                if meta_csv is not None and meta_csv.exists():
                    input_time = max(input_time, meta_csv.stat().st_mtime)
                    inputs_available = True
            except Exception:
                pass

            if not inputs_available:
                return False

            b03 = self.build03_path
            if b03.exists():
                b03_time = b03.stat().st_mtime
                return b03_time < input_time

            # No per-exp output yet → needs run
            return True
        except Exception:
            return False

    # ——— internal sync logic —————————————————————————————————————————————
    def _sync_with_disk(self) -> None:
        """Called once from __post_init__ to refresh flags + timestamps."""
        changed = False
        for step, path in {
            "raw"    : self.raw_path,
            "meta"   : self.meta_path,
            "meta_built" : self.meta_path_built,
            "meta_embryo": self.meta_path_embryo,
            "ff"     : self.ff_path,
            "stitch" : self.stitch_ff_path,
            "stitch_z" : self.stitch_z_path,
            "segment": self.mask_path,
            "snips": self.snip_path,
            }.items():
            
            if step not in ["meta", "meta_built", "meta_embryo"]:
                if step == "ff":
                    present = self._ff_complete()
                else:
                    present = has_output(path, PATTERNS[step])
            else:
                present = path is not None

            previous = self.flags.get(step, None)

            # flag housekeeping ---------------------------------------------------
            if present != previous:
                self.flags[step] = present
                changed = True

            # timestamp housekeeping ---------------------------------------------
            if present:
                if step not in ["meta", "meta_built", "meta_embryo"]:
                    mt = newest_mtime(path, PATTERNS[step])
                else:
                    mt = path.stat().st_mtime
                self.timestamps[step] = mt
            else:
                self.timestamps.pop(step, None)

        if changed:
            self._save_state()


    # ——— call pipeline functions —————————————————————————————————————————————
    @record("ff")
    def export_images(self):
        import os
        def _as_bool(val: str) -> bool:
            return str(val).lower() in ("1", "true", "yes", "on")
        overwrite_env = _as_bool(os.environ.get("MSEQ_OVERWRITE_BUILD01", "0"))
        if self.microscope == "Keyence":
            # Default resume-by-skip; enable full recompute with MSEQ_OVERWRITE_BUILD01=1
            build_ff_from_keyence(data_root=self.data_root, repo_root=self.repo_root, exp_name=self.date, overwrite=overwrite_env)
        else:
            # For YX1, overwrite=False skips existing frames; set MSEQ_OVERWRITE_BUILD01=1 to recompute
            build_ff_from_yx1(data_root=self.data_root, repo_root=self.repo_root, exp_name=self.date, overwrite=overwrite_env)

    @record("meta_built")
    def export_metadata(self):
        if self.microscope == "Keyence":
            build_ff_from_keyence(data_root=self.data_root, exp_name=self.date, overwrite=True, metadata_only=True)
        else:
            build_ff_from_yx1(
                data_root=self.data_root,
                repo_root=self.repo_root,
                exp_name=self.date,
                overwrite=True,
                metadata_only=True,
            )


    @record("stitch")
    def stitch_images(self):
        import os
        def _as_bool(val: str) -> bool:
            return str(val).lower() in ("1", "true", "yes", "on")
        overwrite_stitch = _as_bool(os.environ.get("MSEQ_OVERWRITE_STITCH", "0"))
        skip_z_stitch = _as_bool(os.environ.get("MSEQ_SKIP_Z_STITCH", "0"))
        if self.microscope == "Keyence":
            # Default is resume mode; set MSEQ_OVERWRITE_STITCH=1 to force restitch
            print("       ↳ Build01: FF stitch (Keyence) — required for SAM2")
            stitch_ff_from_keyence(data_root=self.data_root, exp_name=self.date, overwrite=overwrite_stitch, n_workers=self.num_cpu_workers)
            print("       ↳ Build01: FF stitch complete")
            if not skip_z_stitch:
                print("       ↳ Build01: Z-stitch (Keyence) — optional, not required for SAM2")
                stitch_z_from_keyence(data_root=self.data_root, exp_name=self.date, overwrite=True, n_workers=self.num_cpu_workers)
                print("       ↳ Build01: Z-stitch complete")
        else:
            pass

    # @record("stitch")
    def stitch_z_images(self):
        if self.microscope == "Keyence":
            # stitch_ff_from_keyence(data_root=self.data_root, exp_name=self.date, overwrite=True, n_workers=self.num_cpu_workers)
            stitch_z_from_keyence(data_root=self.data_root, exp_name=self.date, overwrite=True, n_workers=self.num_cpu_workers)
        else:
            pass

    @record("segment")
    def segment_images(self, force_update: bool=False):
        # Import here to avoid pulling heavy Build02 deps unless needed.
        from src.build.build02B_segment_bf_main import apply_unet
        # We need to pull the current models
        model_name_vec = ["mask_v0_0100", "via_v1_0100", "yolk_v1_0050", "focus_v0_0100", "bubble_v0_0100"]
        # apply unet for each model
        # Use a conservative worker count to avoid system warnings/freezes
        safe_workers = min(2, self.num_cpu_workers)
        for model_name in model_name_vec:
            apply_unet(
                root=self.data_root,
                model_name=model_name,
                n_classes=1,
                checkpoint_path=None,  # use the latest checkpoint
                n_workers=safe_workers,
                overwrite_flag=force_update,
                make_sample_figures=True,
                n_sample_figures=100,
                segment_list=[self.date],  # Only process this experiment
            )

    # @record()
    def process_image_masks(self, force_update: bool=False):
        tracked_df = segment_wells(root=self.data_root, exp_name=self.date)
        stats_df = compile_embryo_stats(root=self.data_root, tracked_df=tracked_df)
        extract_embryo_snips(root=self.data_root, stats_df=stats_df, overwrite_flag=force_update)

    # ——— Stage 3: Orchestration execution methods ——————————————————————

    @record("sam2")
    def run_sam2(self, workers: int = 8, **kwargs):
        """Execute SAM2 segmentation for this experiment"""
        # Import here to avoid circular dependencies
        from ..run_morphseq_pipeline.steps.run_sam2 import run_sam2
        print(f"🎯 Running SAM2 for {self.date}")
        result = run_sam2(
            root=str(self.data_root), 
            exp=self.date, 
            workers=workers, 
            **kwargs
        )
        return result

    @record("build03")
    def run_build03(self, by_embryo: int = None, frames_per_embryo: int = None, overwrite: bool = False, **kwargs):
        """Execute Build03 for this experiment with SAM2/legacy detection"""
        print(f"🔬 Running Build03 for {self.date}")
        
        # Import here to avoid circular dependencies
        from ..run_morphseq_pipeline.steps.run_build03 import run_build03 as run_build03_step
        
        # Determine which path to use
        sam2_csv = None
        if self.sam2_csv_path.exists():
            print(f"  Using SAM2 masks from {self.sam2_csv_path}")
            sam2_csv = str(self.sam2_csv_path)
        else:
            print(f"  Using legacy Build02 masks")
            # Check if legacy masks exist
            if not self.has_all_qc_masks:
                raise RuntimeError(f"No SAM2 CSV and missing QC masks for {self.date}")
        
        # Call the actual Build03 function with proper parameters
        try:
            # Build argument dict while avoiding overriding defaults with None
            _args = dict(
                root=str(self.data_root),
                exp=self.date,
                sam2_csv=sam2_csv,  # Will be None for legacy path
                by_embryo=by_embryo,
                frames_per_embryo=frames_per_embryo,
                n_workers=kwargs.get('n_workers', self.num_cpu_workers),
                overwrite=overwrite,
            )
            if 'df01_out' in kwargs and kwargs['df01_out'] is not None:
                _args['df01_out'] = kwargs['df01_out']

            result = run_build03_step(**_args)
            
            # Update df01 contribution tracking on successful return
            if result is not None:
                self.flags['contributed_to_df01'] = True
                self.timestamps['last_df01_contribution'] = datetime.utcnow().isoformat()
                
            return result
            
        except Exception as e:
            print(f"  ❌ Build03 failed: {e}")
            raise

    @record("latents")
    def generate_latents(self, model_name: str = "20241107_ds_sweep01_optimum", **kwargs):
        """Generate latent embeddings for this experiment"""
        from ..analyze.gen_embeddings import ensure_embeddings_for_experiments
        print(f"🧬 Generating latents for {self.date}")
        success = ensure_embeddings_for_experiments(
            data_root=str(self.data_root),
            experiments=[self.date],
            model_name=model_name,
            **kwargs
        )
        return success

    @record("build04")
    def run_build04_per_experiment(self, **kwargs):
        """Execute per-experiment Build04 (QC + staging → df02)."""
        print(f"🧪 Running Build04 per-experiment for {self.date}")
        from ..run_morphseq_pipeline.steps.run_build04 import run_build04 as run_build04_step
        try:
            out_path = run_build04_step(
                root=str(self.data_root),
                exp=self.date,
                **kwargs
            )
            print(f"  ✅ Build04 complete: {out_path}")
            return out_path
        except Exception as e:
            print(f"  ❌ Build04 failed: {e}")
            raise

    @record("build06_per_experiment")
    def run_build06_per_experiment(self, model_name: str = "20241107_ds_sweep01_optimum", **kwargs):
        """Execute per-experiment Build06 (df02 + latents -> df03)"""
        print(f"🧬 Running Build06 per-experiment for {self.date}")

        # Import here to avoid circular dependencies
        from ..run_morphseq_pipeline.steps.run_build06_per_exp import build06_merge_per_experiment

        try:
            result = build06_merge_per_experiment(
                root=Path(self.data_root),
                exp=self.date,
                model_name=model_name,
                verbose=kwargs.get('verbose', False),
                generate_missing=kwargs.get('generate_missing', True),
                overwrite=kwargs.get('overwrite', False),
                dry_run=kwargs.get('dry_run', False)
            )

            print(f"  ✅ Per-experiment df03 created: {result}")
            return result

        except Exception as e:
            print(f"  ❌ Build06 per-experiment failed: {e}")
            raise

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
    """
    Orchestrates multiple MorphSeq experiments and manages global pipeline operations.
    
    The ExperimentManager provides intelligent coordination between individual experiments
    and global processing steps. It handles both per-experiment operations and cohort-level
    data processing that spans multiple experiments.
    
    Key Responsibilities:
    1. **Experiment Discovery**: Auto-discovers experiments from raw_image_data structure
    2. **Global File Management**: Tracks combined metadata files (df01, df02, df03)
    3. **Intelligent Orchestration**: Determines what processing is needed across the cohort
    4. **Dependency Management**: Ensures correct order of per-experiment vs global steps
    
    Global Pipeline Flow:
    Per-experiment: [Raw → FF → QC/SAM2 → Build03 → Latents] 
                                     ↓
    Global: [df01] → Build04 → [df02] → Build06 → [df03]
    
    Combined Files:
    - df01: Raw embryo data from all Build03 runs (per-experiment contributions)
    - df02: QC'd + staged embryo data from Build04 (global processing)  
    - df03: Final dataset with embeddings from Build06 (global merge)
    
    Attributes:
        root: Path to MorphSeq data root directory
        exp_dir: Path to experiment state files (metadata/experiments/)
        experiments: Dict mapping experiment dates to Experiment objects
    """
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.exp_dir = self.root / "metadata" / "experiments"
        self.experiments: dict[str, Experiment] = {}
        self.discover_experiments()
        self.update_experiment_status()

    # ——— Global / Non‑Generated Inputs ——————————————————————————————
    @property
    def stage_ref_csv(self) -> Path:
        """Global stage reference CSV path."""
        try:
            return get_stage_ref_csv(self.root)
        except Exception:
            return self.root / "metadata" / "stage_ref_df.csv"

    @property
    def perturbation_key_csv(self) -> Path:
        """Global perturbation name key CSV path."""
        try:
            return get_perturbation_key_csv(self.root)
        except Exception:
            return self.root / "metadata" / "perturbation_name_key.csv"

    def get_well_metadata_xlsx(self, exp: str) -> Path:
        """Per‑experiment well metadata Excel path."""
        try:
            return get_well_metadata_xlsx(self.root, exp)
        except Exception:
            return self.root / "metadata" / "well_metadata" / f"{exp}_well_metadata.xlsx"

    # ——— Global File Management: Combined Metadata Files ————————————————

    @property
    def df01_path(self) -> Path:
        """
        Path to embryo_metadata_df01.csv - the raw combined embryo dataset.
        
        df01 contains embryo-level data from all experiments that have completed Build03.
        Each row represents one embryo at one timepoint, with morphological measurements
        and metadata. This file grows as more experiments complete Build03.
        
        Created by: Build03 (appends per-experiment data)
        Used by: Build04 input
        """
        return self.root / "metadata" / "combined_metadata_files" / "embryo_metadata_df01.csv"

    @property
    def df02_path(self) -> Path:
        """
        Path to embryo_metadata_df02.csv - the QC'd and staged embryo dataset.
        
        df02 is created by Build04 and contains df01 data after quality control,
        outlier removal, and developmental stage inference. This is a cleaned,
        analysis-ready version of df01.
        
        Created by: Build04 (global QC processing)
        Used by: Build06 input
        """
        return self.root / "metadata" / "combined_metadata_files" / "embryo_metadata_df02.csv"

    @property
    def df03_path(self) -> Path:
        """
        Path to embryo_metadata_df03.csv - the final analysis-ready dataset.
        
        df03 is created by Build06 and contains df02 data merged with latent
        morphological embeddings. This is the final dataset used for downstream
        analysis and machine learning applications.
        
        Created by: Build06 (global merge with embeddings)
        Used by: Analysis and ML workflows
        """
        return self.root / "metadata" / "combined_metadata_files" / "embryo_metadata_df03.csv"

    @property
    def needs_build04(self) -> bool:
        """
        DEPRECATED: Use per-experiment exp.needs_build04() instead.

        Legacy global Build04 check for backwards compatibility.
        The new architecture uses per-experiment Build04 processing.

        Build04 now processes per-experiment Build03 output through quality control
        and staging to produce per-experiment df02 files, rather than global processing.

        Returns:
            bool: True if Build04 global QC needs to run (legacy)
        """
        import warnings
        warnings.warn(
            "ExperimentManager.needs_build04 is deprecated. Use per-experiment "
            "exp.needs_build04() for the new per-experiment architecture.",
            DeprecationWarning,
            stacklevel=2
        )

        # Can't run without input data
        if not self.df01_path.exists():
            return False
        # Output missing, needs to run
        if not self.df02_path.exists():
            return True
        # Check if input is newer than output
        try:
            return self.df01_path.stat().st_mtime > self.df02_path.stat().st_mtime
        except Exception:
            return False

    @property
    def needs_build06(self) -> bool:
        """
        DEPRECATED: Use per-experiment exp.needs_build06_per_experiment() instead.

        Legacy global Build06 check for backwards compatibility.
        The new architecture uses per-experiment Build06 processing.

        Build06 now processes per-experiment Build04 output with latents to produce
        per-experiment df03 files, rather than global merge processing.

        Returns:
            bool: True if Build06 global merge needs to run (legacy)
        """
        import warnings
        warnings.warn(
            "ExperimentManager.needs_build06 is deprecated. Use per-experiment "
            "exp.needs_build06_per_experiment() for the new per-experiment architecture.",
            DeprecationWarning,
            stacklevel=2
        )

        if not self.df02_path.exists():
            return False
        if not self.df03_path.exists():
            return True

        try:
            # Get the set of experiments in df02 vs df03
            import pandas as pd

            df02 = pd.read_csv(self.df02_path)
            df03 = pd.read_csv(self.df03_path)

            # Get unique experiment dates from each file
            df02_experiments = set(df02['experiment_date'].unique()) if 'experiment_date' in df02.columns else set()
            df03_experiments = set(df03['experiment_date'].unique()) if 'experiment_date' in df03.columns else set()

            # Check if there are experiments in df02 that aren't in df03
            missing_from_df03 = df02_experiments - df03_experiments
            if missing_from_df03:
                return True

            # Check if any experiment's latent files are newer than df03
            # (but only for experiments that should be in the final dataset)
            df03_time = self.df03_path.stat().st_mtime
            for exp in self.experiments.values():
                if exp.date in df02_experiments and exp.has_latents():
                    latent_path = exp.get_latent_path()
                    if latent_path.exists() and latent_path.stat().st_mtime > df03_time:
                        return True

            return False
        except Exception as e:
            # Fallback to simple timestamp comparison if DataFrame operations fail
            try:
                return self.df02_path.stat().st_mtime > self.df03_path.stat().st_mtime
            except Exception:
                return False

    # ——— Stage 3: Global orchestration methods ——————————————————————————

    def run_build04(self, **kwargs):
        """Execute global Build04 step (df01 -> df02)"""
        print("🔄 Running Build04 (global QC + staging)")
        from ..run_morphseq_pipeline.steps.run_build04 import run_build04
        return run_build04(root=str(self.root), **kwargs)

    def run_build06(self, model_name: str = "20241107_ds_sweep01_optimum", **kwargs):
        """Execute global Build06 step (df02 + latents -> df03)"""
        print("🔄 Running Build06 (global embeddings merge)")
        from ..run_morphseq_pipeline.steps.run_build06 import run_build06
        return run_build06(
            # In two-root mode, `root` is the repo root (for snips); `data_root` is the central data dir
            root=str(self.root.parent.parent if "data" in str(self.root) else self.root),
            data_root=str(self.root),
            model_name=model_name,
            **kwargs
        )


    def discover_experiments(self):
        # scan "raw_image_data" subfolders for dates
        raw = self.root / "raw_image_data"
        # collect all the dates first
        dates = []
        for mic in raw.iterdir():
            if (not mic.is_dir()) or (mic.name.lower()=="ignore"): 
                continue
            for d in mic.iterdir():
                if not d.is_dir():
                    continue
                name = d.name
                # Skip common ignore patterns and hidden/temp dirs
                if name.startswith('.') or name.startswith('_') or name.lower().startswith('ignore'):
                    continue
                dates.append(name)
                    
        # dedupe & sort
        for date in sorted(set(dates)):
            self.experiments[date] = Experiment(date, self.root)

    # Helper to update all experiments to reflect presence/absence of files on disk
    def update_experiment_status(self):
        for exp in self.experiments.values():
            exp._sync_with_disk()

    def export_all(self):
        for exp in self.experiments.values():
            if exp.needs_export:
                try:
                    exp.export_images()
                except:
                    log.exception("Export & FF build failed for %s", exp.date)
    
    def export_all_metadata(self):
        for exp in self.experiments.values():
            try:
                exp.export_metadata()
            except:
                log.exception("Metadata build failed for %s", exp.date)

    def stitch_all(self):
        for exp in self.experiments.values():
            if exp.needs_stitch:
                try:
                    exp.stitch_images()
                except:
                    log.exception("Stitching  failed for %s", exp.date) 


    def _run_step(
        self,
        step: str,                                   # name of Experiment method to call
        need_attr: str,                              # the corresponding “needs_*” flag
        *,
        experiments: list[str] | None = None,
        later_than: int | None = None,
        earlier_than: int = 99_999_999,
        force_update: bool = False,
        extra_filter: callable[[Experiment], bool] | None = None,
        friendly_name: str | None = None,            # text used in printouts
    ) -> None:
        """Find experiments that should run *step* and call it.

        extra_filter(exp) → bool lets you add step-specific constraints
        (e.g. microscope == "Keyence").  Leave it None if not needed.
        """
        # 0) sanity check ------------------------------------------------------
        if (experiments is None) == (later_than is None):
            raise ValueError("pass *either* experiments or later_than (not both)")

        # 1) pick the candidates ----------------------------------------------
        selected, dates = [], []
        for exp in self.experiments.values():
            # --- user-provided subset
            if experiments is not None and exp.date not in experiments:
                continue
            # --- date window
            if experiments is None:
                try:
                    di = int(exp.date[:8])
                except ValueError:
                    continue
                if not (later_than <= di < earlier_than):
                    continue
            # --- “needs_*” or forced
            if not getattr(exp, need_attr) and not force_update:
                continue
            # --- custom predicate
            if extra_filter and not extra_filter(exp):
                continue

            selected.append(exp)
            dates.append(exp.date)

        if not selected:
            print(f"No experiments to {friendly_name or step}.")
            return

        print(f"{friendly_name or step.capitalize()}:", ", ".join(sorted(dates)))

        # 2) run the step ------------------------------------------------------
        for exp in selected:
            try:
                getattr(exp, step)()          # e.g. exp.export_images()
            except Exception as e:
                log.exception("❌  %s failed for %s", step, exp.date)

    # ──────────────────────────────────────────────────────────────────────────
    # thin façade methods
    def export_experiments(self, **kwargs):
        self._run_step(
            "export_images", "needs_export",
            friendly_name="export",
            **kwargs
        )

    # thin façade methods
    def export_experiment_metadata(self, **kwargs):
        self._run_step(
            "export_metadata", "needs_metadata_export",
            friendly_name="metadata export",
            **kwargs
        )

    def stitch_experiments(self, **kwargs):
        self._run_step(
            "stitch_images", "needs_stitch",
            extra_filter=lambda e: e.microscope == "Keyence",
            friendly_name="stitch",
            **kwargs
        )

    def stitch_z_experiments(self, **kwargs):
        self._run_step(
            "stitch_z_images", "needs_stitch_z",
            extra_filter=lambda e: e.microscope == "Keyence",
            friendly_name="stitch_z",
            **kwargs
        )

    def segment_experiments(self, **kwargs):
        self._run_step(
            "segment_images", "needs_segment",
            # extra_filter=lambda e: e.microscope == "Keyence",
            friendly_name="segment",
            **kwargs
        )

    def get_embryo_stats(self, **kwargs):
        self._run_step(
            "process_image_masks", "needs_stats",
            friendly_name="mask_stats",
            **kwargs
        )

    def build06_per_experiments(self, model_name: str = "20241107_ds_sweep01_optimum", **kwargs):
        """Run Build06 per-experiment for experiments that need it."""
        self._run_step(
            "run_build06_per_experiment",
            "needs_build06_per_experiment",  # Use string attribute name
            friendly_name="build06_per_experiment",
            extra_filter=lambda e: e.needs_build06_per_experiment(model_name),
            **kwargs
        )

    # def build_metadata_all(self):
    #     for exp in self.experiments.values():
    #         if exp.needs_build_metadata():
    #             exp.build_image_metadata()

    def report(self):
        for date, exp in self.experiments.items():
            print(f"{date}: raw={exp.flags['raw']}, meta={exp.flags['meta']}, ff={exp.flags['ff']}, stitch={exp.flags['stitch']}, stitch_z={exp.flags['stitch_z']}, segment={exp.flags['segment']}")

    # TODO: Add pipeline status summary functionality
    # Current dry-run shows individual experiment needs but doesn't provide aggregate statistics.
    # Future enhancement: Add methods like:
    # - summary_report(): Show counts of experiments at each pipeline stage
    # - pipeline_status_table(): Tabular view of all experiments and their completion status
    # - completion_stats(): Percentages of experiments that have completed each stage
    # - bottleneck_analysis(): Identify which stages are blocking the most experiments
    # 
    # Example desired output:
    # Pipeline Status Summary:
    # - Total experiments: 102
    # - FF images complete: 95 (93%)
    # - QC masks complete: 67 (66%)  
    # - SAM2 complete: 23 (23%)
    # - Build03 complete: 78 (76%)
    # - Latents complete: 45 (44%)
    # - In df01: 78, In df02: 78, In df03: 45



if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Call pipeline functions
    root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")

    # load master experiment log
    manager = ExperimentManager(root=root)
    manager.report()

    exp = Experiment(date="20250703_chem3_34C_T01_1457", data_root=root)
    exp.process_image_masks()
    print("check")
