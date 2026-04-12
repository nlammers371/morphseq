"""Load latent trajectories for the beta particle predictor.

This module ports the useful trajectory-loading infrastructure from the
legacy dynamo namespace while keeping the output contract simple and aligned
with the particle-prediction docs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_BUILD_DIR = Path("morphseq_playground") / "metadata" / "build06_output"
DEFAULT_METADATA_COLUMNS: Tuple[str, ...] = (
    "background",
    "phenotype",
    "phenotype_label",
)


@dataclass(frozen=True)
class EmbryoTrajectory:
    """One embryo trajectory in PCA latent space."""

    embryo_id: str
    trajectory: np.ndarray
    time_seconds: np.ndarray
    delta_t: float
    temperature: float
    perturbation_class: str
    experiment_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    frame_index: Optional[np.ndarray] = None


@dataclass
class TrajectoryDataset:
    """Loaded trajectory collection plus PCA artifacts."""

    trajectories: List[EmbryoTrajectory]
    pca: PCA
    scaler: Optional[StandardScaler]
    z_mu_cols: List[str]
    class_to_idx: Dict[str, int] = field(default_factory=dict)
    build_dir: Path = DEFAULT_BUILD_DIR

    def __post_init__(self) -> None:
        if not self.class_to_idx:
            classes = sorted({traj.perturbation_class for traj in self.trajectories})
            self.class_to_idx = {name: index for index, name in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.trajectories)

    @property
    def n_components(self) -> int:
        return int(self.pca.n_components_)

    @property
    def class_names(self) -> List[str]:
        return sorted(self.class_to_idx, key=self.class_to_idx.get)

    def filter(
        self,
        experiment_ids: Optional[Sequence[str]] = None,
        perturbation_classes: Optional[Sequence[str]] = None,
    ) -> "TrajectoryDataset":
        trajectories = self.trajectories
        if experiment_ids is not None:
            allowed_experiments = set(experiment_ids)
            trajectories = [traj for traj in trajectories if traj.experiment_id in allowed_experiments]
        if perturbation_classes is not None:
            allowed_classes = set(perturbation_classes)
            trajectories = [traj for traj in trajectories if traj.perturbation_class in allowed_classes]
        return dataclasses.replace(self, trajectories=trajectories)


def _detect_z_mu_b_cols(df: pd.DataFrame) -> List[str]:
    cols = sorted(column for column in df.columns if column.startswith("z_mu_b"))
    if not cols:
        raise ValueError("No z_mu_b columns found in DataFrame")
    return cols


def _compute_experiment_delta_t(
    df: pd.DataFrame,
    dt_min: float = 720.0,
    dt_max: float = 5.0 * 3600.0,
) -> Dict[str, float]:
    dt_by_exp: Dict[str, float] = {}
    for exp_id, exp_df in df.groupby("experiment_id"):
        all_diffs: List[float] = []
        for _, emb_df in exp_df.groupby("embryo_id"):
            times = emb_df.sort_values("frame_index")["relative_time_s"].to_numpy(dtype=np.float64)
            if len(times) > 1:
                all_diffs.extend(np.diff(times).tolist())
        if not all_diffs:
            dt_by_exp[exp_id] = np.nan
            continue

        diffs_arr = np.asarray(all_diffs, dtype=np.float64)
        valid = (diffs_arr >= dt_min) & (diffs_arr <= dt_max)
        if valid.any():
            dt_by_exp[exp_id] = float(np.median(diffs_arr[valid]))
        else:
            dt_by_exp[exp_id] = float(np.median(diffs_arr))
    return dt_by_exp


def _repair_timestamps(
    times: np.ndarray,
    consensus_dt: float,
    dt_min: float = 720.0,
    dt_max: float = 5.0 * 3600.0,
) -> np.ndarray:
    if len(times) < 2 or not np.isfinite(consensus_dt):
        return times

    repaired = times.copy()
    diffs = np.diff(repaired)
    bad = (diffs < dt_min) | (diffs > dt_max)
    if not bad.any():
        return repaired

    for bad_index in np.where(bad)[0]:
        offset = repaired[bad_index] + consensus_dt - repaired[bad_index + 1]
        repaired[bad_index + 1 :] += offset
    return repaired


def _fit_pca(
    df: pd.DataFrame,
    z_mu_cols: List[str],
    n_components: int,
    scale: bool,
) -> Tuple[PCA, Optional[StandardScaler]]:
    x_raw = df[z_mu_cols].to_numpy(dtype=np.float64)
    valid = ~np.isnan(x_raw).any(axis=1)
    x_valid = x_raw[valid]
    if len(x_valid) == 0:
        raise ValueError("No valid embedding rows available for PCA fitting")

    scaler: Optional[StandardScaler] = None
    if scale:
        scaler = StandardScaler()
        x_valid = scaler.fit_transform(x_valid)

    pca = PCA(n_components=n_components)
    pca.fit(x_valid)
    return pca, scaler


def _project(x_raw: np.ndarray, pca: PCA, scaler: Optional[StandardScaler]) -> np.ndarray:
    x_project = scaler.transform(x_raw) if scaler is not None else x_raw
    return pca.transform(x_project)


def _resolve_perturbation_class(grp_sorted: pd.DataFrame) -> str:
    for column in ("genotype", "perturbation_class", "class_label"):
        if column in grp_sorted.columns:
            return str(grp_sorted[column].iloc[0])
    return "unknown"


def _extract_metadata(grp_sorted: pd.DataFrame, metadata_columns: Sequence[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for column in metadata_columns:
        if column not in grp_sorted.columns:
            continue
        value = grp_sorted[column].iloc[0]
        if pd.isna(value):
            continue
        metadata[column] = value.item() if hasattr(value, "item") else value
    return metadata


def load_trajectories(
    experiment_ids: Optional[Sequence[str]] = None,
    build_dir: str | Path = DEFAULT_BUILD_DIR,
    n_components: int = 10,
    scale: bool = True,
    min_trajectory_length: int = 15,
    verbose: bool = True,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
) -> TrajectoryDataset:
    """Load build06 trajectory CSVs and project them into PCA space."""

    build_dir = Path(build_dir)
    if experiment_ids is None:
        prefix = "df03_final_output_with_latents_"
        experiment_ids = sorted(
            path.stem[len(prefix) :]
            for path in build_dir.glob(f"{prefix}*.csv")
        )
        if verbose:
            print(f"  Auto-discovered {len(experiment_ids)} experiments in {build_dir}")
        if not experiment_ids:
            raise ValueError(f"No df03_final_output_with_latents_*.csv files found in {build_dir}")

    frames: List[pd.DataFrame] = []
    for exp_id in experiment_ids:
        path = build_dir / f"df03_final_output_with_latents_{exp_id}.csv"
        if not path.exists():
            if verbose:
                print(f"  [skip] {exp_id}: file not found")
            continue
        df_exp = pd.read_csv(path, low_memory=False)
        if "experiment_id" not in df_exp.columns:
            df_exp["experiment_id"] = exp_id
        frames.append(df_exp)
        if verbose:
            print(f"  Loaded {exp_id}: {len(df_exp)} rows")

    if not frames:
        raise ValueError("No experiment files could be loaded")

    df = pd.concat(frames, ignore_index=True)
    if "use_embryo_flag" in df.columns:
        n_before = len(df)
        df = df[df["use_embryo_flag"] == True].copy()  # noqa: E712
        if verbose:
            print(f"  Filtered use_embryo_flag: {n_before} -> {len(df)} rows")

    z_mu_cols = _detect_z_mu_b_cols(df)
    if verbose:
        print(f"  Found {len(z_mu_cols)} z_mu_b columns")

    valid_mask = ~df[z_mu_cols].isna().any(axis=1)
    df = df[valid_mask].copy()

    pca, scaler = _fit_pca(df, z_mu_cols, n_components=n_components, scale=scale)
    x_pc = _project(df[z_mu_cols].to_numpy(dtype=np.float64), pca, scaler)

    if verbose:
        explained = np.cumsum(pca.explained_variance_ratio_)[-1]
        print(f"  PCA: {n_components} components, {explained * 100:.1f}% variance explained")

    dt_by_exp = _compute_experiment_delta_t(df)
    trajectories: List[EmbryoTrajectory] = []
    n_repaired = 0

    grouped = df.groupby(["experiment_id", "embryo_id"])
    for (exp_id, embryo_id), grp in grouped:
        grp_sorted = grp.sort_values("frame_index")
        if len(grp_sorted) < min_trajectory_length:
            continue

        raw_times = grp_sorted["relative_time_s"].to_numpy(dtype=np.float64)
        repaired_times = _repair_timestamps(raw_times, dt_by_exp.get(exp_id, np.nan))
        if not np.array_equal(raw_times, repaired_times):
            n_repaired += 1

        row_indices = df.index.get_indexer(grp_sorted.index)
        trajectory = x_pc[row_indices]
        frame_index = None
        if "frame_index" in grp_sorted.columns:
            frame_index = grp_sorted["frame_index"].to_numpy(dtype=np.int64)

        temperature = np.nan
        if "temperature" in grp_sorted.columns:
            temperature = float(grp_sorted["temperature"].iloc[0])

        trajectories.append(
            EmbryoTrajectory(
                embryo_id=str(embryo_id),
                trajectory=trajectory,
                time_seconds=repaired_times,
                delta_t=float(dt_by_exp.get(exp_id, np.nan)),
                temperature=temperature,
                perturbation_class=_resolve_perturbation_class(grp_sorted),
                experiment_id=str(exp_id),
                metadata=_extract_metadata(grp_sorted, metadata_columns),
                frame_index=frame_index,
            )
        )

    if not trajectories:
        raise ValueError(
            f"No valid trajectories met min_trajectory_length={min_trajectory_length} in {build_dir}"
        )

    if verbose:
        dropped = len(grouped) - len(trajectories)
        lengths = [len(traj.trajectory) for traj in trajectories]
        print(
            f"  Built {len(trajectories)} trajectories "
            f"(dropped {dropped} with <{min_trajectory_length} frames)"
        )
        if n_repaired > 0:
            print(f"  Repaired timestamps in {n_repaired} trajectories")
        print(
            f"  Trajectory lengths: min={min(lengths)}, "
            f"median={int(np.median(lengths))}, max={max(lengths)}"
        )

    return TrajectoryDataset(
        trajectories=trajectories,
        pca=pca,
        scaler=scaler,
        z_mu_cols=z_mu_cols,
        build_dir=build_dir,
    )"""Load latent trajectories for the beta particle predictor.

This module ports the useful trajectory-loading infrastructure from the
legacy dynamo namespace while keeping the output contract simple and aligned
with the particle-prediction docs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_BUILD_DIR = Path("morphseq_playground") / "metadata" / "build06_output"
DEFAULT_METADATA_COLUMNS: Tuple[str, ...] = (
    "background",
    "phenotype",
    "phenotype_label",
)


@dataclass(frozen=True)
class EmbryoTrajectory:
    """One embryo trajectory in PCA latent space."""

    embryo_id: str
    trajectory: np.ndarray
    time_seconds: np.ndarray
    delta_t: float
    temperature: float
    perturbation_class: str
    experiment_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    frame_index: Optional[np.ndarray] = None


@dataclass
class TrajectoryDataset:
    """Loaded trajectory collection plus PCA artifacts."""

    trajectories: List[EmbryoTrajectory]
    pca: PCA
    scaler: Optional[StandardScaler]
    z_mu_cols: List[str]
    class_to_idx: Dict[str, int] = field(default_factory=dict)
    build_dir: Path = DEFAULT_BUILD_DIR

    def __post_init__(self) -> None:
        if not self.class_to_idx:
            classes = sorted({traj.perturbation_class for traj in self.trajectories})
            self.class_to_idx = {name: index for index, name in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.trajectories)

    @property
    def n_components(self) -> int:
        return int(self.pca.n_components_)

    @property
    def class_names(self) -> List[str]:
        return sorted(self.class_to_idx, key=self.class_to_idx.get)

    def filter(
        self,
        experiment_ids: Optional[Sequence[str]] = None,
        perturbation_classes: Optional[Sequence[str]] = None,
    ) -> "TrajectoryDataset":
        trajectories = self.trajectories
        if experiment_ids is not None:
            allowed_experiments = set(experiment_ids)
            trajectories = [traj for traj in trajectories if traj.experiment_id in allowed_experiments]
        if perturbation_classes is not None:
            allowed_classes = set(perturbation_classes)
            trajectories = [traj for traj in trajectories if traj.perturbation_class in allowed_classes]
        return dataclasses.replace(self, trajectories=trajectories)


def _detect_z_mu_b_cols(df: pd.DataFrame) -> List[str]:
    cols = sorted(column for column in df.columns if column.startswith("z_mu_b"))
    if not cols:
        raise ValueError("No z_mu_b columns found in DataFrame")
    return cols


def _compute_experiment_delta_t(
    df: pd.DataFrame,
    dt_min: float = 720.0,
    dt_max: float = 5.0 * 3600.0,
) -> Dict[str, float]:
    dt_by_exp: Dict[str, float] = {}
    for exp_id, exp_df in df.groupby("experiment_id"):
        all_diffs: List[float] = []
        for _, emb_df in exp_df.groupby("embryo_id"):
            times = emb_df.sort_values("frame_index")["relative_time_s"].to_numpy(dtype=np.float64)
            if len(times) > 1:
                all_diffs.extend(np.diff(times).tolist())
        if not all_diffs:
            dt_by_exp[exp_id] = np.nan
            continue

        diffs_arr = np.asarray(all_diffs, dtype=np.float64)
        valid = (diffs_arr >= dt_min) & (diffs_arr <= dt_max)
        if valid.any():
            dt_by_exp[exp_id] = float(np.median(diffs_arr[valid]))
        else:
            dt_by_exp[exp_id] = float(np.median(diffs_arr))
    return dt_by_exp


def _repair_timestamps(
    times: np.ndarray,
    consensus_dt: float,
    dt_min: float = 720.0,
    dt_max: float = 5.0 * 3600.0,
) -> np.ndarray:
    if len(times) < 2 or not np.isfinite(consensus_dt):
        return times

    repaired = times.copy()
    diffs = np.diff(repaired)
    bad = (diffs < dt_min) | (diffs > dt_max)
    if not bad.any():
        return repaired

    for bad_index in np.where(bad)[0]:
        offset = repaired[bad_index] + consensus_dt - repaired[bad_index + 1]
        repaired[bad_index + 1 :] += offset
    return repaired


def _fit_pca(
    df: pd.DataFrame,
    z_mu_cols: List[str],
    n_components: int,
    scale: bool,
) -> Tuple[PCA, Optional[StandardScaler]]:
    x_raw = df[z_mu_cols].to_numpy(dtype=np.float64)
    valid = ~np.isnan(x_raw).any(axis=1)
    x_valid = x_raw[valid]
    if len(x_valid) == 0:
        raise ValueError("No valid embedding rows available for PCA fitting")

    scaler: Optional[StandardScaler] = None
    if scale:
        scaler = StandardScaler()
        x_valid = scaler.fit_transform(x_valid)

    pca = PCA(n_components=n_components)
    pca.fit(x_valid)
    return pca, scaler


def _project(x_raw: np.ndarray, pca: PCA, scaler: Optional[StandardScaler]) -> np.ndarray:
    x_project = scaler.transform(x_raw) if scaler is not None else x_raw
    return pca.transform(x_project)


def _resolve_perturbation_class(grp_sorted: pd.DataFrame) -> str:
    for column in ("genotype", "perturbation_class", "class_label"):
        if column in grp_sorted.columns:
            return str(grp_sorted[column].iloc[0])
    return "unknown"


def _extract_metadata(grp_sorted: pd.DataFrame, metadata_columns: Sequence[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for column in metadata_columns:
        if column not in grp_sorted.columns:
            continue
        value = grp_sorted[column].iloc[0]
        if pd.isna(value):
            continue
        metadata[column] = value.item() if hasattr(value, "item") else value
    return metadata


def load_trajectories(
    experiment_ids: Optional[Sequence[str]] = None,
    build_dir: str | Path = DEFAULT_BUILD_DIR,
    n_components: int = 10,
    scale: bool = True,
    min_trajectory_length: int = 15,
    verbose: bool = True,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
) -> TrajectoryDataset:
    """Load build06 trajectory CSVs and project them into PCA space."""

    build_dir = Path(build_dir)
    if experiment_ids is None:
        prefix = "df03_final_output_with_latents_"
        experiment_ids = sorted(
            path.stem[len(prefix) :]
            for path in build_dir.glob(f"{prefix}*.csv")
        )
        if verbose:
            print(f"  Auto-discovered {len(experiment_ids)} experiments in {build_dir}")
        if not experiment_ids:
            raise ValueError(f"No df03_final_output_with_latents_*.csv files found in {build_dir}")

    frames: List[pd.DataFrame] = []
    for exp_id in experiment_ids:
        path = build_dir / f"df03_final_output_with_latents_{exp_id}.csv"
        if not path.exists():
            if verbose:
                print(f"  [skip] {exp_id}: file not found")
            continue
        df_exp = pd.read_csv(path, low_memory=False)
        if "experiment_id" not in df_exp.columns:
            df_exp["experiment_id"] = exp_id
        frames.append(df_exp)
        if verbose:
            print(f"  Loaded {exp_id}: {len(df_exp)} rows")

    if not frames:
        raise ValueError("No experiment files could be loaded")

    df = pd.concat(frames, ignore_index=True)
    if "use_embryo_flag" in df.columns:
        n_before = len(df)
        df = df[df["use_embryo_flag"] == True].copy()  # noqa: E712
        if verbose:
            print(f"  Filtered use_embryo_flag: {n_before} -> {len(df)} rows")

    z_mu_cols = _detect_z_mu_b_cols(df)
    if verbose:
        print(f"  Found {len(z_mu_cols)} z_mu_b columns")

    valid_mask = ~df[z_mu_cols].isna().any(axis=1)
    df = df[valid_mask].copy()

    pca, scaler = _fit_pca(df, z_mu_cols, n_components=n_components, scale=scale)
    x_pc = _project(df[z_mu_cols].to_numpy(dtype=np.float64), pca, scaler)

    if verbose:
        explained = np.cumsum(pca.explained_variance_ratio_)[-1]
        print(f"  PCA: {n_components} components, {explained * 100:.1f}% variance explained")

    dt_by_exp = _compute_experiment_delta_t(df)
    trajectories: List[EmbryoTrajectory] = []
    n_repaired = 0

    grouped = df.groupby(["experiment_id", "embryo_id"])
    for (exp_id, embryo_id), grp in grouped:
        grp_sorted = grp.sort_values("frame_index")
        if len(grp_sorted) < min_trajectory_length:
            continue

        raw_times = grp_sorted["relative_time_s"].to_numpy(dtype=np.float64)
        repaired_times = _repair_timestamps(raw_times, dt_by_exp.get(exp_id, np.nan))
        if not np.array_equal(raw_times, repaired_times):
            n_repaired += 1

        row_indices = df.index.get_indexer(grp_sorted.index)
        trajectory = x_pc[row_indices]
        frame_index = None
        if "frame_index" in grp_sorted.columns:
            frame_index = grp_sorted["frame_index"].to_numpy(dtype=np.int64)

        temperature = np.nan
        if "temperature" in grp_sorted.columns:
            temperature = float(grp_sorted["temperature"].iloc[0])

        trajectories.append(
            EmbryoTrajectory(
                embryo_id=str(embryo_id),
                trajectory=trajectory,
                time_seconds=repaired_times,
                delta_t=float(dt_by_exp.get(exp_id, np.nan)),
                temperature=temperature,
                perturbation_class=_resolve_perturbation_class(grp_sorted),
                experiment_id=str(exp_id),
                metadata=_extract_metadata(grp_sorted, metadata_columns),
                frame_index=frame_index,
            )
        )

    if not trajectories:
        raise ValueError(
            f"No valid trajectories met min_trajectory_length={min_trajectory_length} in {build_dir}"
        )

    if verbose:
        dropped = len(grouped) - len(trajectories)
        lengths = [len(traj.trajectory) for traj in trajectories]
        print(
            f"  Built {len(trajectories)} trajectories "
            f"(dropped {dropped} with <{min_trajectory_length} frames)"
        )
        if n_repaired > 0:
            print(f"  Repaired timestamps in {n_repaired} trajectories")
        print(
            f"  Trajectory lengths: min={min(lengths)}, "
            f"median={int(np.median(lengths))}, max={max(lengths)}"
        )

    return TrajectoryDataset(
        trajectories=trajectories,
        pca=pca,
        scaler=scaler,
        z_mu_cols=z_mu_cols,
        build_dir=build_dir,
    )"""Load latent trajectories for the beta particle predictor.

This module ports the useful trajectory-loading infrastructure from the
legacy dynamo namespace while keeping the output contract simple and aligned
with the particle-prediction docs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_BUILD_DIR = Path("morphseq_playground") / "metadata" / "build06_output"
DEFAULT_METADATA_COLUMNS: Tuple[str, ...] = (
    "background",
    "phenotype",
    "phenotype_label",
)


@dataclass(frozen=True)
class EmbryoTrajectory:
    """One embryo trajectory in PCA latent space."""

    embryo_id: str
    trajectory: np.ndarray
    time_seconds: np.ndarray
    delta_t: float
    temperature: float
    perturbation_class: str
    experiment_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    frame_index: Optional[np.ndarray] = None


@dataclass
class TrajectoryDataset:
    """Loaded trajectory collection plus PCA artifacts."""

    trajectories: List[EmbryoTrajectory]
    pca: PCA
    scaler: Optional[StandardScaler]
    z_mu_cols: List[str]
    class_to_idx: Dict[str, int] = field(default_factory=dict)
    build_dir: Path = DEFAULT_BUILD_DIR

    def __post_init__(self) -> None:
        if not self.class_to_idx:
            classes = sorted({traj.perturbation_class for traj in self.trajectories})
            self.class_to_idx = {name: index for index, name in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.trajectories)

    @property
    def n_components(self) -> int:
        return int(self.pca.n_components_)

    @property
    def class_names(self) -> List[str]:
        return sorted(self.class_to_idx, key=self.class_to_idx.get)

    def filter(
        self,
        experiment_ids: Optional[Sequence[str]] = None,
        perturbation_classes: Optional[Sequence[str]] = None,
    ) -> "TrajectoryDataset":
        trajectories = self.trajectories
        if experiment_ids is not None:
            allowed_experiments = set(experiment_ids)
            trajectories = [traj for traj in trajectories if traj.experiment_id in allowed_experiments]
        if perturbation_classes is not None:
            allowed_classes = set(perturbation_classes)
            trajectories = [traj for traj in trajectories if traj.perturbation_class in allowed_classes]
        return dataclasses.replace(self, trajectories=trajectories)


def _detect_z_mu_b_cols(df: pd.DataFrame) -> List[str]:
    cols = sorted(column for column in df.columns if column.startswith("z_mu_b"))
    if not cols:
        raise ValueError("No z_mu_b columns found in DataFrame")
    return cols


def _compute_experiment_delta_t(
    df: pd.DataFrame,
    dt_min: float = 720.0,
    dt_max: float = 5.0 * 3600.0,
) -> Dict[str, float]:
    dt_by_exp: Dict[str, float] = {}
    for exp_id, exp_df in df.groupby("experiment_id"):
        all_diffs: List[float] = []
        for _, emb_df in exp_df.groupby("embryo_id"):
            times = emb_df.sort_values("frame_index")["relative_time_s"].to_numpy(dtype=np.float64)
            if len(times) > 1:
                all_diffs.extend(np.diff(times).tolist())
        if not all_diffs:
            dt_by_exp[exp_id] = np.nan
            continue

        diffs_arr = np.asarray(all_diffs, dtype=np.float64)
        valid = (diffs_arr >= dt_min) & (diffs_arr <= dt_max)
        if valid.any():
            dt_by_exp[exp_id] = float(np.median(diffs_arr[valid]))
        else:
            dt_by_exp[exp_id] = float(np.median(diffs_arr))
    return dt_by_exp


def _repair_timestamps(
    times: np.ndarray,
    consensus_dt: float,
    dt_min: float = 720.0,
    dt_max: float = 5.0 * 3600.0,
) -> np.ndarray:
    if len(times) < 2 or not np.isfinite(consensus_dt):
        return times

    repaired = times.copy()
    diffs = np.diff(repaired)
    bad = (diffs < dt_min) | (diffs > dt_max)
    if not bad.any():
        return repaired

    for bad_index in np.where(bad)[0]:
        offset = repaired[bad_index] + consensus_dt - repaired[bad_index + 1]
        repaired[bad_index + 1 :] += offset
    return repaired


def _fit_pca(
    df: pd.DataFrame,
    z_mu_cols: List[str],
    n_components: int,
    scale: bool,
) -> Tuple[PCA, Optional[StandardScaler]]:
    x_raw = df[z_mu_cols].to_numpy(dtype=np.float64)
    valid = ~np.isnan(x_raw).any(axis=1)
    x_valid = x_raw[valid]
    if len(x_valid) == 0:
        raise ValueError("No valid embedding rows available for PCA fitting")

    scaler: Optional[StandardScaler] = None
    if scale:
        scaler = StandardScaler()
        x_valid = scaler.fit_transform(x_valid)

    pca = PCA(n_components=n_components)
    pca.fit(x_valid)
    return pca, scaler


def _project(x_raw: np.ndarray, pca: PCA, scaler: Optional[StandardScaler]) -> np.ndarray:
    x_project = scaler.transform(x_raw) if scaler is not None else x_raw
    return pca.transform(x_project)


def _resolve_perturbation_class(grp_sorted: pd.DataFrame) -> str:
    for column in ("genotype", "perturbation_class", "class_label"):
        if column in grp_sorted.columns:
            return str(grp_sorted[column].iloc[0])
    return "unknown"


def _extract_metadata(grp_sorted: pd.DataFrame, metadata_columns: Sequence[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for column in metadata_columns:
        if column not in grp_sorted.columns:
            continue
        value = grp_sorted[column].iloc[0]
        if pd.isna(value):
            continue
        metadata[column] = value.item() if hasattr(value, "item") else value
    return metadata


def load_trajectories(
    experiment_ids: Optional[Sequence[str]] = None,
    build_dir: str | Path = DEFAULT_BUILD_DIR,
    n_components: int = 10,
    scale: bool = True,
    min_trajectory_length: int = 15,
    verbose: bool = True,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
) -> TrajectoryDataset:
    """Load build06 trajectory CSVs and project them into PCA space."""

    build_dir = Path(build_dir)
    if experiment_ids is None:
        prefix = "df03_final_output_with_latents_"
        experiment_ids = sorted(
            path.stem[len(prefix) :]
            for path in build_dir.glob(f"{prefix}*.csv")
        )
        if verbose:
            print(f"  Auto-discovered {len(experiment_ids)} experiments in {build_dir}")
        if not experiment_ids:
            raise ValueError(f"No df03_final_output_with_latents_*.csv files found in {build_dir}")

    frames: List[pd.DataFrame] = []
    for exp_id in experiment_ids:
        path = build_dir / f"df03_final_output_with_latents_{exp_id}.csv"
        if not path.exists():
            if verbose:
                print(f"  [skip] {exp_id}: file not found")
            continue
        df_exp = pd.read_csv(path, low_memory=False)
        if "experiment_id" not in df_exp.columns:
            df_exp["experiment_id"] = exp_id
        frames.append(df_exp)
        if verbose:
            print(f"  Loaded {exp_id}: {len(df_exp)} rows")

    if not frames:
        raise ValueError("No experiment files could be loaded")

    df = pd.concat(frames, ignore_index=True)
    if "use_embryo_flag" in df.columns:
        n_before = len(df)
        df = df[df["use_embryo_flag"] == True].copy()  # noqa: E712
        if verbose:
            print(f"  Filtered use_embryo_flag: {n_before} -> {len(df)} rows")

    z_mu_cols = _detect_z_mu_b_cols(df)
    if verbose:
        print(f"  Found {len(z_mu_cols)} z_mu_b columns")

    valid_mask = ~df[z_mu_cols].isna().any(axis=1)
    df = df[valid_mask].copy()

    pca, scaler = _fit_pca(df, z_mu_cols, n_components=n_components, scale=scale)
    x_pc = _project(df[z_mu_cols].to_numpy(dtype=np.float64), pca, scaler)

    if verbose:
        explained = np.cumsum(pca.explained_variance_ratio_)[-1]
        print(f"  PCA: {n_components} components, {explained * 100:.1f}% variance explained")

    dt_by_exp = _compute_experiment_delta_t(df)
    trajectories: List[EmbryoTrajectory] = []
    n_repaired = 0

    grouped = df.groupby(["experiment_id", "embryo_id"])
    for (exp_id, embryo_id), grp in grouped:
        grp_sorted = grp.sort_values("frame_index")
        if len(grp_sorted) < min_trajectory_length:
            continue

        raw_times = grp_sorted["relative_time_s"].to_numpy(dtype=np.float64)
        repaired_times = _repair_timestamps(raw_times, dt_by_exp.get(exp_id, np.nan))
        if not np.array_equal(raw_times, repaired_times):
            n_repaired += 1

        row_indices = df.index.get_indexer(grp_sorted.index)
        trajectory = x_pc[row_indices]
        frame_index = None
        if "frame_index" in grp_sorted.columns:
            frame_index = grp_sorted["frame_index"].to_numpy(dtype=np.int64)

        temperature = np.nan
        if "temperature" in grp_sorted.columns:
            temperature = float(grp_sorted["temperature"].iloc[0])

        trajectories.append(
            EmbryoTrajectory(
                embryo_id=str(embryo_id),
                trajectory=trajectory,
                time_seconds=repaired_times,
                delta_t=float(dt_by_exp.get(exp_id, np.nan)),
                temperature=temperature,
                perturbation_class=_resolve_perturbation_class(grp_sorted),
                experiment_id=str(exp_id),
                metadata=_extract_metadata(grp_sorted, metadata_columns),
                frame_index=frame_index,
            )
        )

    if not trajectories:
        raise ValueError(
            f"No valid trajectories met min_trajectory_length={min_trajectory_length} in {build_dir}"
        )

    if verbose:
        dropped = len(grouped) - len(trajectories)
        lengths = [len(traj.trajectory) for traj in trajectories]
        print(
            f"  Built {len(trajectories)} trajectories "
            f"(dropped {dropped} with <{min_trajectory_length} frames)"
        )
        if n_repaired > 0:
            print(f"  Repaired timestamps in {n_repaired} trajectories")
        print(
            f"  Trajectory lengths: min={min(lengths)}, "
            f"median={int(np.median(lengths))}, max={max(lengths)}"
        )

    return TrajectoryDataset(
        trajectories=trajectories,
        pca=pca,
        scaler=scaler,
        z_mu_cols=z_mu_cols,
        build_dir=build_dir,
    )
            experiment_id_set = set(experiment_ids)
            trajectories = [
                trajectory for trajectory in trajectories if trajectory.experiment_id in experiment_id_set
            ]
        if perturbation_classes is not None:
            class_set = set(perturbation_classes)
            trajectories = [
                trajectory for trajectory in trajectories if trajectory.perturbation_class in class_set
            ]
        return dataclasses.replace(self, trajectories=trajectories)


def _detect_z_mu_b_cols(dataframe: pd.DataFrame) -> List[str]:
    columns = sorted(column for column in dataframe.columns if column.startswith("z_mu_b"))
    if not columns:
        raise ValueError("No z_mu_b columns found in DataFrame")
    return columns


def _compute_experiment_delta_t(
    dataframe: pd.DataFrame,
    dt_min: float = 720.0,
    dt_max: float = 5.0 * 3600.0,
) -> Dict[str, float]:
    dt_by_experiment: Dict[str, float] = {}
    for experiment_id, experiment_df in dataframe.groupby("experiment_id"):
        diffs: List[float] = []
        for _, embryo_df in experiment_df.groupby("embryo_id"):
            embryo_times = embryo_df.sort_values("frame_index")["relative_time_s"].to_numpy()
            if len(embryo_times) > 1:
                diffs.extend(np.diff(embryo_times).tolist())
        if not diffs:
            dt_by_experiment[experiment_id] = np.nan
            continue

        diff_array = np.asarray(diffs, dtype=np.float64)
        valid_mask = (diff_array >= dt_min) & (diff_array <= dt_max)
        if valid_mask.any():
            dt_by_experiment[experiment_id] = float(np.median(diff_array[valid_mask]))
        else:
            dt_by_experiment[experiment_id] = float(np.median(diff_array))
    return dt_by_experiment


def _repair_timestamps(
    time_seconds: np.ndarray,
    consensus_dt: float,
    dt_min: float = 720.0,
    dt_max: float = 5.0 * 3600.0,
) -> np.ndarray:
    if len(time_seconds) < 2 or np.isnan(consensus_dt):
        return time_seconds

    repaired = time_seconds.copy()
    diffs = np.diff(repaired)
    bad_gaps = (diffs < dt_min) | (diffs > dt_max)
    if not bad_gaps.any():
        return repaired

    for gap_index in np.where(bad_gaps)[0]:
        offset = repaired[gap_index] + consensus_dt - repaired[gap_index + 1]
        repaired[gap_index + 1 :] += offset
    return repaired


def _fit_pca(
    dataframe: pd.DataFrame,
    z_mu_cols: List[str],
    n_components: int,
    scale: bool,
) -> Tuple[PCA, Optional[StandardScaler]]:
    embedding_array = dataframe[z_mu_cols].to_numpy()
    valid_mask = ~np.isnan(embedding_array).any(axis=1)
    valid_embeddings = embedding_array[valid_mask]
    if len(valid_embeddings) == 0:
        raise ValueError("No valid embedding rows available for PCA")

    scaler: Optional[StandardScaler] = None
    if scale:
        scaler = StandardScaler()
        valid_embeddings = scaler.fit_transform(valid_embeddings)

    pca = PCA(n_components=n_components)
    pca.fit(valid_embeddings)
    return pca, scaler


def _project_embeddings(
    embedding_array: np.ndarray,
    pca: PCA,
    scaler: Optional[StandardScaler],
) -> np.ndarray:
    transformed = embedding_array
    if scaler is not None:
        transformed = scaler.transform(transformed)
    return pca.transform(transformed)


def _extract_curated_metadata(
    embryo_df: pd.DataFrame,
    metadata_columns: Sequence[str],
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for column in metadata_columns:
        if column not in embryo_df.columns:
            continue
        values = embryo_df[column].dropna()
        if len(values) == 0:
            continue
        unique_values = values.unique()
        if len(unique_values) == 1:
            metadata[column] = unique_values[0].item() if hasattr(unique_values[0], "item") else unique_values[0]
            continue
        metadata[column] = embryo_df[column].to_numpy()
    return metadata


def load_trajectories(
    experiment_ids: Optional[Sequence[str]] = None,
    build_dir: str | Path = "",
    n_components: int = 10,
    scale: bool = True,
    min_trajectory_length: int = 15,
    metadata_columns: Optional[Sequence[str]] = None,
    include_frame_index: bool = False,
    verbose: bool = True,
) -> TrajectoryDataset:
    """Load build06 CSVs and construct per-embryo PCA trajectories."""

    build_dir = Path(build_dir)
    metadata_columns = tuple(DEFAULT_METADATA_COLUMNS if metadata_columns is None else metadata_columns)

    if experiment_ids is None:
        prefix = "df03_final_output_with_latents_"
        suffix = ".csv"
        experiment_ids = sorted(
            path.stem[len(prefix) :]
            for path in build_dir.glob(f"{prefix}*{suffix}")
        )
        if verbose:
            print(f"  Auto-discovered {len(experiment_ids)} experiments in {build_dir}")
        if not experiment_ids:
            raise ValueError(f"No df03_final_output_with_latents_*.csv files found in {build_dir}")

    dataframes: List[pd.DataFrame] = []
    for experiment_id in experiment_ids:
        csv_path = build_dir / f"df03_final_output_with_latents_{experiment_id}.csv"
        if not csv_path.exists():
            if verbose:
                print(f"  [skip] {experiment_id}: file not found")
            continue
        experiment_df = pd.read_csv(csv_path, low_memory=False)
        if "experiment_id" not in experiment_df.columns:
            experiment_df["experiment_id"] = experiment_id
        dataframes.append(experiment_df)
        if verbose:
            print(f"  Loaded {experiment_id}: {len(experiment_df)} rows")

    if not dataframes:
        raise ValueError("No experiment files could be loaded")
    dataframe = pd.concat(dataframes, ignore_index=True)

    if "use_embryo_flag" in dataframe.columns:
        n_rows_before = len(dataframe)
        dataframe = dataframe[dataframe["use_embryo_flag"] == True].copy()  # noqa: E712
        if verbose:
            print(f"  Filtered use_embryo_flag: {n_rows_before} → {len(dataframe)} rows")

    z_mu_cols = _detect_z_mu_b_cols(dataframe)
    if verbose:
        print(f"  Found {len(z_mu_cols)} z_mu_b columns")

    valid_embedding_mask = ~dataframe[z_mu_cols].isna().any(axis=1)
    dataframe = dataframe[valid_embedding_mask].copy()

    pca, scaler = _fit_pca(dataframe, z_mu_cols, n_components=n_components, scale=scale)
    if verbose:
        explained_variance = pca.explained_variance_ratio_
        print(
            f"  PCA: {n_components} components, {np.cumsum(explained_variance)[-1] * 100:.1f}% variance explained"
        )

    projected_embeddings = _project_embeddings(dataframe[z_mu_cols].to_numpy(), pca=pca, scaler=scaler)
    delta_t_by_experiment = _compute_experiment_delta_t(dataframe)
    if verbose:
        for experiment_id, delta_t in delta_t_by_experiment.items():
            print(f"  Experiment {experiment_id}: median Δt = {delta_t:.1f}s")

    trajectories: List[EmbryoTrajectory] = []
    grouped = dataframe.groupby(["experiment_id", "embryo_id"])
    n_repaired = 0

    for (experiment_id, embryo_id), embryo_df in grouped:
        embryo_df = embryo_df.sort_values("frame_index")
        if len(embryo_df) < min_trajectory_length:
            continue

        dataframe_indices = dataframe.index.get_indexer(embryo_df.index)
        raw_time_seconds = embryo_df["relative_time_s"].to_numpy(dtype=np.float64)
        consensus_dt = delta_t_by_experiment.get(experiment_id, np.nan)
        repaired_time_seconds = _repair_timestamps(raw_time_seconds, consensus_dt=consensus_dt)
        if not np.array_equal(raw_time_seconds, repaired_time_seconds):
            n_repaired += 1

        trajectories.append(
            EmbryoTrajectory(
                embryo_id=str(embryo_id),
                trajectory=projected_embeddings[dataframe_indices],
                time_seconds=repaired_time_seconds,
                delta_t=consensus_dt,
                temperature=(
                    float(embryo_df["temperature"].iloc[0])
                    if "temperature" in embryo_df.columns
                    else np.nan
                ),
                perturbation_class=str(embryo_df["genotype"].iloc[0]),
                experiment_id=str(experiment_id),
                metadata=_extract_curated_metadata(embryo_df, metadata_columns=metadata_columns),
                frame_index=(
                    embryo_df["frame_index"].to_numpy(dtype=np.int64)
                    if include_frame_index and "frame_index" in embryo_df.columns
                    else None
                ),
            )
        )

    if verbose and trajectories:
        lengths = [len(trajectory.trajectory) for trajectory in trajectories]
        print(
            f"  Built {len(trajectories)} trajectories "
            f"(dropped {len(grouped) - len(trajectories)} with <{min_trajectory_length} frames)"
        )
        if n_repaired > 0:
            print(
                "  Repaired timestamps in "
                f"{n_repaired} trajectories (outlier gaps replaced with consensus Δt)"
            )
        print(
            f"  Trajectory lengths: min={min(lengths)}, median={int(np.median(lengths))}, max={max(lengths)}"
        )

    if not trajectories:
        raise ValueError(
            "No embryo trajectories satisfied the loader filters. "
            "Check experiment_ids and min_trajectory_length."
        )

    return TrajectoryDataset(
        trajectories=trajectories,
        pca=pca,
        scaler=scaler,
        z_mu_cols=z_mu_cols,
    )