"""Minimal trajectory loading bootstrap for particle prediction."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_BUILD_DIR = Path("morphseq_playground") / "metadata" / "build06_output"
FILE_PREFIX = "df03_final_output_with_latents_"


@dataclass(frozen=True)
class EmbryoTrajectory:
    """One embryo's raw latent trajectory projected into PCA space."""

    embryo_id: str
    trajectory: np.ndarray
    time_seconds: np.ndarray
    delta_t: float
    temperature: float
    perturbation_class: str
    experiment_id: str
    metadata: Dict[str, object] = field(default_factory=dict)
    frame_index: Optional[np.ndarray] = None


@dataclass
class TrajectoryDataset:
    """Collection of embryo trajectories plus PCA artifacts."""

    trajectories: List[EmbryoTrajectory]
    pca: PCA
    scaler: Optional[StandardScaler]
    z_mu_cols: List[str]
    class_to_idx: Dict[str, int] = field(default_factory=dict)
    build_dir: Path = DEFAULT_BUILD_DIR

    def __post_init__(self) -> None:
        if not self.class_to_idx:
            class_names = sorted({traj.perturbation_class for traj in self.trajectories})
            self.class_to_idx = {name: index for index, name in enumerate(class_names)}

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
    columns = sorted(column for column in df.columns if column.startswith("z_mu_b"))
    if not columns:
        raise ValueError("No z_mu_b columns found in DataFrame")
    return columns


def _resolve_experiment_ids(build_dir: Path, experiment_ids: Optional[Sequence[str]]) -> List[str]:
    if experiment_ids is not None:
        return list(experiment_ids)

    resolved = sorted(path.stem[len(FILE_PREFIX) :] for path in build_dir.glob(f"{FILE_PREFIX}*.csv"))
    if not resolved:
        raise ValueError(f"No {FILE_PREFIX}*.csv files found in {build_dir}")
    return resolved


def _load_experiment_frames(build_dir: Path, experiment_ids: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for experiment_id in experiment_ids:
        csv_path = build_dir / f"{FILE_PREFIX}{experiment_id}.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path, low_memory=False)
        if "experiment_id" not in frame.columns:
            frame["experiment_id"] = experiment_id
        frames.append(frame)

    if not frames:
        raise ValueError("No experiment files could be loaded")

    return pd.concat(frames, ignore_index=True)


def _compute_experiment_delta_t(df: pd.DataFrame) -> Dict[str, float]:
    delta_t_by_experiment: Dict[str, float] = {}
    for experiment_id, experiment_df in df.groupby("experiment_id"):
        diffs: List[float] = []
        for _, embryo_df in experiment_df.groupby("embryo_id"):
            time_seconds = embryo_df.sort_values("frame_index")["relative_time_s"].to_numpy(dtype=np.float64)
            if len(time_seconds) > 1:
                diffs.extend(np.diff(time_seconds).tolist())
        delta_t_by_experiment[experiment_id] = float(np.median(diffs)) if diffs else np.nan
    return delta_t_by_experiment


def _fit_pca(
    df: pd.DataFrame,
    z_mu_cols: Sequence[str],
    n_components: int,
    scale: bool,
) -> tuple[PCA, Optional[StandardScaler]]:
    features = df[list(z_mu_cols)].to_numpy(dtype=np.float64)
    if len(features) == 0:
        raise ValueError("No valid embedding rows available for PCA fitting")

    scaler: Optional[StandardScaler] = None
    if scale:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    component_count = min(n_components, features.shape[0], features.shape[1])
    if component_count < 1:
        raise ValueError("Unable to fit PCA with zero components")

    pca = PCA(n_components=component_count)
    pca.fit(features)
    return pca, scaler


def _project_features(
    df: pd.DataFrame,
    z_mu_cols: Sequence[str],
    pca: PCA,
    scaler: Optional[StandardScaler],
) -> np.ndarray:
    features = df[list(z_mu_cols)].to_numpy(dtype=np.float64)
    if scaler is not None:
        features = scaler.transform(features)
    return pca.transform(features)


def _extract_metadata(group: pd.DataFrame) -> Dict[str, object]:
    metadata: Dict[str, object] = {}
    for column in ("background", "phenotype", "phenotype_label"):
        if column not in group.columns:
            continue
        value = group[column].iloc[0]
        if pd.isna(value):
            continue
        metadata[column] = value.item() if hasattr(value, "item") else value
    return metadata


def _resolve_perturbation_class(group: pd.DataFrame) -> str:
    for column in ("genotype", "perturbation_class", "class_label"):
        if column in group.columns:
            return str(group[column].iloc[0])
    return "unknown"


def load_trajectories(
    experiment_ids: Optional[Sequence[str]] = None,
    build_dir: str | Path = DEFAULT_BUILD_DIR,
    n_components: int = 10,
    scale: bool = True,
    min_trajectory_length: int = 3,
    verbose: bool = False,
) -> TrajectoryDataset:
    """Load raw trajectories and project `z_mu_b*` features into PCA space."""

    del verbose

    build_dir = Path(build_dir)
    resolved_experiment_ids = _resolve_experiment_ids(build_dir, experiment_ids)
    df = _load_experiment_frames(build_dir, resolved_experiment_ids)

    if "use_embryo_flag" in df.columns:
        df = df[df["use_embryo_flag"] == True].copy()  # noqa: E712

    z_mu_cols = _detect_z_mu_b_cols(df)
    df = df[np.isfinite(df[z_mu_cols].to_numpy(dtype=np.float64)).all(axis=1)].copy()
    if df.empty:
        raise ValueError("No valid embedding rows remain after filtering")

    pca, scaler = _fit_pca(df, z_mu_cols, n_components=n_components, scale=scale)
    projected = _project_features(df, z_mu_cols, pca, scaler)
    delta_t_by_experiment = _compute_experiment_delta_t(df)

    trajectories: List[EmbryoTrajectory] = []
    for (experiment_id, embryo_id), group in df.groupby(["experiment_id", "embryo_id"]):
        group = group.sort_values("frame_index")
        if len(group) < min_trajectory_length:
            continue

        row_indices = df.index.get_indexer(group.index)
        temperature = np.nan
        if "temperature" in group.columns:
            temperature = float(group["temperature"].iloc[0])

        frame_index = None
        if "frame_index" in group.columns:
            frame_index = group["frame_index"].to_numpy(dtype=np.int64)

        time_seconds = group["relative_time_s"].to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(time_seconds)):
            continue

        trajectories.append(
            EmbryoTrajectory(
                embryo_id=str(embryo_id),
                trajectory=projected[row_indices],
                time_seconds=time_seconds,
                delta_t=float(delta_t_by_experiment.get(experiment_id, np.nan)),
                temperature=temperature,
                perturbation_class=_resolve_perturbation_class(group),
                experiment_id=str(experiment_id),
                metadata=_extract_metadata(group),
                frame_index=frame_index,
            )
        )

    if not trajectories:
        raise ValueError(
            f"No valid trajectories met min_trajectory_length={min_trajectory_length} in {build_dir}"
        )

    return TrajectoryDataset(
        trajectories=trajectories,
        pca=pca,
        scaler=scaler,
        z_mu_cols=z_mu_cols,
        build_dir=build_dir,
    )
