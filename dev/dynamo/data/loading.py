"""Load morphseq build06 data and organize into per-embryo trajectory structures.

Workflow:
  1. Load CSV files from build06 output (one per experiment).
  2. Filter to valid embryos (use_embryo_flag).
  3. Extract z_mu_b columns, fit PCA on full dataset.
  4. Group by embryo, sort by frame_index, extract trajectory arrays.

Returns a TrajectoryDataset: a lightweight container holding a list of
EmbryoTrajectory dataclass instances plus the fitted PCA/scaler objects.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbryoTrajectory:
    """A single embryo's trajectory in PC space.

    Attributes:
        embryo_id: Unique identifier string.
        trajectory: (T, D) array of PC-projected latent vectors, sorted by time.
        time_seconds: (T,) array of absolute observation times in seconds.
        delta_t: Median inter-frame interval (seconds) for this experiment.
        temperature: Incubation temperature in °C (may be NaN).
        perturbation_class: Genotype / treatment label.
        experiment_id: Source experiment identifier.
    """
    embryo_id: str
    trajectory: np.ndarray        # (T, D)
    time_seconds: np.ndarray      # (T,)
    delta_t: float                # experiment-level median Δt
    temperature: float
    perturbation_class: str
    experiment_id: str


@dataclass
class TrajectoryDataset:
    """Container for all loaded trajectories plus PCA artifacts.

    Attributes:
        trajectories: List of EmbryoTrajectory instances.
        pca: Fitted PCA object used to project z_mu_b → PCs.
        scaler: Fitted StandardScaler (None if scale=False).
        z_mu_cols: Column names of the raw z_mu_b features.
        class_to_idx: Mapping from perturbation class label → integer index.
    """
    trajectories: List[EmbryoTrajectory]
    pca: PCA
    scaler: Optional[StandardScaler]
    z_mu_cols: List[str]
    class_to_idx: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.class_to_idx:
            classes = sorted({t.perturbation_class for t in self.trajectories})
            self.class_to_idx = {c: i for i, c in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.trajectories)

    @property
    def n_components(self) -> int:
        return self.pca.n_components_

    @property
    def class_names(self) -> List[str]:
        return sorted(self.class_to_idx, key=self.class_to_idx.get)

    def filter(self, experiment_ids: Optional[Sequence[str]] = None,
               perturbation_classes: Optional[Sequence[str]] = None) -> "TrajectoryDataset":
        """Return a new dataset containing only matching trajectories."""
        trajs = self.trajectories
        if experiment_ids is not None:
            exp_set = set(experiment_ids)
            trajs = [t for t in trajs if t.experiment_id in exp_set]
        if perturbation_classes is not None:
            cls_set = set(perturbation_classes)
            trajs = [t for t in trajs if t.perturbation_class in cls_set]
        return dataclasses.replace(self, trajectories=trajs)


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------

def _detect_z_mu_b_cols(df: pd.DataFrame) -> List[str]:
    """Return sorted list of z_mu_b column names."""
    cols = sorted([c for c in df.columns if c.startswith("z_mu_b")])
    if not cols:
        raise ValueError("No z_mu_b columns found in DataFrame")
    return cols


def _compute_experiment_delta_t(df: pd.DataFrame) -> Dict[str, float]:
    """Compute median inter-frame Δt per experiment from relative_time_s."""
    dt_by_exp: Dict[str, float] = {}
    for exp_id, exp_df in df.groupby("experiment_id"):
        all_diffs = []
        for _, emb_df in exp_df.groupby("embryo_id"):
            times = emb_df.sort_values("frame_index")["relative_time_s"].values
            if len(times) > 1:
                all_diffs.extend(np.diff(times).tolist())
        dt_by_exp[exp_id] = float(np.median(all_diffs)) if all_diffs else np.nan
    return dt_by_exp


def _fit_pca(
    df: pd.DataFrame,
    z_mu_cols: List[str],
    n_components: int,
    scale: bool,
) -> Tuple[PCA, Optional[StandardScaler]]:
    """Fit PCA (and optional scaler) on z_mu_b embeddings."""
    X = df[z_mu_cols].values
    valid = ~np.isnan(X).any(axis=1)
    X_valid = X[valid]
    if len(X_valid) == 0:
        raise ValueError("No valid (non-NaN) embedding rows")

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_valid = scaler.fit_transform(X_valid)

    pca = PCA(n_components=n_components)
    pca.fit(X_valid)
    return pca, scaler


def _project(
    X: np.ndarray,
    pca: PCA,
    scaler: Optional[StandardScaler],
) -> np.ndarray:
    """Project raw z_mu_b vectors through scaler + PCA."""
    if scaler is not None:
        X = scaler.transform(X)
    return pca.transform(X)


def load_trajectories(
    experiment_ids: Sequence[str],
    build_dir: str | Path,
    n_components: int = 10,
    scale: bool = True,
    min_trajectory_length: int = 3,
    verbose: bool = True,
) -> TrajectoryDataset:
    """Load experiments and build per-embryo trajectory structures.

    Args:
        experiment_ids: Experiment identifiers (filename stems without prefix).
        build_dir: Path to directory containing df03_final_output_with_latents_*.csv files.
        n_components: Number of PCA components (default 10 per model spec).
        scale: Whether to standardize features before PCA.
        min_trajectory_length: Discard embryos with fewer frames.
        verbose: Print progress information.

    Returns:
        TrajectoryDataset containing all valid embryo trajectories.
    """
    build_dir = Path(build_dir)

    # -- 1. Load and concatenate CSVs ----------------------------------------
    frames: List[pd.DataFrame] = []
    for exp_id in experiment_ids:
        path = build_dir / f"df03_final_output_with_latents_{exp_id}.csv"
        if not path.exists():
            if verbose:
                print(f"  [skip] {exp_id}: file not found")
            continue
        df_exp = pd.read_csv(path, low_memory=False)
        # Ensure experiment_id column
        if "experiment_id" not in df_exp.columns:
            df_exp["experiment_id"] = exp_id
        frames.append(df_exp)
        if verbose:
            print(f"  Loaded {exp_id}: {len(df_exp)} rows")

    if not frames:
        raise ValueError("No experiment files could be loaded")
    df = pd.concat(frames, ignore_index=True)

    # -- 2. Filter to valid embryos ------------------------------------------
    if "use_embryo_flag" in df.columns:
        n_before = len(df)
        df = df[df["use_embryo_flag"] == True].copy()  # noqa: E712
        if verbose:
            print(f"  Filtered use_embryo_flag: {n_before} → {len(df)} rows")

    # -- 3. Detect z_mu_b columns, fit PCA -----------------------------------
    z_mu_cols = _detect_z_mu_b_cols(df)
    if verbose:
        print(f"  Found {len(z_mu_cols)} z_mu_b columns")

    # Drop rows with NaN in embeddings before PCA fit
    valid_mask = ~df[z_mu_cols].isna().any(axis=1)
    df = df[valid_mask].copy()

    pca, scaler = _fit_pca(df, z_mu_cols, n_components, scale)
    if verbose:
        var = pca.explained_variance_ratio_
        cum = np.cumsum(var)
        print(f"  PCA: {n_components} components, {cum[-1]*100:.1f}% variance explained")

    # -- 4. Project all embeddings -------------------------------------------
    X_raw = df[z_mu_cols].values
    X_pc = _project(X_raw, pca, scaler)

    # -- 5. Compute experiment-level Δt --------------------------------------
    dt_by_exp = _compute_experiment_delta_t(df)
    if verbose:
        for eid, dt in dt_by_exp.items():
            print(f"  Experiment {eid}: median Δt = {dt:.1f}s")

    # -- 6. Group by embryo and build trajectories ---------------------------
    trajectories: List[EmbryoTrajectory] = []
    grouped = df.groupby("embryo_id")

    for embryo_id, grp in grouped:
        grp_sorted = grp.sort_values("frame_index")
        idx = grp_sorted.index
        n_frames = len(grp_sorted)

        if n_frames < min_trajectory_length:
            continue

        exp_id = grp_sorted["experiment_id"].iloc[0]

        trajectories.append(EmbryoTrajectory(
            embryo_id=embryo_id,
            trajectory=X_pc[df.index.get_indexer(idx)],
            time_seconds=grp_sorted["relative_time_s"].values.astype(np.float64),
            delta_t=dt_by_exp.get(exp_id, np.nan),
            temperature=float(grp_sorted["temperature"].iloc[0])
                if "temperature" in grp_sorted.columns
                else np.nan,
            perturbation_class=str(grp_sorted["genotype"].iloc[0]),
            experiment_id=exp_id,
        ))

    if verbose:
        print(f"  Built {len(trajectories)} trajectories "
              f"(dropped {len(grouped) - len(trajectories)} with <{min_trajectory_length} frames)")
        lengths = [len(t.trajectory) for t in trajectories]
        print(f"  Trajectory lengths: min={min(lengths)}, median={int(np.median(lengths))}, max={max(lengths)}")

    return TrajectoryDataset(
        trajectories=trajectories,
        pca=pca,
        scaler=scaler,
        z_mu_cols=z_mu_cols,
    )
