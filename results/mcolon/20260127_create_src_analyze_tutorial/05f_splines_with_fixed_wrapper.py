"""
Tutorial 05f: Spline Fitting with Patched Wrapper

Uses a fixed version of spline_fit_wrapper that handles failed LPC fits.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "05"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

meta_dir = project_root / "morphseq_playground" / "metadata" / "build04_output"
SOURCE_EXPERIMENTS = ["20260122", "20260124"]

from analyze.spline_fitting.lpc_model import LocalPrincipalCurve

def spline_fit_wrapper_fixed(
    df,
    pca_cols=None,
    group_by=None,
    stage_col="predicted_stage_hpf",
    bandwidth=0.5,
    h=None,
    max_iter=2500,
    tol=1e-5,
    angle_penalty_exp=1,
    n_bootstrap=10,
    bootstrap_size=2500,
    n_spline_points=500,
    time_window=2,
    obs_weights=None,
    max_retries=3  # NEW: retry failed fits
):
    """
    Fixed version of spline_fit_wrapper that handles LPC fit failures.

    Key fix: Retries bootstrap iterations where LPC.fit() returns None.
    """

    if group_by is not None:
        # Grouped fitting
        if group_by not in df.columns:
            raise ValueError(f"group_by column '{group_by}' not found")

        groups = df[group_by].unique()
        spline_results = []

        for group_val in tqdm(groups, desc=f"Fitting splines by {group_by}"):
            group_df = df[df[group_by] == group_val].copy()

            if group_df.empty:
                continue

            # Fit single spline for this group
            group_spline = _fit_single_spline_fixed(
                df=group_df,
                pca_cols=pca_cols,
                stage_col=stage_col,
                bandwidth=bandwidth,
                h=h,
                max_iter=max_iter,
                tol=tol,
                angle_penalty_exp=angle_penalty_exp,
                n_bootstrap=n_bootstrap,
                bootstrap_size=bootstrap_size,
                n_spline_points=n_spline_points,
                time_window=time_window,
                obs_weights=obs_weights,
                max_retries=max_retries
            )

            # Add group identifier
            group_spline[group_by] = group_val
            spline_results.append(group_spline)

        if spline_results:
            return pd.concat(spline_results, ignore_index=True)
        else:
            return pd.DataFrame()
    else:
        # Single spline
        return _fit_single_spline_fixed(
            df=df,
            pca_cols=pca_cols,
            stage_col=stage_col,
            bandwidth=bandwidth,
            h=h,
            max_iter=max_iter,
            tol=tol,
            angle_penalty_exp=angle_penalty_exp,
            n_bootstrap=n_bootstrap,
            bootstrap_size=bootstrap_size,
            n_spline_points=n_spline_points,
            time_window=time_window,
            obs_weights=obs_weights,
            max_retries=max_retries
        )


def _fit_single_spline_fixed(
    df,
    pca_cols,
    stage_col="predicted_stage_hpf",
    bandwidth=0.5,
    h=None,
    max_iter=2500,
    tol=1e-5,
    angle_penalty_exp=1,
    n_bootstrap=10,
    bootstrap_size=2500,
    n_spline_points=500,
    time_window=2,
    obs_weights=None,
    max_retries=3
):
    """Fixed single spline fitting with retry logic for failed LPC fits."""

    # Validate columns
    missing_cols = set(pca_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Columns not found: {missing_cols}")

    if stage_col not in df.columns:
        raise ValueError(f"stage_col '{stage_col}' not found")

    # Setup observation weights
    if obs_weights is None:
        obs_weights = np.ones(df.shape[0])
    obs_weights = obs_weights / np.sum(obs_weights)

    bootstrap_size = min(df.shape[0], bootstrap_size)

    # Extract coordinate array
    coord_array = df[pca_cols].values

    # Compute anchor points
    min_time = df[stage_col].min()
    early_mask = (df[stage_col] >= min_time) & (df[stage_col] < min_time + time_window)
    early_points = df.loc[early_mask, pca_cols].values

    max_time = df[stage_col].max()
    late_mask = df[stage_col] >= (max_time - time_window)
    late_points = df.loc[late_mask, pca_cols].values

    if len(early_points) == 0 or len(late_points) == 0:
        raise ValueError(
            f"No anchor points found. Stage range: [{min_time}, {max_time}], "
            f"time_window: {time_window}"
        )

    # Bootstrap iterations with retry logic
    spline_boot_list = []  # Use list instead of array to handle variable success
    rng = np.random.RandomState(42)

    successful_bootstraps = 0
    failed_attempts = 0

    pbar = tqdm(total=n_bootstrap, desc="Bootstrap iterations", leave=False)

    while successful_bootstraps < n_bootstrap:
        # Sample data
        subset_indices = rng.choice(len(coord_array), size=bootstrap_size, replace=True, p=obs_weights)
        coord_subset = coord_array[subset_indices, :]

        # Random anchor points
        start_idx = rng.choice(len(early_points))
        stop_idx = rng.choice(len(late_points))
        start_point = early_points[start_idx, :]
        stop_point = late_points[stop_idx, :]

        # Fit LPC
        lpc = LocalPrincipalCurve(
            bandwidth=bandwidth,
            h=h,
            max_iter=max_iter,
            tol=tol,
            angle_penalty_exp=angle_penalty_exp
        )

        try:
            lpc.fit(
                coord_subset,
                start_points=start_point[None, :],
                end_point=stop_point[None, :],
                num_points=n_spline_points
            )

            # CHECK: Did we get a valid spline?
            if (hasattr(lpc, 'cubic_splines') and
                lpc.cubic_splines is not None and
                len(lpc.cubic_splines) > 0 and
                lpc.cubic_splines[0] is not None):

                spline = lpc.cubic_splines[0]

                # Additional check: no NaNs in result
                if not np.isnan(spline).any():
                    spline_boot_list.append(spline)
                    successful_bootstraps += 1
                    pbar.update(1)
                else:
                    failed_attempts += 1
                    if failed_attempts > n_bootstrap * max_retries:
                        raise RuntimeError(
                            f"Too many failed LPC fits ({failed_attempts}). "
                            f"Check data quality and parameters."
                        )
            else:
                # LPC fit returned None - retry
                failed_attempts += 1
                if failed_attempts > n_bootstrap * max_retries:
                    raise RuntimeError(
                        f"Too many failed LPC fits ({failed_attempts}). "
                        f"LPC.cubic_splines is None or empty."
                    )

        except Exception as e:
            failed_attempts += 1
            if failed_attempts > n_bootstrap * max_retries:
                raise RuntimeError(
                    f"LPC fitting failed after {failed_attempts} attempts. Error: {e}"
                )

    pbar.close()

    if failed_attempts > 0:
        print(f"  Note: {failed_attempts} bootstrap iterations failed and were retried")

    # Convert list to array for aggregation
    spline_boot_array = np.array(spline_boot_list)  # Shape: (n_bootstrap, n_spline_points, n_coords)
    spline_boot_array = np.transpose(spline_boot_array, (1, 2, 0))  # Shape: (n_spline_points, n_coords, n_bootstrap)

    # Compute mean and standard error
    mean_spline = np.mean(spline_boot_array, axis=2)
    se_spline = np.std(spline_boot_array, axis=2)

    # Build output DataFrame
    se_cols = [col + "_se" for col in pca_cols]
    spline_df = pd.DataFrame(mean_spline, columns=pca_cols)
    spline_df[se_cols] = se_spline
    spline_df['spline_point_index'] = range(len(spline_df))

    return spline_df


# ============================================================================
# Main script
# ============================================================================

print("=" * 80)
print("TUTORIAL 05F: SPLINE FITTING WITH FIXED WRAPPER")
print("=" * 80)

# Load projection assignments
PROJECTION_DIR = OUTPUT_DIR / "figures" / "04" / "projection_results"
PROJECTION_CSV = PROJECTION_DIR / "combined_projection_bootstrap.csv"
print(f"\nLoading projections: {PROJECTION_CSV}")
proj = pd.read_csv(PROJECTION_CSV, low_memory=False)

# Load source trajectories
print("Loading source trajectories...")
source_dfs = []
for exp_id in SOURCE_EXPERIMENTS:
    df_exp = pd.read_csv(meta_dir / f"qc_staged_{exp_id}.csv", low_memory=False)
    df_exp = df_exp[df_exp["use_embryo_flag"]].copy()
    df_exp["experiment_id"] = exp_id
    source_dfs.append(df_exp)

df = pd.concat(source_dfs, ignore_index=True)

# Merge cluster labels
proj_cols = ["embryo_id", "cluster_label", "membership"]
proj = proj[proj_cols].drop_duplicates(subset="embryo_id")
df = df.merge(proj, on="embryo_id", how="inner")
df = df[df["cluster_label"].notna()].copy()

print(f"Trajectories with labels: {df['embryo_id'].nunique()} embryos")
print(f"Cluster labels: {sorted(df['cluster_label'].unique())}")

coords = ["baseline_deviation_normalized", "total_length_um"]
print(f"Using coordinates: {coords}")

# Fit splines using FIXED wrapper
print("\n" + "=" * 80)
print("FITTING SPLINES WITH RETRY LOGIC")
print("=" * 80)

spline_df = spline_fit_wrapper_fixed(
    df,
    group_by="cluster_label",
    pca_cols=coords,
    stage_col="predicted_stage_hpf",
    n_bootstrap=50,
    bootstrap_size=2500,
    n_spline_points=200,
    time_window=2,
    max_retries=3
)

print(f"\n✓ Fitted splines for {spline_df['cluster_label'].nunique()} clusters")
print(f"  Total rows: {len(spline_df)}")

# Check for NaNs
print("\nNaN check:")
for cluster in sorted(spline_df["cluster_label"].unique()):
    df_c = spline_df[spline_df["cluster_label"] == cluster]
    nans = df_c[coords[0]].isna().sum()
    print(f"  {cluster:20s}: {nans}/200 NaNs")

# Save results
csv_path = RESULTS_DIR / "05_projection_splines_FIXED.csv"
pkl_path = RESULTS_DIR / "05_projection_splines_FIXED.pkl"

spline_df.to_csv(csv_path, index=False)
print(f"\n✓ Saved: {csv_path}")

import pickle
with open(pkl_path, "wb") as f:
    pickle.dump(spline_df, f)
print(f"✓ Saved: {pkl_path}")

# Create visualization
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

clusters = sorted(df["cluster_label"].unique())
n_cols = 2
n_rows = int(np.ceil(len(clusters) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

for idx, cluster in enumerate(clusters):
    r, c = divmod(idx, n_cols)
    ax = axes[r][c]

    df_c = df[df["cluster_label"] == cluster]
    spline_c = spline_df[spline_df["cluster_label"] == cluster]

    # Raw data
    ax.scatter(
        df_c[coords[0]],
        df_c[coords[1]],
        s=6,
        alpha=0.2,
        color='gray',
        label="points"
    )

    # Spline
    if not spline_c.empty and not spline_c[coords[0]].isna().all():
        ax.plot(
            spline_c[coords[0]],
            spline_c[coords[1]],
            color="red",
            linewidth=2,
            label="spline"
        )

    ax.set_title(cluster)
    ax.set_xlabel(coords[0])
    ax.set_ylabel(coords[1])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

for idx in range(len(clusters), n_rows * n_cols):
    r, c = divmod(idx, n_cols)
    axes[r][c].axis("off")

fig.suptitle("Spline Fits per Cluster (FIXED)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])

fig_path = FIGURES_DIR / "05_projection_splines_FIXED.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {fig_path}")
plt.close(fig)

print("\n" + "=" * 80)
print("✓ Tutorial 05f complete - splines with fixed wrapper!")
print("=" * 80)
