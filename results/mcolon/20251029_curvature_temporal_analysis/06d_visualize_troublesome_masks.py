"""
Visualize masks and splines for outlier snip_ids to diagnose whether
curvature extremes are real biological variation or segmentation/spline artifacts.

For each "spiky" WT embryo identified by 06c, creates a comparison grid showing:
- Outlier timepoints (where embryo was flagged as penetrant outside IQR bounds)
- Clean timepoints (same embryo, non-flagged times)

Each panel shows:
- Binary mask (grayscale)
- Spline overlay (red line with head/tail endpoints)
- Curvature metrics annotated (arc_length_ratio, baseline_deviation_um)
- Outlier status labeled
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from typing import Dict, List, Tuple

# Add project root
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from load_data import get_analysis_dataframe

# ============================================================================
# Configuration
# ============================================================================

METRIC_NAME = 'normalized_baseline_deviation'
TIME_BIN_WIDTH = 2.0
WT_GENOTYPE = 'cep290_wildtype'
K_PARAM = 2.0  # IQR ±2.0σ for threshold

# Paths
METADATA_DIR = repo_root / 'morphseq_playground/metadata/build06_output'
CURVATURE_ARRAYS_PATH = (
    repo_root / 'morphseq_playground/metadata/body_axis/arrays/curvature_arrays_20251017_combined.csv'
)
CURVATURE_SUMMARY_PATH = (
    repo_root / 'morphseq_playground/metadata/body_axis/summary/curvature_metrics_summary_20251017_combined.csv'
)

OUTPUT_DIR = Path(__file__).parent / 'outputs' / '06d_troublesome_masks'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_mask_for_snip_id(snip_id: str) -> Tuple[np.ndarray, pd.Series]:
    """
    Load binary mask and metadata for a specific snip_id.

    Parameters
    ----------
    snip_id : str
        Unique identifier (e.g., "20251017_combined_A01_e01_t0004")

    Returns
    -------
    mask : np.ndarray
        Binary mask (H, W) where 1 = embryo, 0 = background
    metadata_row : pd.Series
        Row from metadata CSV containing mask dimensions, genotype, etc.
    """
    csv_path = METADATA_DIR / "df03_final_output_with_latents_20251017_combined.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    rows = df[df['snip_id'] == snip_id]

    if len(rows) == 0:
        raise ValueError(f"snip_id not found in metadata: {snip_id}")

    row = rows.iloc[0]

    # Decode RLE mask
    mask = decode_mask_rle({
        'size': [int(row['mask_height_px']), int(row['mask_width_px'])],
        'counts': row['mask_rle']
    })

    return mask, row


def load_spline_for_snip_id(snip_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spline coordinates for a specific snip_id.

    Parameters
    ----------
    snip_id : str
        Unique identifier

    Returns
    -------
    centerline_x : np.ndarray
        X coordinates (pixel) of spline (N,)
    centerline_y : np.ndarray
        Y coordinates (pixel) of spline (N,)
    """
    if not CURVATURE_ARRAYS_PATH.exists():
        raise FileNotFoundError(f"Curvature arrays CSV not found: {CURVATURE_ARRAYS_PATH}")

    arrays_df = pd.read_csv(CURVATURE_ARRAYS_PATH)
    rows = arrays_df[arrays_df['snip_id'] == snip_id]

    if len(rows) == 0:
        raise ValueError(f"snip_id not found in spline data: {snip_id}")

    row = rows.iloc[0]

    # Parse JSON arrays
    centerline_x = np.array(json.loads(row['centerline_x']))
    centerline_y = np.array(json.loads(row['centerline_y']))

    return centerline_x, centerline_y


def load_curvature_metrics(snip_id: str) -> Dict[str, float]:
    """
    Load curvature metrics for a specific snip_id.

    Parameters
    ----------
    snip_id : str

    Returns
    -------
    dict
        Dictionary with keys: arc_length_ratio, baseline_deviation_um,
        normalized_baseline_deviation, total_length_um, mean_curvature_per_um
    """
    if not CURVATURE_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Curvature summary CSV not found: {CURVATURE_SUMMARY_PATH}")

    curv_df = pd.read_csv(CURVATURE_SUMMARY_PATH)
    rows = curv_df[curv_df['snip_id'] == snip_id]

    if len(rows) == 0:
        return {}

    row = rows.iloc[0]

    return {
        'arc_length_ratio': row.get('arc_length_ratio', np.nan),
        'baseline_deviation_um': row.get('baseline_deviation_um', np.nan),
        'total_length_um': row.get('total_length_um', np.nan),
        'mean_curvature_per_um': row.get('mean_curvature_per_um', np.nan),
    }


# ============================================================================
# Outlier Detection (WT reference bands)
# ============================================================================

def bin_data_by_time(df, bin_width=2.0, time_col='predicted_stage_hpf'):
    """Bin data by developmental time."""
    df = df.copy()
    min_time = df[time_col].min()
    max_time = df[time_col].max()

    bin_edges = np.arange(
        np.floor(min_time / bin_width) * bin_width,
        np.ceil(max_time / bin_width) * bin_width + bin_width,
        bin_width
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    df['time_bin'] = pd.cut(df[time_col], bins=bin_edges, labels=bin_centers)
    df['time_bin'] = df['time_bin'].astype(float)

    return df, bin_centers


def compute_iqr_bounds(wt_df, time_bins, metric=METRIC_NAME, k=2.0):
    """Compute IQR bounds per time bin."""
    envelope = {}

    for time_bin in time_bins:
        bin_df = wt_df[wt_df['time_bin'] == time_bin]
        if len(bin_df) == 0:
            continue

        values = bin_df[metric].values
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        envelope[time_bin] = {
            'low': q1 - k * iqr,
            'high': q3 + k * iqr,
            'median': np.median(values),
        }

    return envelope


def find_outlier_snip_ids_for_embryo(embryo_id: str, df: pd.DataFrame,
                                      envelope: Dict) -> Tuple[List[str], List[str]]:
    """
    Find outlier and clean snip_ids for a specific embryo.

    Returns
    -------
    outlier_snip_ids : list of str
        snip_ids where embryo was outside IQR bounds
    clean_snip_ids : list of str
        snip_ids where embryo was within bounds
    """
    embryo_df = df[df['embryo_id'] == embryo_id].sort_values('predicted_stage_hpf')

    outlier_snips = []
    clean_snips = []

    for _, row in embryo_df.iterrows():
        time_bin = row['time_bin']
        if pd.isna(time_bin) or time_bin not in envelope:
            continue

        metric_val = row[METRIC_NAME]
        bounds = envelope[time_bin]

        snip_id = row['snip_id']

        if metric_val < bounds['low'] or metric_val > bounds['high']:
            outlier_snips.append(snip_id)
        else:
            clean_snips.append(snip_id)

    return outlier_snips, clean_snips


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_snip_id_panel(ax, snip_id: str, metric_values: Dict[str, float],
                             is_outlier: bool = False):
    """
    Create a single visualization panel for one snip_id.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    snip_id : str
    metric_values : dict
        Dictionary with metric values (arc_length_ratio, baseline_deviation_um, etc.)
    is_outlier : bool
        Whether this snip_id was flagged as an outlier
    """
    try:
        # Load data
        mask, _ = load_mask_for_snip_id(snip_id)
        centerline_x, centerline_y = load_spline_for_snip_id(snip_id)

        # Display mask
        ax.imshow(mask, cmap='gray', alpha=0.7)

        # Overlay spline
        ax.plot(centerline_x, centerline_y, 'r-', linewidth=2.5, label='Centerline')

        # Mark endpoints
        ax.plot(centerline_x[0], centerline_y[0], 'go', markersize=10, label='Head', zorder=10)
        ax.plot(centerline_x[-1], centerline_y[-1], 'bo', markersize=10, label='Tail', zorder=10)

        # Format title with metrics
        arc_length_ratio = metric_values.get('arc_length_ratio', np.nan)
        baseline_dev = metric_values.get('baseline_deviation_um', np.nan)

        title = f"{snip_id.split('_')[-1]}\n"  # Just the time index

        if not np.isnan(arc_length_ratio):
            title += f"ALR: {arc_length_ratio:.3f} | "
        if not np.isnan(baseline_dev):
            title += f"BL Dev: {baseline_dev:.2f} μm\n"

        if is_outlier:
            title += "⚠ OUTLIER"
            title_color = 'red'
            title_weight = 'bold'
        else:
            title += "✓ Clean"
            title_color = 'green'
            title_weight = 'normal'

        ax.set_title(title, fontsize=10, fontweight=title_weight, color=title_color)
        ax.set_xticks([])
        ax.set_yticks([])

    except Exception as e:
        ax.text(0.5, 0.5, f"Error loading:\n{str(e)[:50]}",
                ha='center', va='center', transform=ax.transAxes,
                color='red', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])


def create_comparison_grid(embryo_id: str, outlier_snip_ids: List[str],
                           clean_snip_ids: List[str], output_path: Path):
    """
    Create 2-column comparison grid for one embryo.

    Left column: outlier snip_ids
    Right column: clean snip_ids
    """
    # Select up to 3 of each
    n_rows = max(len(outlier_snip_ids), len(clean_snip_ids))
    n_rows = min(n_rows, 4)  # Limit to 4 rows max

    selected_outliers = outlier_snip_ids[:n_rows]
    selected_clean = clean_snip_ids[:n_rows]

    # Pad with None if needed
    while len(selected_outliers) < n_rows:
        selected_outliers.append(None)
    while len(selected_clean) < n_rows:
        selected_clean.append(None)

    # Create grid
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Embryo {embryo_id}: Outliers (Left) vs Clean (Right)',
                 fontsize=14, fontweight='bold', y=0.995)

    # Load metrics once
    all_snips = [s for s in selected_outliers + selected_clean if s is not None]
    metrics_cache = {}
    for snip_id in all_snips:
        try:
            metrics_cache[snip_id] = load_curvature_metrics(snip_id)
        except:
            metrics_cache[snip_id] = {}

    # Fill left column (outliers)
    for i, snip_id in enumerate(selected_outliers):
        if snip_id is None:
            axes[i, 0].axis('off')
        else:
            visualize_snip_id_panel(axes[i, 0], snip_id,
                                    metrics_cache.get(snip_id, {}),
                                    is_outlier=True)

    # Fill right column (clean)
    for i, snip_id in enumerate(selected_clean):
        if snip_id is None:
            axes[i, 1].axis('off')
        else:
            visualize_snip_id_panel(axes[i, 1], snip_id,
                                    metrics_cache.get(snip_id, {}),
                                    is_outlier=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    print("=" * 80)
    print("VISUALIZATION: Troublesome Masks & Splines")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df, _ = get_analysis_dataframe()

    # Bin by time
    print("Binning data by time...")
    df, time_bins = bin_data_by_time(df, bin_width=TIME_BIN_WIDTH)

    # Extract WT
    wt_df = df[df['genotype'] == WT_GENOTYPE].copy()
    print(f"WT: {wt_df['embryo_id'].nunique()} embryos, {len(wt_df)} timepoints")

    # Compute IQR bounds
    print(f"Computing IQR ±{K_PARAM}σ bounds...")
    envelope = compute_iqr_bounds(wt_df, time_bins, metric=METRIC_NAME, k=K_PARAM)

    # Count outliers per embryo
    print("Counting outliers per embryo...")
    outlier_counts = {}
    for embryo_id in wt_df['embryo_id'].unique():
        embryo_df = wt_df[wt_df['embryo_id'] == embryo_id]
        n_outliers = 0
        for _, row in embryo_df.iterrows():
            time_bin = row['time_bin']
            if pd.isna(time_bin) or time_bin not in envelope:
                continue
            metric_val = row[METRIC_NAME]
            bounds = envelope[time_bin]
            if metric_val < bounds['low'] or metric_val > bounds['high']:
                n_outliers += 1
        outlier_counts[embryo_id] = n_outliers

    # Find top 3 spiky embryos
    sorted_embryos = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)
    top_3_spiky = [e for e, c in sorted_embryos[:3]]

    print(f"\nTop 3 spiky embryos (most outliers):")
    for embryo_id, count in sorted_embryos[:3]:
        print(f"  {embryo_id}: {count} outliers")

    # Create visualizations
    print(f"\nGenerating visualizations...")

    summary_data = []

    for embryo_id in top_3_spiky:
        print(f"\n  Processing {embryo_id}...")

        # Find outlier and clean snip_ids
        outlier_snips, clean_snips = find_outlier_snip_ids_for_embryo(
            embryo_id, wt_df, envelope
        )

        print(f"    Found {len(outlier_snips)} outliers, {len(clean_snips)} clean timepoints")

        if len(outlier_snips) > 0:
            # Create comparison grid
            output_path = OUTPUT_DIR / f"embryo_{embryo_id}_comparison.png"
            create_comparison_grid(embryo_id, outlier_snips, clean_snips, output_path)

            # Log summary
            for snip_id in outlier_snips:
                try:
                    metrics = load_curvature_metrics(snip_id)
                    summary_data.append({
                        'embryo_id': embryo_id,
                        'snip_id': snip_id,
                        'status': 'OUTLIER',
                        'arc_length_ratio': metrics.get('arc_length_ratio'),
                        'baseline_deviation_um': metrics.get('baseline_deviation_um'),
                        'total_length_um': metrics.get('total_length_um'),
                    })
                except:
                    pass

            for snip_id in clean_snips[:3]:  # Limit clean samples
                try:
                    metrics = load_curvature_metrics(snip_id)
                    summary_data.append({
                        'embryo_id': embryo_id,
                        'snip_id': snip_id,
                        'status': 'CLEAN',
                        'arc_length_ratio': metrics.get('arc_length_ratio'),
                        'baseline_deviation_um': metrics.get('baseline_deviation_um'),
                        'total_length_um': metrics.get('total_length_um'),
                    })
                except:
                    pass

    # Save summary table
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = OUTPUT_DIR / 'summary_table.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Saved summary table: {summary_path.name}")

    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"\nInterpretation guide:")
    print(f"- Left panels: Outlier timepoints (flagged as penetrant)")
    print(f"- Right panels: Clean timepoints (within IQR bounds)")
    print(f"- Check if outliers show genuinely bent embryos (biology)")
    print(f"- Or poorly fit splines (segmentation artifacts)")


if __name__ == '__main__':
    main()
