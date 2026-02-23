"""
Analyze Morphological Metrics Distribution for Mask Quality Detection

Computes cheap morphological features on 500 random masks from part1 dataset
to calibrate "normal" vs "outlier" ranges for detecting masks that need opening.

Metrics computed (all from regionprops, no skeleton needed):
1. Solidity (area / convex_hull_area)
2. Circularity (perimeter² / 4π×area)
3. Extent (area / bbox_area)
4. Eccentricity (elongation measure)
5. Perimeter-to-area ratio (perimeter / sqrt(area))
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import measure
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle


def compute_morphology_metrics(mask: np.ndarray):
    """
    Compute cheap morphological metrics for a single mask.

    Returns dict with metrics.
    """
    props = measure.regionprops(measure.label(mask))[0]

    metrics = {
        'area': props.area,
        'perimeter': props.perimeter,
        'solidity': props.solidity,
        'extent': props.extent,
        'eccentricity': props.eccentricity,
    }

    # Derived metrics
    metrics['circularity'] = (props.perimeter ** 2) / (4 * np.pi * props.area)
    metrics['perimeter_area_ratio'] = props.perimeter / np.sqrt(props.area)

    return metrics


def load_and_compute_metrics(csv_path: Path, n_samples: int = 500, random_seed: int = 42):
    """
    Load random sample of embryos and compute morphology metrics.
    """
    print(f"Loading: {csv_path.name}")
    df = pd.read_csv(csv_path)

    # Random sample
    np.random.seed(random_seed)
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=random_seed)

    print(f"Computing metrics for {len(sample_df)} embryos...")

    results = []
    failed = 0

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        try:
            # Decode mask
            mask = decode_mask_rle({
                'size': [int(row['mask_height_px']), int(row['mask_width_px'])],
                'counts': row['mask_rle']
            })
            mask = np.ascontiguousarray(mask.astype(np.uint8))

            # Compute metrics
            metrics = compute_morphology_metrics(mask)
            metrics['snip_id'] = row['snip_id']
            metrics['dataset'] = 'part1_sample'

            results.append(metrics)

        except Exception as e:
            failed += 1
            print(f"Failed on {row['snip_id']}: {e}")

    print(f"Successfully processed: {len(results)}/{len(sample_df)} ({failed} failed)")

    return pd.DataFrame(results)


def add_problem_cases(results_df: pd.DataFrame):
    """
    Add known problem embryos from part2 dataset.
    """
    problem_snip_ids = [
        '20251017_part2_B05_e01_t0037',  # 2 components, spindly
        '20251017_part2_G07_e01_t0013',  # 8 components, very spindly
        '20251017_part2_B05_e01_t0005',  # test case
    ]

    csv_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251017_part2.csv")

    print(f"\nAdding problem cases from part2...")
    df = pd.read_csv(csv_path)

    problem_results = []

    for snip_id in problem_snip_ids:
        row = df[df['snip_id'] == snip_id]
        if len(row) == 0:
            print(f"  Warning: {snip_id} not found!")
            continue

        row = row.iloc[0]

        try:
            # Decode mask
            mask = decode_mask_rle({
                'size': [int(row['mask_height_px']), int(row['mask_width_px'])],
                'counts': row['mask_rle']
            })
            mask = np.ascontiguousarray(mask.astype(np.uint8))

            # Compute metrics
            metrics = compute_morphology_metrics(mask)
            metrics['snip_id'] = snip_id
            metrics['dataset'] = 'problem_case'

            problem_results.append(metrics)
            print(f"  Added: {snip_id}")

        except Exception as e:
            print(f"  Failed on {snip_id}: {e}")

    # Combine
    combined_df = pd.concat([results_df, pd.DataFrame(problem_results)], ignore_index=True)

    return combined_df


def plot_distributions(df: pd.DataFrame, output_dir: Path):
    """
    Plot distribution histograms for each metric.
    """
    metrics = ['solidity', 'circularity', 'extent', 'eccentricity', 'perimeter_area_ratio']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Separate normal samples from problem cases
    normal_df = df[df['dataset'] == 'part1_sample']
    problem_df = df[df['dataset'] == 'problem_case']

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Histogram of normal samples
        values = normal_df[metric].values
        ax.hist(values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')

        # Overlay problem cases
        if len(problem_df) > 0:
            problem_values = problem_df[metric].values
            for val, snip in zip(problem_values, problem_df['snip_id']):
                ax.axvline(val, color='red', linestyle='--', linewidth=2, alpha=0.7)

        # Add percentiles
        p5, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
        ax.axvline(p5, color='green', linestyle=':', linewidth=1.5, alpha=0.6, label=f'5th: {p5:.2f}')
        ax.axvline(p95, color='orange', linestyle=':', linewidth=1.5, alpha=0.6, label=f'95th: {p95:.2f}')
        ax.axvline(p50, color='purple', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Median: {p50:.2f}')

        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # Hide unused subplot
    axes[-1].axis('off')

    plt.tight_layout()
    output_path = output_dir / "morphology_distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    plt.close()


def plot_boxplots(df: pd.DataFrame, output_dir: Path):
    """
    Plot box plots to identify outliers.
    """
    metrics = ['solidity', 'circularity', 'extent', 'eccentricity', 'perimeter_area_ratio']

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    normal_df = df[df['dataset'] == 'part1_sample']
    problem_df = df[df['dataset'] == 'problem_case']

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Box plot of normal samples
        bp = ax.boxplot([normal_df[metric].values],
                        positions=[1],
                        widths=0.6,
                        patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))

        # Overlay problem cases as red dots
        if len(problem_df) > 0:
            problem_values = problem_df[metric].values
            ax.scatter([1] * len(problem_values), problem_values,
                      color='red', s=100, zorder=5, marker='D',
                      label='Problem Cases', edgecolors='black', linewidth=1.5)

        ax.set_xticks([1])
        ax.set_xticklabels([metric.replace('_', '\n').title()], fontsize=11)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

        if i == 0 and len(problem_df) > 0:
            ax.legend(fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "morphology_boxplots.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()


def plot_scatter_matrix(df: pd.DataFrame, output_dir: Path):
    """
    Plot scatter matrix showing relationships between metrics.
    """
    metrics = ['solidity', 'circularity', 'extent', 'eccentricity']

    normal_df = df[df['dataset'] == 'part1_sample']
    problem_df = df[df['dataset'] == 'problem_case']

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for i, metric_y in enumerate(metrics):
        for j, metric_x in enumerate(metrics):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram
                ax.hist(normal_df[metric_x].values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                ax.set_ylabel('Count', fontsize=9)
            else:
                # Off-diagonal: scatter
                ax.scatter(normal_df[metric_x], normal_df[metric_y],
                          alpha=0.3, s=10, color='steelblue')

                # Overlay problem cases
                if len(problem_df) > 0:
                    ax.scatter(problem_df[metric_x], problem_df[metric_y],
                              color='red', s=80, marker='D',
                              edgecolors='black', linewidth=1.5, zorder=5)

            # Labels
            if i == len(metrics) - 1:
                ax.set_xlabel(metric_x.replace('_', ' ').title(), fontsize=10)
            if j == 0:
                ax.set_ylabel(metric_y.replace('_', ' ').title(), fontsize=10)

            ax.grid(alpha=0.2)

    plt.tight_layout()
    output_path = output_dir / "morphology_scatter_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()


def generate_summary(df: pd.DataFrame, output_dir: Path):
    """
    Generate text summary with statistics and threshold recommendations.
    """
    metrics = ['solidity', 'circularity', 'extent', 'eccentricity', 'perimeter_area_ratio']

    normal_df = df[df['dataset'] == 'part1_sample']
    problem_df = df[df['dataset'] == 'problem_case']

    summary = []
    summary.append("="*80)
    summary.append("MORPHOLOGICAL METRICS ANALYSIS SUMMARY")
    summary.append("="*80)
    summary.append(f"\nDataset: part1 (n={len(normal_df)} samples)")
    summary.append(f"Problem cases from part2: n={len(problem_df)}")
    summary.append("\n" + "="*80)
    summary.append("DISTRIBUTION STATISTICS")
    summary.append("="*80)

    for metric in metrics:
        values = normal_df[metric].values

        summary.append(f"\n{metric.upper().replace('_', ' ')}:")
        summary.append(f"  Mean:   {values.mean():.4f}")
        summary.append(f"  Median: {np.median(values):.4f}")
        summary.append(f"  Std:    {values.std():.4f}")
        summary.append(f"  Min:    {values.min():.4f}")
        summary.append(f"  Max:    {values.max():.4f}")
        summary.append(f"  5th percentile:  {np.percentile(values, 5):.4f}")
        summary.append(f"  95th percentile: {np.percentile(values, 95):.4f}")

        # Problem case values
        if len(problem_df) > 0:
            problem_values = problem_df[metric].values
            summary.append(f"  Problem cases: {problem_values}")

    summary.append("\n" + "="*80)
    summary.append("THRESHOLD RECOMMENDATIONS FOR OPENING DETECTION")
    summary.append("="*80)

    # Solidity
    solidity_5th = np.percentile(normal_df['solidity'].values, 5)
    summary.append(f"\n1. SOLIDITY < {solidity_5th:.3f} (5th percentile)")
    summary.append(f"   → Catches bottom 5% of normal distribution")
    if len(problem_df) > 0:
        n_caught = (problem_df['solidity'] < solidity_5th).sum()
        summary.append(f"   → Catches {n_caught}/{len(problem_df)} problem cases")

    # Circularity
    circularity_95th = np.percentile(normal_df['circularity'].values, 95)
    summary.append(f"\n2. CIRCULARITY > {circularity_95th:.3f} (95th percentile)")
    summary.append(f"   → Catches top 5% of normal distribution (most irregular)")
    if len(problem_df) > 0:
        n_caught = (problem_df['circularity'] > circularity_95th).sum()
        summary.append(f"   → Catches {n_caught}/{len(problem_df)} problem cases")

    # Extent
    extent_5th = np.percentile(normal_df['extent'].values, 5)
    summary.append(f"\n3. EXTENT < {extent_5th:.3f} (5th percentile)")
    summary.append(f"   → Catches bottom 5% (least bbox-filling)")
    if len(problem_df) > 0:
        n_caught = (problem_df['extent'] < extent_5th).sum()
        summary.append(f"   → Catches {n_caught}/{len(problem_df)} problem cases")

    # Combined rule
    summary.append(f"\n4. COMBINED RULE: solidity < {solidity_5th:.3f} OR circularity > {circularity_95th:.3f}")
    if len(problem_df) > 0:
        n_caught = ((problem_df['solidity'] < solidity_5th) |
                   (problem_df['circularity'] > circularity_95th)).sum()
        summary.append(f"   → Catches {n_caught}/{len(problem_df)} problem cases")

    summary.append("\n" + "="*80)
    summary.append("PROBLEM CASE DETAILS")
    summary.append("="*80)

    if len(problem_df) > 0:
        for idx, row in problem_df.iterrows():
            summary.append(f"\n{row['snip_id']}:")
            summary.append(f"  Solidity:     {row['solidity']:.4f}")
            summary.append(f"  Circularity:  {row['circularity']:.4f}")
            summary.append(f"  Extent:       {row['extent']:.4f}")
            summary.append(f"  Eccentricity: {row['eccentricity']:.4f}")
            summary.append(f"  Perim/Area:   {row['perimeter_area_ratio']:.4f}")

    summary.append("\n" + "="*80)

    # Write to file
    output_path = output_dir / "morphology_summary.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary))

    print(f"Saved: {output_path}")

    # Also print to console
    print('\n'.join(summary))


def main():
    """Main execution."""
    csv_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251017_part1.csv")
    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251027")

    print("="*80)
    print("MORPHOLOGICAL METRICS DISTRIBUTION ANALYSIS")
    print("="*80)

    # Compute metrics on 500 random samples from part1
    results_df = load_and_compute_metrics(csv_path, n_samples=500)

    # Add known problem cases from part2
    results_df = add_problem_cases(results_df)

    # Save results
    results_csv = output_dir / "morphology_metrics_500samples.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved: {results_csv}")

    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    plot_distributions(results_df, output_dir)
    plot_boxplots(results_df, output_dir)
    plot_scatter_matrix(results_df, output_dir)

    # Generate summary
    print("\n" + "="*80)
    print("GENERATING SUMMARY")
    print("="*80)
    generate_summary(results_df, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
