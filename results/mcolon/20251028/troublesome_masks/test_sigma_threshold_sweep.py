"""
Test varying BOTH Gaussian blur sigma AND threshold on problematic embryos.

For each embryo, test combinations of:
- Sigma values: 15, 20
- Threshold values: 0.5, 0.6, 0.7, 0.8, 0.9

Save results to sigma_threshold_sweep/ subfolder showing preprocessed masks
and resulting centerlines for each combination.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask
from segmentation_sandbox.scripts.body_axis_analysis.mask_preprocessing import apply_preprocessing
from segmentation_sandbox.scripts.body_axis_analysis.geodesic_method import GeodesicCenterlineAnalyzer


def load_embryo_data(snip_id: str):
    """Load embryo from CSV."""
    metadata_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output")

    if "20251017" in snip_id:
        csv_path = metadata_dir / "df03_final_output_with_latents_20251017_combined.csv"
    elif "20250512" in snip_id:
        csv_path = metadata_dir / "df03_final_output_with_latents_20250512.csv"
    else:
        raise ValueError(f"Cannot determine CSV file for snip_id: {snip_id}")

    df = pd.read_csv(csv_path)
    embryo_row = df[df['snip_id'] == snip_id].iloc[0]

    mask = decode_mask_rle({
        'size': [int(embryo_row['mask_height_px']), int(embryo_row['mask_width_px'])],
        'counts': embryo_row['mask_rle']
    })

    um_per_pixel = embryo_row['height_um'] / int(embryo_row['mask_height_px'])

    return mask, um_per_pixel


def test_embryo_sigma_threshold_sweep(snip_id: str, sigma_values: list, threshold_values: list, output_dir: Path):
    """Test a single embryo with multiple sigma and threshold combinations."""
    print(f"\n{'='*120}")
    print(f"TESTING: {snip_id}")
    print(f"{'='*120}")

    # Load data
    mask, um_per_pixel = load_embryo_data(snip_id)
    mask_cleaned, _ = clean_embryo_mask(mask, verbose=False)

    print(f"Original mask: {mask.shape}, area={mask.sum():,} px")
    print(f"Cleaned mask: area={mask_cleaned.sum():,} px")

    results_by_params = {}
    summary_rows = []

    for sigma in sigma_values:
        for threshold in threshold_values:
            params_key = (sigma, threshold)
            print(f"  σ={sigma}, θ={threshold}...", end=" ", flush=True)

            try:
                # Apply preprocessing with these parameters
                mask_preprocessed = apply_preprocessing(
                    mask_cleaned,
                    method='gaussian_blur',
                    sigma=sigma,
                    threshold=threshold
                )

                # Run geodesic centerline extraction
                analyzer = GeodesicCenterlineAnalyzer(
                    mask_preprocessed,
                    um_per_pixel=um_per_pixel,
                    bspline_smoothing=5.0,
                    random_seed=42,
                    fast=True
                )

                results = analyzer.analyze()

                results_by_params[params_key] = {
                    'success': True,
                    'preprocessed_mask': mask_preprocessed,
                    'centerline_raw': results['centerline_raw'],
                    'centerline_smoothed': results['centerline_smoothed'],
                    'skeleton': results['skeleton'],
                    'endpoints': results['endpoints'],
                    'length_um': results['stats']['total_length'],
                    'mean_curvature': results['stats']['mean_curvature'],
                    'n_centerline_points': results['stats']['n_centerline_points'],
                }

                print(f"✓ {results['stats']['total_length']:.1f}μm")

                summary_rows.append({
                    'embryo_id': snip_id,
                    'sigma': sigma,
                    'threshold': threshold,
                    'success': True,
                    'length_um': results['stats']['total_length'],
                    'mean_curvature': results['stats']['mean_curvature'],
                    'n_centerline_points': results['stats']['n_centerline_points'],
                })

            except Exception as e:
                results_by_params[params_key] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"✗ {str(e)[:40]}")

                summary_rows.append({
                    'embryo_id': snip_id,
                    'sigma': sigma,
                    'threshold': threshold,
                    'success': False,
                    'length_um': None,
                    'mean_curvature': None,
                    'n_centerline_points': None,
                })

    # Create comprehensive visualization grid
    n_sigmas = len(sigma_values)
    n_thresholds = len(threshold_values)

    fig = plt.figure(figsize=(5 * n_thresholds, 6 * n_sigmas))
    gs = GridSpec(n_sigmas, n_thresholds, figure=fig, hspace=0.4, wspace=0.3)

    for sigma_idx, sigma in enumerate(sigma_values):
        for threshold_idx, threshold in enumerate(threshold_values):
            params_key = (sigma, threshold)
            ax = fig.add_subplot(gs[sigma_idx, threshold_idx])

            if params_key not in results_by_params or not results_by_params[params_key]['success']:
                ax.axis('off')
                error_msg = results_by_params.get(params_key, {}).get('error', 'Failed')
                ax.text(0.5, 0.5, f"σ={sigma}\nθ={threshold}\nFAILED\n{error_msg[:30]}",
                       ha='center', va='center', fontsize=10, color='red',
                       transform=ax.transAxes)
                continue

            result = results_by_params[params_key]

            # Show preprocessed mask with centerline overlay
            ax.imshow(mask, cmap='gray', alpha=0.5)
            ax.imshow(result['preprocessed_mask'], cmap='Blues', alpha=0.4)
            ax.plot(result['centerline_smoothed'][:, 0], result['centerline_smoothed'][:, 1], 'r-',
                   linewidth=2, label='Centerline')
            ax.plot(result['endpoints'][:, 0], result['endpoints'][:, 1], 'g*', markersize=12)

            ax.set_title(f"σ={sigma}, θ={threshold}\n"
                        f"L={result['length_um']:.0f}μm, κ={result['mean_curvature']:.6f}",
                       fontsize=9, fontweight='bold')
            ax.axis('tight')

    plt.suptitle(f'Sigma-Threshold Sweep: {snip_id}', fontsize=14, fontweight='bold', y=0.995)
    output_path = output_dir / f"{snip_id}_sigma_threshold_sweep.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / f"{snip_id}_sigma_threshold_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✓ Saved summary: {summary_csv_path}")

    # Print summary table
    print(f"\n{'SIGMA':<8} {'THRESHOLD':<12} {'SUCCESS':<10} {'LENGTH (μm)':<15} {'MEAN κ':<15}")
    print("-" * 70)
    for sigma in sigma_values:
        for threshold in threshold_values:
            params_key = (sigma, threshold)
            if params_key in results_by_params and results_by_params[params_key]['success']:
                r = results_by_params[params_key]
                print(f"{sigma:<8} {threshold:<12} {'✓':<10} {r['length_um']:<15.2f} {r['mean_curvature']:<15.6f}")
            else:
                print(f"{sigma:<8} {threshold:<12} {'✗':<10} {'FAILED':<15} {'':<15}")

    return results_by_params, summary_df


if __name__ == "__main__":
    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251028/troublesome_masks/sigma_threshold_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test these embryos with sigma and threshold combinations
    embryos = [
        "20251017_combined_D11_e01_t0073",
        "20251017_combined_D12_e01_t0059",
        # "20251017_combined_A02_e01_t0064",
        # "20251017_combined_D12_e01_t0057",
        # "20251017_combined_D10_e01_t0108",
        # "20250512_E06_e01_t0086",  # Curvy embryo - baseline for comparison
    ]

    sigma_values = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    threshold_values = [0.4 ,0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n{'='*120}")
    print("SIGMA-THRESHOLD SWEEP ON PROBLEMATIC EMBRYOS (INCLUDING BASELINE CURVY)")
    print(f"{'='*120}")
    print(f"Sigma values: {sigma_values}")
    print(f"Threshold values: {threshold_values}")
    print(f"Output directory: {output_dir}")

    all_results = {}
    all_summary_dfs = []

    for embryo_id in embryos:
        try:
            results, summary_df = test_embryo_sigma_threshold_sweep(
                embryo_id, sigma_values, threshold_values, output_dir
            )
            all_results[embryo_id] = results
            all_summary_dfs.append(summary_df)
        except Exception as e:
            print(f"✗ Failed to process {embryo_id}: {e}")
            import traceback
            traceback.print_exc()

    # Combine all summary CSVs
    if all_summary_dfs:
        combined_summary = pd.concat(all_summary_dfs, ignore_index=True)
        combined_csv_path = output_dir / "combined_sigma_threshold_summary.csv"
        combined_summary.to_csv(combined_csv_path, index=False)
        print(f"\n✓ Saved combined summary: {combined_csv_path}")

    print(f"\n{'='*120}")
    print("✓ SIGMA-THRESHOLD SWEEP COMPLETE")
    print(f"{'='*120}\n")
