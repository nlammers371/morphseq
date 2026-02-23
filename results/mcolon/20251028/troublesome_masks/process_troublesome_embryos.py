"""
Process Troublesome Embryos - Phase 1: Reproduce Current Behavior

Processes 4-5 embryos that failed in previous batch processing with the current
default pipeline to determine if failures persist or have been resolved.

Outputs:
- Individual visualizations showing processing stages
- Summary grid of all embryos
- CSV with detailed results
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask
from segmentation_sandbox.scripts.body_axis_analysis.centerline_extraction import extract_centerline
from segmentation_sandbox.scripts.body_axis_analysis.mask_preprocessing import apply_preprocessing


# =============================================================================
# EMBRYO LOADING (reusing logic from batch processing)
# =============================================================================

def load_embryo_data(snip_id: str):
    """
    Load embryo mask and metadata from CSV files.
    Reuses the same logic as process_curvature_batch.py

    Args:
        snip_id: Embryo identifier

    Returns:
        mask, um_per_pixel, embryo_metadata
    """
    metadata_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output")

    # Determine which CSV to load based on snip_id
    if "20251017" in snip_id:
        csv_path = metadata_dir / "df03_final_output_with_latents_20251017_combined.csv"
    elif "20250512" in snip_id:
        csv_path = metadata_dir / "df03_final_output_with_latents_20250512.csv"
    else:
        raise ValueError(f"Cannot determine CSV file for snip_id: {snip_id}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load metadata
    df = pd.read_csv(csv_path)
    embryo_row = df[df['snip_id'] == snip_id]

    if len(embryo_row) == 0:
        raise ValueError(f"Embryo {snip_id} not found in {csv_path}")

    embryo_data = embryo_row.iloc[0]

    # Decode mask (same as batch processing)
    mask = decode_mask_rle({
        'size': [int(embryo_data['mask_height_px']), int(embryo_data['mask_width_px'])],
        'counts': embryo_data['mask_rle']
    })

    # Calculate um_per_pixel (same as batch processing)
    um_per_pixel = embryo_data['height_um'] / int(embryo_data['mask_height_px'])

    return mask, um_per_pixel, embryo_data


# =============================================================================
# PROCESSING PIPELINE
# =============================================================================

def process_embryo_with_stages(snip_id: str):
    """
    Process embryo through full pipeline and capture intermediate stages.

    Args:
        snip_id: Embryo identifier

    Returns:
        Dictionary with all processing stages and results
    """
    print(f"\n{'='*80}")
    print(f"Processing: {snip_id}")
    print(f"{'='*80}")

    result = {
        'snip_id': snip_id,
        'success': False,
        'error': None,
    }

    try:
        # Load embryo
        print("Loading embryo data...")
        t0 = time.perf_counter()
        mask_original, um_per_pixel, metadata = load_embryo_data(snip_id)
        load_time = (time.perf_counter() - t0) * 1000

        print(f"  Mask shape: {mask_original.shape}")
        print(f"  Mask area: {mask_original.sum():,} px")
        print(f"  um_per_pixel: {um_per_pixel:.4f}")
        print(f"  Load time: {load_time:.1f} ms")

        # Stage 1: Clean mask (5-step pipeline)
        print("\nStage 1: Mask cleaning (5-step pipeline)...")
        t0 = time.perf_counter()
        mask_cleaned, cleaning_stats = clean_embryo_mask(mask_original, verbose=False)
        cleaning_time = (time.perf_counter() - t0) * 1000

        print(f"  Cleaned area: {mask_cleaned.sum():,} px ({mask_cleaned.sum()/mask_original.sum()*100:.1f}%)")
        print(f"  Cleaning time: {cleaning_time:.1f} ms")

        # Stage 2: Preprocess mask (Gaussian blur)
        print("\nStage 2: Preprocessing (Gaussian blur)...")
        t0 = time.perf_counter()
        mask_preprocessed = apply_preprocessing(mask_cleaned, method='gaussian_blur', sigma=10.0, threshold=0.7)
        preprocess_time = (time.perf_counter() - t0) * 1000

        print(f"  Preprocessed area: {mask_preprocessed.sum():,} px ({mask_preprocessed.sum()/mask_cleaned.sum()*100:.1f}%)")
        print(f"  Preprocess time: {preprocess_time:.1f} ms")

        # Stage 3: Extract centerline (Geodesic method)
        print("\nStage 3: Centerline extraction (Geodesic)...")
        t0 = time.perf_counter()
        spline_x, spline_y, curvature, arc_length = extract_centerline(
            mask_cleaned,  # This will apply preprocessing internally
            method='geodesic',
            um_per_pixel=um_per_pixel,
            bspline_smoothing=5.0
        )
        extraction_time = (time.perf_counter() - t0) * 1000

        # Check success
        success = len(spline_x) > 0 and len(curvature) > 0

        if success:
            print(f"  ✓ SUCCESS")
            print(f"  Centerline points: {len(spline_x)}")
            print(f"  Total length: {arc_length[-1]:.2f} μm")
            print(f"  Mean curvature: {np.mean(np.abs(curvature)):.4f} 1/μm")
            print(f"  Max curvature: {np.max(np.abs(curvature)):.4f} 1/μm")
            print(f"  Extraction time: {extraction_time:.1f} ms")

            # Store results
            result.update({
                'success': True,
                'n_centerline_points': len(spline_x),
                'total_length_um': float(arc_length[-1]),
                'mean_curvature_per_um': float(np.mean(np.abs(curvature))),
                'std_curvature_per_um': float(np.std(curvature)),
                'max_curvature_per_um': float(np.max(np.abs(curvature))),
                'load_time_ms': load_time,
                'cleaning_time_ms': cleaning_time,
                'preprocess_time_ms': preprocess_time,
                'extraction_time_ms': extraction_time,
                'total_time_ms': load_time + cleaning_time + preprocess_time + extraction_time,
            })
        else:
            print(f"  ✗ FAILED - Empty centerline")
            result['error'] = "Empty centerline returned"

        # Store intermediate stages for visualization
        result.update({
            'mask_original': mask_original,
            'mask_cleaned': mask_cleaned,
            'mask_preprocessed': mask_preprocessed,
            'spline_x': spline_x,
            'spline_y': spline_y,
            'curvature': curvature,
            'arc_length': arc_length,
            'um_per_pixel': um_per_pixel,
        })

    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        result['error'] = str(e)

    return result


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_individual_embryo(result: dict, output_path: Path):
    """
    Generate 4-panel visualization showing processing stages.

    Layout:
        [Original Mask]     [Cleaned Mask]
        [Preprocessed]      [Final + Centerline]
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    snip_id = result['snip_id']
    success = result['success']

    # Panel 1: Original mask
    ax = axes[0, 0]
    ax.imshow(result['mask_original'], cmap='gray', alpha=0.7)
    ax.set_title(f"Stage 0: Original Mask\nArea: {result['mask_original'].sum():,} px",
                 fontsize=12, fontweight='bold')
    ax.axis('off')

    # Panel 2: Cleaned mask
    ax = axes[0, 1]
    ax.imshow(result['mask_cleaned'], cmap='gray', alpha=0.7)
    cleaned_pct = result['mask_cleaned'].sum() / result['mask_original'].sum() * 100
    ax.set_title(f"Stage 1: Cleaned Mask (5-step pipeline)\nArea: {result['mask_cleaned'].sum():,} px ({cleaned_pct:.1f}%)",
                 fontsize=12, fontweight='bold')
    ax.axis('off')

    # Panel 3: Preprocessed mask
    ax = axes[1, 0]
    ax.imshow(result['mask_preprocessed'], cmap='gray', alpha=0.7)
    preproc_pct = result['mask_preprocessed'].sum() / result['mask_cleaned'].sum() * 100
    ax.set_title(f"Stage 2: Preprocessed (Gaussian blur σ=10)\nArea: {result['mask_preprocessed'].sum():,} px ({preproc_pct:.1f}%)",
                 fontsize=12, fontweight='bold')
    ax.axis('off')

    # Panel 4: Final with centerline
    ax = axes[1, 1]
    ax.imshow(result['mask_cleaned'], cmap='gray', alpha=0.7)

    if success and len(result['spline_x']) > 0:
        ax.plot(result['spline_x'], result['spline_y'], 'r-', linewidth=2.5, label='Centerline')
        ax.plot(result['spline_x'][0], result['spline_y'][0], 'go', markersize=10,
                label='Head', markeredgecolor='white', markeredgewidth=1.5)
        ax.plot(result['spline_x'][-1], result['spline_y'][-1], 'bo', markersize=10,
                label='Tail', markeredgecolor='white', markeredgewidth=1.5)

        title = f"Stage 3: Centerline Extraction ✓\n"
        title += f"Length: {result['total_length_um']:.1f} μm, "
        title += f"Mean κ: {result['mean_curvature_per_um']:.4f} 1/μm"
        ax.set_title(title, fontsize=12, fontweight='bold', color='green')
        ax.legend(fontsize=10)
    else:
        title = f"Stage 3: Centerline Extraction ✗ FAILED\n"
        title += f"Error: {result.get('error', 'Unknown')}"
        ax.set_title(title, fontsize=12, fontweight='bold', color='red')

    ax.axis('off')

    # Main title
    status = "✓ SUCCESS" if success else "✗ FAILED"
    color = 'green' if success else 'red'
    fig.suptitle(f"{snip_id} - {status}", fontsize=16, fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_summary_grid(results: list, output_path: Path):
    """
    Generate summary grid showing all embryos.

    Layout: 2 rows × ceil(n/2) columns
    """
    n_embryos = len(results)
    n_cols = min(3, (n_embryos + 1) // 2)
    n_rows = (n_embryos + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 8*n_rows))

    # Flatten axes for easier iteration
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, result in enumerate(results):
        ax = axes[idx]

        # Show cleaned mask with centerline overlay
        ax.imshow(result['mask_cleaned'], cmap='gray', alpha=0.7)

        if result['success'] and len(result['spline_x']) > 0:
            ax.plot(result['spline_x'], result['spline_y'], 'r-', linewidth=2, label='Centerline')
            ax.plot(result['spline_x'][0], result['spline_y'][0], 'go', markersize=8, label='Head')
            ax.plot(result['spline_x'][-1], result['spline_y'][-1], 'bo', markersize=8, label='Tail')

            title = f"{result['snip_id']}\n✓ SUCCESS\n"
            title += f"Length: {result['total_length_um']:.1f} μm"
            ax.set_title(title, fontsize=10, fontweight='bold', color='green')
        else:
            title = f"{result['snip_id']}\n✗ FAILED\n"
            title += f"{result.get('error', 'Unknown error')}"
            ax.set_title(title, fontsize=10, fontweight='bold', color='red')

        ax.axis('off')

    # Hide extra subplots
    for idx in range(n_embryos, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved summary grid: {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main processing pipeline."""

    # Troublesome embryos
    embryos = [
        "20251017_combined_A02_e01_t0064",
        "20250512_E06_e01_t0086",
        "20251017_combined_C04_e01_t0114",
        "20251017_combined_F11_e01_t0065",
    ]

    # Output directory
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*80)
    print("TROUBLESOME EMBRYOS - PHASE 1: REPRODUCE CURRENT BEHAVIOR")
    print("="*80)
    print(f"Processing {len(embryos)} embryos with current default pipeline")
    print(f"Output directory: {output_dir}")
    print("="*80)

    # Process each embryo
    results = []
    for snip_id in embryos:
        result = process_embryo_with_stages(snip_id)
        results.append(result)

        # Generate individual visualization
        safe_filename = snip_id.replace("/", "_").replace("\\", "_")
        viz_path = output_dir / f"{safe_filename}_visualization.png"
        plot_individual_embryo(result, viz_path)

    # Generate summary grid
    summary_path = output_dir / "summary_grid.png"
    plot_summary_grid(results, summary_path)

    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'snip_id': r['snip_id'],
            'success': r['success'],
            'error': r.get('error', None),
            'n_centerline_points': r.get('n_centerline_points', 0),
            'total_length_um': r.get('total_length_um', np.nan),
            'mean_curvature_per_um': r.get('mean_curvature_per_um', np.nan),
            'std_curvature_per_um': r.get('std_curvature_per_um', np.nan),
            'max_curvature_per_um': r.get('max_curvature_per_um', np.nan),
            'load_time_ms': r.get('load_time_ms', np.nan),
            'cleaning_time_ms': r.get('cleaning_time_ms', np.nan),
            'preprocess_time_ms': r.get('preprocess_time_ms', np.nan),
            'extraction_time_ms': r.get('extraction_time_ms', np.nan),
            'total_time_ms': r.get('total_time_ms', np.nan),
        }
        for r in results
    ])

    csv_path = output_dir / "results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results: {csv_path}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    n_success = sum(1 for r in results if r['success'])
    n_failed = len(results) - n_success

    print(f"\n✓ Successful: {n_success}/{len(results)}")
    print(f"✗ Failed: {n_failed}/{len(results)}")

    if n_failed > 0:
        print("\nFailed embryos:")
        for r in results:
            if not r['success']:
                print(f"  • {r['snip_id']}: {r.get('error', 'Unknown error')}")

    print("\n" + "="*80)
    print("Phase 1 Complete!")
    print("="*80)
    print("\nNext steps:")
    if n_failed == 0:
        print("  All embryos processed successfully with current pipeline.")
        print("  → Investigate what changed since previous failures")
    else:
        print("  Some embryos still failing with current pipeline.")
        print("  → Proceed to Phase 2: Test alternative preprocessing methods")

    print("="*80)


if __name__ == "__main__":
    main()
