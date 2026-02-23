"""
Test script for plot_multimetric_trajectories() function.

This script tests the new multi-metric plotting functionality with synthetic data.
"""

import sys
from pathlib import Path

# Add project root to path so we can import src module
project_root = Path(__file__).resolve().parents[3]  # Go up 3 levels to morphseq/
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

# Test import
try:
    from src.analyze.trajectory_analysis import plot_multimetric_trajectories
    print("✓ Successfully imported plot_multimetric_trajectories")
    print(f"  (Project root: {project_root})")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    print(f"  Project root attempted: {project_root}")
    print(f"  sys.path: {sys.path[:3]}")
    exit(1)

# Create synthetic test data
np.random.seed(42)

n_embryos_per_cluster = 15
n_timepoints = 50
time_grid = np.linspace(18, 52, n_timepoints)

data_rows = []

for cluster_id in [0, 1, 2]:
    for embryo_idx in range(n_embryos_per_cluster):
        embryo_id = f"cluster{cluster_id}_embryo{embryo_idx}"

        # Generate different patterns per cluster
        if cluster_id == 0:
            # Cluster 0: High curvature, normal length
            curvature = 2.0 + 0.5 * np.sin(time_grid / 10) + np.random.normal(0, 0.2, n_timepoints)
            length = 300 + 5 * time_grid + np.random.normal(0, 10, n_timepoints)
        elif cluster_id == 1:
            # Cluster 1: Low curvature, short length
            curvature = 0.5 + 0.2 * np.sin(time_grid / 10) + np.random.normal(0, 0.2, n_timepoints)
            length = 250 + 3 * time_grid + np.random.normal(0, 10, n_timepoints)
        else:
            # Cluster 2: High curvature, short length (CE phenotype)
            curvature = 2.5 + 0.8 * np.sin(time_grid / 10) + np.random.normal(0, 0.2, n_timepoints)
            length = 200 + 2 * time_grid + np.random.normal(0, 10, n_timepoints)

        for t_idx, t in enumerate(time_grid):
            data_rows.append({
                'embryo_id': embryo_id,
                'predicted_stage_hpf': t,
                'baseline_deviation_normalized': curvature[t_idx],
                'total_length_um': length[t_idx],
                'cluster': cluster_id,
                'genotype': f'test_genotype_{cluster_id % 2}',  # Alternate genotypes
                'pair': f'test_pair_{cluster_id}'
            })

df = pd.DataFrame(data_rows)

print(f"✓ Created test DataFrame with {len(df)} rows")
print(f"  - {df['embryo_id'].nunique()} embryos")
print(f"  - {df['cluster'].nunique()} clusters")
print(f"  - {df['predicted_stage_hpf'].nunique()} timepoints")

# Create output directory
output_dir = Path('results/mcolon/20251218_MD-DTW-morphseq_analysis/test_multimetric_plotting_outputs')
output_dir.mkdir(exist_ok=True)
print(f"✓ Created output directory: {output_dir}")

# Test 1: Basic functionality with color by cluster
print("\nTest 1: Color by cluster...")
try:
    fig1 = plot_multimetric_trajectories(
        df,
        metrics=['baseline_deviation_normalized', 'total_length_um'],
        col_by='cluster',
        color_by='cluster',
        metric_labels={
            'baseline_deviation_normalized': 'Curvature (Z-score)',
            'total_length_um': 'Length (μm)'
        },
        backend='plotly',
        output_path=output_dir / 'test1_color_by_cluster.html',
        title='Test 1: Multi-Metric Plot (Color by Cluster)'
    )
    print("✓ Test 1 passed - Plotly figure created and saved")
except Exception as e:
    print(f"✗ Test 1 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Color by genotype
print("\nTest 2: Color by genotype...")
try:
    fig2 = plot_multimetric_trajectories(
        df,
        metrics=['baseline_deviation_normalized', 'total_length_um'],
        col_by='cluster',
        color_by='genotype',
        metric_labels={
            'baseline_deviation_normalized': 'Curvature',
            'total_length_um': 'Length'
        },
        backend='plotly',
        output_path=output_dir / 'test2_color_by_genotype.html',
        title='Test 2: Multi-Metric Plot (Color by Genotype)'
    )
    print("✓ Test 2 passed - Colored by genotype")
except Exception as e:
    print(f"✗ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Single metric (edge case)
print("\nTest 3: Single metric...")
try:
    fig3 = plot_multimetric_trajectories(
        df,
        metrics=['baseline_deviation_normalized'],
        col_by='cluster',
        color_by='cluster',
        backend='plotly',
        output_path=output_dir / 'test3_single_metric.html',
        title='Test 3: Single Metric'
    )
    print("✓ Test 3 passed - Single metric works")
except Exception as e:
    print(f"✗ Test 3 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Matplotlib backend
print("\nTest 4: Matplotlib backend...")
try:
    fig4 = plot_multimetric_trajectories(
        df,
        metrics=['baseline_deviation_normalized', 'total_length_um'],
        col_by='cluster',
        color_by='cluster',
        metric_labels={
            'baseline_deviation_normalized': 'Curvature',
            'total_length_um': 'Length'
        },
        backend='matplotlib',
        output_path=output_dir / 'test4_matplotlib.png',
        title='Test 4: Matplotlib Backend'
    )
    print("✓ Test 4 passed - Matplotlib figure created and saved")
except Exception as e:
    print(f"✗ Test 4 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Both backends
print("\nTest 5: Both backends...")
try:
    result = plot_multimetric_trajectories(
        df,
        metrics=['baseline_deviation_normalized', 'total_length_um'],
        col_by='cluster',
        color_by='pair',
        metric_labels={
            'baseline_deviation_normalized': 'Curvature',
            'total_length_um': 'Length'
        },
        backend='both',
        output_path=output_dir / 'test5_both',
        title='Test 5: Both Backends'
    )
    print("✓ Test 5 passed - Both backends work")
    print(f"  - Plotly figure: {type(result['plotly'])}")
    print(f"  - Matplotlib figure: {type(result['matplotlib'])}")
except Exception as e:
    print(f"✗ Test 5 failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All tests completed!")
print(f"Output files saved to: {output_dir.absolute()}")
print("="*60)
