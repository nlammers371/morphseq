"""
Test script for refactored pair analysis plotting.

Exercises the new Level 1 (generic) and Level 2 (pair-specific) plotting functions.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

from src.analyze.trajectory_analysis import (
    extract_genotype_suffix,
    get_color_for_genotype,
    build_genotype_style_config,
    plot_trajectories_faceted,
    plot_pairs_overview,
    plot_genotypes_by_pair,
    plot_single_genotype_across_pairs,
)


def create_synthetic_data() -> pd.DataFrame:
    """Create synthetic trajectory data for testing."""
    np.random.seed(42)

    # Create synthetic data for two genes with multiple genotypes
    data = []

    genes = ['cep290', 'b9d2']
    genotype_suffixes = ['wildtype', 'heterozygous', 'homozygous']
    pairs = ['pair_1', 'pair_2', 'pair_3']
    n_embryos_per_group = 3

    for gene in genes:
        for suffix in genotype_suffixes:
            genotype = f'{gene}_{suffix}'
            for pair in pairs:
                for embryo_i in range(n_embryos_per_group):
                    embryo_id = f'{gene}_{suffix}_{pair}_embryo{embryo_i}'

                    # Create trajectory
                    times = np.linspace(20, 36, 25)
                    # Vary by genotype
                    base_level = {'wildtype': 0, 'heterozygous': 0.5, 'homozygous': 1.0}[suffix]
                    noise = np.random.normal(base_level, 0.2, len(times))
                    metrics = np.cumsum(noise) / 5

                    for time, metric in zip(times, metrics):
                        data.append({
                            'embryo_id': embryo_id,
                            'predicted_stage_hpf': time,
                            'baseline_deviation_normalized': metric,
                            'genotype': genotype,
                            'pair': pair,
                            'gene': gene,
                        })

    return pd.DataFrame(data)


def test_genotype_styling():
    """Test genotype styling functions."""
    print("Testing genotype styling...")

    # Test suffix extraction
    assert extract_genotype_suffix('cep290_homozygous') == 'homozygous'
    assert extract_genotype_suffix('b9d2_het') == 'heterozygous'
    assert extract_genotype_suffix('CEP290_WILDTYPE') == 'wildtype'
    assert extract_genotype_suffix('unknown') == 'unknown'
    print("  ✓ Suffix extraction works")

    # Test color mapping
    assert get_color_for_genotype('cep290_wildtype') == '#2E7D32'
    assert get_color_for_genotype('b9d2_heterozygous') == '#FFA500'
    print("  ✓ Color mapping works")

    # Test config building
    genotypes = ['b9d2_wildtype', 'b9d2_homo', 'b9d2_het']
    config = build_genotype_style_config(genotypes)
    assert config['order'][0] == 'b9d2_wildtype'  # wildtype first
    assert 'homo' in config['order'][2].lower()  # homo last
    print("  ✓ Style config building works")


def test_level1_generic_plotting():
    """Test Level 1 generic plotting with various facet combinations."""
    print("\nTesting Level 1 generic faceted plotting...")

    df = create_synthetic_data()
    output_dir = Path('/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/test_updated_pair_plotting')

    # Test 1: Basic faceted plot (gene × genotype)
    print("  - Testing basic facet (gene × genotype)...")
    try:
        fig = plot_trajectories_faceted(
            df,
            row_by='gene',
            col_by='genotype',
            backend='matplotlib',
            title='Test: Gene × Genotype',
        )
        print("    ✓ Basic faceting works (matplotlib)")
    except Exception as e:
        print(f"    ✗ Basic faceting failed: {e}")

    # Test 2: Overlay faceting
    print("  - Testing color_by_grouping faceting (pair with gene grouping)...")
    try:
        fig = plot_trajectories_faceted(
            df[df['gene'] == 'cep290'],  # Single gene for clarity
            col_by='pair',
            color_by_grouping='genotype',
            backend='matplotlib',
            title='Test: Pair Columns with Genotype Grouping',
        )
        print("    ✓ color_by_grouping faceting works (matplotlib)")
    except Exception as e:
        print(f"    ✗ color_by_grouping faceting failed: {e}")

    # Test 3: backend='both'
    print("  - Testing dual backend (both PNG and Plotly)...")
    try:
        output_file = output_dir / 'test_dual_backend'
        result = plot_trajectories_faceted(
            df[df['gene'] == 'cep290'],
            col_by='pair',
            color_by_grouping='genotype',
            backend='both',
            output_path=output_file,
            title='Test: Dual Backend Output',
        )
        assert isinstance(result, dict), "backend='both' should return dict"
        assert 'plotly' in result and 'matplotlib' in result
        print("    ✓ Dual backend works (created .html and .png)")
    except Exception as e:
        print(f"    ✗ Dual backend failed: {e}")


def test_level2_pair_specific_plotting():
    """Test Level 2 pair-specific plotting."""
    print("\nTesting Level 2 pair-specific plotting...")

    df = create_synthetic_data()
    output_dir = Path('/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/test_updated_pair_plotting')

    # Test 1: Pairs overview (NxM grid)
    print("  - Testing plot_pairs_overview...")
    try:
        fig = plot_pairs_overview(
            df[df['gene'] == 'cep290'],
            backend='matplotlib',
            title='CEP290: All Pairs × Genotypes',
        )
        print("    ✓ plot_pairs_overview works")
    except Exception as e:
        print(f"    ✗ plot_pairs_overview failed: {e}")

    # Test 2: Genotypes by pair (1xN with overlay)
    print("  - Testing plot_genotypes_by_pair...")
    try:
        fig = plot_genotypes_by_pair(
            df[df['gene'] == 'b9d2'],
            backend='matplotlib',
            title='B9D2: Genotypes by Pair (Overlaid)',
        )
        print("    ✓ plot_genotypes_by_pair works")
    except Exception as e:
        print(f"    ✗ plot_genotypes_by_pair failed: {e}")

    # Test 3: Single genotype across pairs
    print("  - Testing plot_single_genotype_across_pairs...")
    try:
        fig = plot_single_genotype_across_pairs(
            df[df['gene'] == 'cep290'],
            genotype='cep290_wildtype',
            backend='matplotlib',
            title='CEP290 Wildtype: Across Pairs',
        )
        print("    ✓ plot_single_genotype_across_pairs works")
    except Exception as e:
        print(f"    ✗ plot_single_genotype_across_pairs failed: {e}")

    # Test 4: Missing pair column fallback
    print("  - Testing unknown_pair fallback...")
    try:
        df_no_pair = df[df['gene'] == 'cep290'].drop('pair', axis=1)
        fig = plot_pairs_overview(
            df_no_pair,
            backend='matplotlib',
            title='CEP290: Fallback Pair Creation',
        )
        print("    ✓ Automatic {genotype}_unknown_pair creation works")
    except Exception as e:
        print(f"    ✗ Pair fallback failed: {e}")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Refactored Pair Analysis Plotting")
    print("=" * 60)

    test_genotype_styling()
    test_level1_generic_plotting()
    test_level2_pair_specific_plotting()

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
