#!/usr/bin/env python3
"""
Test the new difference detection API.

Tests:
1. add_group_column() with manual groups
2. compare_groups()
3. plot_comparison()
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

print("="*80)
print("TESTING NEW DIFFERENCE DETECTION API")
print("="*80)

# ============================================================================
# Test 1: Import the new functions
# ============================================================================
print("\n[1/5] Testing imports...")
try:
    from src.analyze.difference_detection import (
        add_group_column,
        compare_groups,
        plot_comparison,
    )
    from src.analyze.trajectory_analysis import (
        load_phenotype_file,
        save_phenotype_file,
        extract_cluster_embryos,
        get_cluster_summary,
    )
    print("  ✓ All imports successful")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 2: Load phenotype files
# ============================================================================
print("\n[2/5] Testing load_phenotype_file()...")
phenotype_dir = Path('/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251219_b9d2_phenotype_extraction/phenotype_lists')

try:
    ce_ids = load_phenotype_file(phenotype_dir / 'b9d2-CE-phenotype.txt')
    hta_ids = load_phenotype_file(phenotype_dir / 'b9d2-HTA-embryos.txt')
    print(f"  ✓ Loaded CE: {len(ce_ids)} embryos")
    print(f"  ✓ Loaded HTA: {len(hta_ids)} embryos")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 3: Load experiment data
# ============================================================================
print("\n[3/5] Loading experiment data...")
from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe

try:
    df1 = load_experiment_dataframe('20251121', format_version='df03')
    df2 = load_experiment_dataframe('20251125', format_version='df03')

    import pandas as pd
    df_raw = pd.concat([df1, df2], ignore_index=True)

    # Get wildtype IDs
    wt_mask = df_raw['genotype'] == 'b9d2_wildtype'
    wt_ids = df_raw[wt_mask]['embryo_id'].unique().tolist()
    # Exclude any phenotype embryos
    all_pheno = set(ce_ids + hta_ids)
    wt_ids = [e for e in wt_ids if e not in all_pheno]

    print(f"  ✓ Loaded {len(df_raw)} rows, {df_raw['embryo_id'].nunique()} embryos")
    print(f"  ✓ Wildtype: {len(wt_ids)} embryos")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 4: add_group_column() - Manual mode
# ============================================================================
print("\n[4/5] Testing add_group_column() - Manual mode...")
try:
    # Test with small subset
    df_test = add_group_column(
        df_raw,
        groups={'CE': ce_ids[:5], 'WT': wt_ids[:5]},
        column_name='test_group'
    )

    # Check column exists
    assert 'test_group' in df_test.columns, "Column not added"

    # Check values
    n_ce = (df_test['test_group'] == 'CE').sum()
    n_wt = (df_test['test_group'] == 'WT').sum()
    print(f"  ✓ Added 'test_group' column")
    print(f"  ✓ CE rows: {n_ce}, WT rows: {n_wt}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 5: compare_groups() - Run comparison
# ============================================================================
print("\n[5/5] Testing compare_groups()...")
try:
    # Add groups for full comparison
    df_comparison = add_group_column(
        df_raw,
        groups={'CE': ce_ids, 'WT': wt_ids},
        column_name='phenotype'
    )

    # Run comparison with few permutations for speed
    results = compare_groups(
        df_comparison,
        group_col='phenotype',
        group1='CE',
        group2='WT',
        n_permutations=10,  # Fast for testing
        verbose=True
    )

    # Check results structure
    assert 'classification' in results, "Missing 'classification'"
    assert 'divergence' in results, "Missing 'divergence'"
    assert 'summary' in results, "Missing 'summary'"
    assert 'config' in results, "Missing 'config'"

    print(f"\n  ✓ Results structure correct")
    print(f"  ✓ Classification: {len(results['classification'])} time bins")
    print(f"  ✓ Divergence: {len(results['divergence']) if results['divergence'] is not None else 0} timepoints")
    print(f"  ✓ Summary: {results['summary']}")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
print("\nNew API functions are working:")
print("  - add_group_column()")
print("  - compare_groups()")
print("  - load_phenotype_file()")
print("\nYou can now use these in notebooks like:")
print("""
>>> from src.analyze.difference_detection import add_group_column, compare_groups, plot_comparison
>>> from src.analyze.trajectory_analysis import load_phenotype_file
>>>
>>> ce_ids = load_phenotype_file('phenotype_lists/ce_embryos.txt')
>>> df = add_group_column(df_raw, groups={'CE': ce_ids, 'WT': wt_ids})
>>> results = compare_groups(df, group_col='group', group1='CE', group2='WT')
>>> fig = plot_comparison(df, results)
""")
