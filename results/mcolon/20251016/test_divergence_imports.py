#!/usr/bin/env python3
"""
Quick import test for divergence analysis module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that divergence analysis imports work."""
    print("Testing divergence_analysis imports...")
    
    try:
        # Import main interface
        from divergence_analysis import (
            compute_divergence_scores,
            summarize_divergence
        )
        print("✓ Main interface imported")
        
        # Import distance functions
        from divergence_analysis.distances import (
            compute_mahalanobis_distance,
            compute_euclidean_distance,
            compute_standardized_distance,
            compute_cosine_distance,
            detect_outliers_mahalanobis
        )
        print("✓ Distance functions imported")
        
        # Import reference functions
        from divergence_analysis.reference import (
            compute_reference_statistics,
            validate_reference_stats,
            get_reference_for_time
        )
        print("✓ Reference functions imported")
        
        print("\n" + "="*60)
        print("All divergence_analysis imports successful! ✓")
        print("="*60)
        
        # Print function signatures
        import inspect
        
        print("\nMain function signature:")
        sig = inspect.signature(compute_divergence_scores)
        print(f"\ncompute_divergence_scores parameters:")
        for param_name, param in sig.parameters.items():
            default = param.default if param.default != inspect.Parameter.empty else "required"
            print(f"  {param_name}: {default}")
        
        print("\n" + "="*60)
        print("Divergence analysis module ready to use!")
        print("="*60)
        
        print("\nKey features:")
        print("  ✓ Flexible reference: use ANY genotype (not just wildtype)")
        print("  ✓ Multiple metrics: Mahalanobis, Euclidean, Standardized, Cosine")
        print("  ✓ Outlier detection: Statistical framework for extreme cases")
        print("  ✓ Time-resolved: Track divergence over development")
        
        print("\nExample usage:")
        print("  # Compare homozygous to wildtype")
        print("  df_div = compute_divergence_scores(")
        print("      df_binned,")
        print("      test_genotypes='cep290_homozygous',")
        print("      reference_genotype='cep290_wildtype'")
        print("  )")
        print("")
        print("  # Compare homozygous to heterozygous (!)") 
        print("  df_div = compute_divergence_scores(")
        print("      df_binned,")
        print("      test_genotypes='cep290_homozygous',")
        print("      reference_genotype='cep290_heterozygous'  # Any genotype!")
        print("  )")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
