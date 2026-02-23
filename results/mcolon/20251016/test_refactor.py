#!/usr/bin/env python3
"""
Test the refactored imports and basic functionality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Config
        import config_new as config
        print("✓ config_new imported")
        
        # Utils
        from utils.data_loading import load_experiments
        from utils.binning import bin_by_embryo_time
        print("✓ utils imported")
        
        # Difference detection
        from difference_detection import (
            run_classification_test,
            run_distribution_test,
            predictive_signal_test,
            compute_embryo_penetrance
        )
        print("✓ difference_detection imported")
        
        # Visualization
        from visualization import (
            plot_auroc_with_significance,
            plot_signed_margin_trajectories,
            plot_penetrance_distribution
        )
        print("✓ visualization imported")
        
        print("\n" + "="*60)
        print("All imports successful! ✓")
        print("="*60)
        
        # Print config info
        print("\nConfiguration:")
        print(f"  Results dir: {config.RESULTS_DIR}")
        print(f"  CEP290 experiments: {len(config.CEP290_EXPERIMENTS)} experiments")
        print(f"  Genotype groups: {list(config.GENOTYPE_GROUPS.keys())}")
        
        # Test function signatures
        print("\nTesting function signatures...")
        import inspect
        
        sig = inspect.signature(run_classification_test)
        print(f"\nrun_classification_test parameters:")
        for param_name, param in sig.parameters.items():
            default = param.default if param.default != inspect.Parameter.empty else "required"
            print(f"  {param_name}: {default}")
        
        print("\n" + "="*60)
        print("Structure test complete! Ready to run analysis.")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
