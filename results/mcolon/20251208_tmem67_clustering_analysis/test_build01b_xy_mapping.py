"""
Test script to verify build01B XY mapping integration.

This script tests that build01B can successfully use the XY-based mapping
without running the full image processing pipeline (metadata_only=True).
"""

import sys
from pathlib import Path

# Add repo root to path (3 levels up from results/mcolon/20251208_tmem67_clustering_analysis/)
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.build.build01B_compile_yx1_images_torch import build_ff_from_yx1

def test_xy_mapping_integration(exp_name: str):
    """Test XY mapping on a single experiment."""
    print(f"\n{'='*80}")
    print(f"Testing build01B XY mapping integration: {exp_name}")
    print(f"{'='*80}\n")

    try:
        # Run with metadata_only=True to skip image processing
        build_ff_from_yx1(
            data_root="/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground",
            repo_root="/net/trapnell/vol1/home/mdcolon/proj/morphseq",
            exp_name=exp_name,
            metadata_only=True,
            device="cpu"
        )

        print(f"\n{'='*80}")
        print(f"✓ TEST PASSED: {exp_name}")
        print(f"  XY mapping integration working correctly")
        print(f"{'='*80}\n")
        return True

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ TEST FAILED: {exp_name}")
        print(f"  Error: {e}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test on experiments we know work with XY mapping
    test_experiments = [
        "20251112",  # Reference experiment (should pass with 0µm distances)
        "20250711",  # Previously failing, should now work
    ]

    results = {}
    for exp in test_experiments:
        results[exp] = test_xy_mapping_integration(exp)

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    for exp, passed in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {exp}: {status}")
    print(f"{'='*80}\n")

    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)
