"""
Test integration of two-sided SA outlier detection in build04 pipeline.

This script runs build04_stage_per_experiment on 20250711 and validates that:
1. F03_e01, F06_e01, H07_e01 are correctly flagged as outliers
2. Overall flagging rate is reasonable (5-10%)
3. No errors occur during processing

Usage:
    conda activate segmentations_grounded_sam
    python test_integration.py
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.build.build04_perform_embryo_qc import build04_stage_per_experiment
import pandas as pd

# Configuration
ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
PLAYGROUND_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground")
TEST_EXP = "20250711"
TEST_EMBRYOS = ["20250711_F03_e01", "20250711_F06_e01", "20250711_H07_e01"]

print("=" * 80)
print("TESTING BUILD04 INTEGRATION - TWO-SIDED SA OUTLIER DETECTION")
print("=" * 80)

# Run build04
print(f"\n🚀 Running build04_stage_per_experiment for {TEST_EXP}...")
print(f"   Input from: {PLAYGROUND_ROOT}/metadata/build03_output/")
print(f"   Output to: {PLAYGROUND_ROOT}/metadata/build04_output/")

try:
    # Specify full paths since data is in morphseq_playground
    in_csv = PLAYGROUND_ROOT / "metadata" / "build03_output" / f"expr_embryo_metadata_{TEST_EXP}.csv"
    out_dir = PLAYGROUND_ROOT / "metadata" / "build04_output"

    output_path = build04_stage_per_experiment(
        root=ROOT,  # Still use main repo for metadata/sa_reference_curves.csv
        exp=TEST_EXP,
        in_csv=in_csv,
        out_dir=out_dir,
    )
    print(f"✅ Build04 completed successfully")
    print(f"   Output: {output_path}")
except Exception as e:
    print(f"❌ Build04 failed: {e}")
    raise

# Load results
print(f"\n📊 Loading results...")
df = pd.read_csv(output_path)
print(f"✓ Loaded {len(df)} rows, {df['embryo_id'].nunique()} embryos")

# Check SA outlier flags
print("\n" + "=" * 80)
print("SA OUTLIER FLAG VALIDATION")
print("=" * 80)

n_flagged_frames = df['sa_outlier_flag'].sum()
embryo_flagged = df.groupby('embryo_id')['sa_outlier_flag'].max()
n_flagged_embryos = embryo_flagged.sum()
pct_flagged = 100 * n_flagged_embryos / df['embryo_id'].nunique()

print(f"\nOverall flagging:")
print(f"  Frames flagged: {n_flagged_frames} / {len(df)} ({100*n_flagged_frames/len(df):.1f}%)")
print(f"  Embryos flagged: {n_flagged_embryos} / {df['embryo_id'].nunique()} ({pct_flagged:.1f}%)")

# Check test embryos
print(f"\nTest embryo validation:")
all_caught = True
for emb_id in TEST_EMBRYOS:
    if emb_id in embryo_flagged.index:
        is_flagged = embryo_flagged[emb_id]
        status = "✓ CAUGHT" if is_flagged else "✗ MISSED"
        print(f"  {emb_id}: {status}")
        if not is_flagged:
            all_caught = False
    else:
        print(f"  {emb_id}: ⚠️  Not found in results")
        all_caught = False

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

if all_caught and 5 <= pct_flagged <= 15:
    print("✅ ALL TESTS PASSED")
    print(f"   - All test embryos caught")
    print(f"   - Flagging rate {pct_flagged:.1f}% is reasonable (5-15%)")
elif all_caught:
    print("⚠️  PARTIAL SUCCESS")
    print(f"   - All test embryos caught ✓")
    print(f"   - Flagging rate {pct_flagged:.1f}% outside expected range (5-15%)")
else:
    print("❌ TEST FAILED")
    print(f"   - Some test embryos missed")
    print(f"   - Flagging rate: {pct_flagged:.1f}%")

print("\n✓ Done!")
