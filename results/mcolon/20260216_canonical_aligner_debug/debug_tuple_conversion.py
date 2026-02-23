import sys
from pathlib import Path
MORPHSEQ_ROOT = Path(__file__).resolve().parents[3]
assert (MORPHSEQ_ROOT / "src").is_dir(), f"Expected repo root at {MORPHSEQ_ROOT}"
sys.path.insert(0, str(MORPHSEQ_ROOT))

import pandas as pd
import numpy as np

# Load data
data_csv = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
df = pd.read_csv(data_csv, low_memory=False)

# Get a test embryo
test_embryo = "20251205_B08_e01"
test_frame = 70

# Method 1: Direct from dataframe (like debug script)
row_direct = df[(df["embryo_id"] == test_embryo) & (df["frame_index"] == test_frame)].iloc[0]

# Method 2: Through itertuples (like s01b)
row_tuple = df[(df["embryo_id"] == test_embryo) & (df["frame_index"] == test_frame)].itertuples(index=False)
row_tuple = list(row_tuple)[0]
row_dict = row_tuple._asdict()
row_from_tuple = pd.Series(row_dict)

print("="*80)
print("COMPARING DATA PASSING METHODS")
print("="*80)

# Check critical fields
fields = ["embryo_id", "frame_index", "genotype", "predicted_stage_hpf", 
          "mask_rle", "mask_height_px", "mask_width_px"]

print("\nDirect from DataFrame (WORKING):")
for field in fields:
    val = row_direct.get(field, "MISSING")
    if field == "mask_rle":
        print(f"  {field}: {type(val).__name__}, len={len(val) if isinstance(val, str) else 'N/A'}")
    else:
        print(f"  {field}: {val}")

print("\nThrough itertuples (s01b method):")
for field in fields:
    val = row_from_tuple.get(field, "MISSING")
    if field == "mask_rle":
        print(f"  {field}: {type(val).__name__}, len={len(val) if isinstance(val, str) else 'N/A'}")
    else:
        print(f"  {field}: {val}")

print("\n" + "="*80)
print("DIFFERENCES:")
print("="*80)

for field in fields:
    val_direct = row_direct.get(field, None)
    val_tuple = row_from_tuple.get(field, None)
    
    if val_direct != val_tuple:
        print(f"\n{field}:")
        print(f"  Direct: {val_direct}")
        print(f"  Tuple:  {val_tuple}")
        print(f"  Types:  {type(val_direct)} vs {type(val_tuple)}")

print("\n" + "="*80)
