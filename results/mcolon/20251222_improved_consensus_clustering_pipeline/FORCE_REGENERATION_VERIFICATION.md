# Force Regeneration Verification

**Date:** December 24, 2025  
**Purpose:** Confirm that `--force` commands will actually regenerate stale SAM2 and Build03 files

---

## The Problem: Mixed-Age Files

20251125 shows the pattern that necessitates regeneration:

```
Stitched FF:   Dec 12  (regenerated with well mapping fix)
SAM2 masks:    Dec 4   (oldest) → Dec 14 (newest) = 10-day span
BF snips:      Dec 4   (oldest) → Dec 15 (newest) = 11-day span
```

**Issue:** Some SAM2 masks from Dec 4 predate the Dec 12 stitched images (stale by 8 days).

This affects **12 experiments total**:
- **11 need SAM2 regeneration**
- **9 need Build03 regeneration** (subset of those with SAM2 issues)

---

## Command Verification

### SAM2 Regeneration Command

```bash
python -m src.run_morphseq_pipeline.cli pipeline \
  --data-root morphseq_playground \
  --experiments 20250305,20250501,20251017_part1,20251017_part2,20251020,20251104,20251106,20251113,20251119,20251121,20251125 \
  --action sam2 --force
```

### What `--force` Does for SAM2

**CLI Code** (`cli.py` lines 1012-1030):
```python
elif args.action == "sam2":
    for exp in selected:
        if args.force or exp.needs_sam2:
            run_kwargs = dict(
                workers=args.sam2_workers,
                confidence_threshold=args.sam2_confidence,
                iou_threshold=args.sam2_iou,
            )
            if args.force:
                # Force re-detection, mask export, and ensure built metadata is present
                run_kwargs.update(
                    force_detection=True,
                    ensure_built_metadata=True,
                    force_metadata_overwrite=True,
                    force_mask_export=True,
                )
            exp.run_sam2(**run_kwargs)
```

**SAM2 Runner** (`run_sam2.py` lines 308-325):

When `force_detection=True`:
1. **Deletes existing per-experiment annotations:**
   ```python
   if force_detection and annotations_path.exists():
       annotations_path.unlink()  # gdino_detections_{exp}.json
   ```

2. **Deletes existing per-experiment segmentations:**
   ```python
   if force_detection and sam2_output_path.exists():
       sam2_output_path.unlink()  # grounded_sam_segmentations_{exp}.json
   ```

When `force_mask_export=True`:
3. **Overwrites exported masks:**
   ```python
   if force_mask_export:
       export_args.append("--overwrite")  # 06_export_masks.py
   ```

**Result:** ✅ **Complete regeneration from fresh stitched images**
- Old detection files deleted before re-detection
- Old segmentation files deleted before re-segmentation  
- Exported masks overwritten
- CSV metadata regenerated from new masks

---

### Build03 Regeneration Command

```bash
python -m src.run_morphseq_pipeline.cli pipeline \
  --data-root morphseq_playground \
  --experiments 20250501,20250912,20251020,20251104,20251106,20251113,20251119,20251121,20251125 \
  --action build03 --force
```

### What `--force` Does for Build03

**CLI Code** (`cli.py` lines 1032-1044):
```python
elif args.action == "build03":
    for exp in selected:
        if args.force or exp.needs_build03:
            exp.run_build03(
                by_embryo=args.by_embryo,
                frames_per_embryo=args.frames_per_embryo,
                overwrite=args.force  # ← passes force flag
            )
```

**Build03 Runner** (`run_build03.py` lines 172-176):
```python
def run_build03(
    root: str | Path,
    exp: str,
    ...
    overwrite: bool = False,
) -> "object":
    ...
    out_csv = out_dir / f"expr_embryo_metadata_{exp}.csv"
    
    if out_csv.exists() and not overwrite and verbose:
        print(f"ℹ️  Build03 output exists (overwrite=False): {out_csv}")
        return  # ← EXITS WITHOUT REGENERATING
```

**When `overwrite=True`** (from `--force`):
```python
return run_build03_pipeline(
    experiment_name=exp,
    sam2_csv_path=str(sam2_csv_path),
    output_file_path=str(out_csv),
    ...
    snip_overwrite=overwrite,  # ← Regenerates snips too
)
```

**Build03 Pipeline** (`run_build03.py` lines 94-137):

1. **Overwrites per-experiment CSV:**
   ```python
   stats_df.to_csv(output_file_path, index=False)  # No existence check when overwrite=True
   ```

2. **Regenerates BF snips:**
   ```python
   if export_snips:
       extract_embryo_snips(
           root=Path(root_dir),
           stats_df=stats_df,
           outscale=snip_outscale,
           dl_rad_um=snip_dl_rad_um,
           overwrite_flag=snip_overwrite,  # ← True when --force
           n_workers=snip_workers,
       )
   ```

**Result:** ✅ **Complete regeneration from fresh SAM2 CSV**
- Per-experiment CSV overwritten (not skipped)
- BF snips regenerated (overwrite_flag=True)
- All processing based on newest SAM2 masks

---

## Affected Experiments

### SAM2 Regeneration (11 experiments)

All have mixed-age SAM2 masks or masks predating stitched images:

| Experiment | Stitched | SAM2 Oldest | Gap (days) | Issue |
|------------|----------|-------------|------------|-------|
| 20250305 | Nov 6 | Nov 4 | 2 | Masks predate stitched |
| 20250501 | Nov 13 | Oct 31 | 13 | Masks predate stitched |
| 20251017_part1 | Nov 6 | Nov 4 | 2 | Masks predate stitched |
| 20251017_part2 | Nov 6 | Nov 4 | 2 | Masks predate stitched |
| 20251020 | Nov 6 | Nov 4 | 2 | Masks predate stitched |
| 20251104 | Dec 3 | Dec 1 | 2 | Masks predate stitched |
| 20251106 | Dec 4 | Nov 13 | 21 | Masks predate stitched |
| 20251113 | Dec 3 | Dec 1 | 2 | Masks predate stitched |
| 20251119 | Dec 3 | Dec 1 | 2 | Masks predate stitched |
| 20251121 | Dec 3 | Dec 1 | 2 | Masks predate stitched |
| 20251125 | Dec 12 | Dec 4 | 8 | Masks predate stitched |

### Build03 Regeneration (9 experiments)

Subset with BF snips also affected (predating SAM2 masks):

| Experiment | SAM2 CSV | BF Snips Oldest | Gap (days) | Issue |
|------------|----------|-----------------|------------|-------|
| 20250501 | Dec 14 | Nov 1 | 44 | Snips predate SAM2 |
| 20250912 | Dec 14 | Nov 1 | 44 | Snips predate SAM2 |
| 20251020 | Dec 14 | Dec 1 | 13 | Snips predate SAM2 |
| 20251104 | Dec 14 | Dec 1 | 13 | Snips predate SAM2 |
| 20251106 | Dec 14 | Dec 1 | 13 | Snips predate SAM2 |
| 20251113 | Dec 14 | Dec 1 | 13 | Snips predate SAM2 |
| 20251119 | Dec 14 | Dec 1 | 13 | Snips predate SAM2 |
| 20251121 | Dec 14 | Dec 1 | 13 | Snips predate SAM2 |
| 20251125 | Dec 14 | Dec 4 | 10 | Snips predate SAM2 |

**Note:** 20250305, 20251017_part1, 20251017_part2 only need SAM2 regeneration (snips already fresh).

---

## Verification Steps

### Before Running Commands

1. **Check oldest file timestamps:**
   ```bash
   # For SAM2 masks
   find morphseq_playground/sam2_pipeline_files/exported_masks/20251125/masks/ -type f -exec stat -c '%Y %n' {} + | sort -n | head -1
   
   # For BF snips
   find morphseq_playground/training_data/bf_embryo_snips/20251125/ -type f -exec stat -c '%Y %n' {} + | sort -n | head -1
   ```

2. **Run detection script:**
   ```bash
   python detect_mixed_staleness.py
   ```

### After Running Commands

1. **Verify SAM2 regeneration:**
   ```bash
   # All files should be dated after command execution
   find morphseq_playground/sam2_pipeline_files/exported_masks/20251125/masks/ -type f -mtime -1
   # Should show ALL files (nothing older than 1 day)
   ```

2. **Verify Build03 regeneration:**
   ```bash
   # All snips should be dated after command execution
   find morphseq_playground/training_data/bf_embryo_snips/20251125/ -type f -mtime -1
   # Should show ALL files (nothing older than 1 day)
   ```

3. **Re-run detection script:**
   ```bash
   python detect_mixed_staleness.py
   # Should report 0 stale experiments
   ```

---

## Expected Outcomes

### SAM2 Regeneration (11 experiments)

**Before:**
- 11 experiments flagged with stale SAM2 masks
- Age spans: 2-21 days between oldest and newest masks
- Oldest masks predate stitched images by 2-21 days

**After:**
- All SAM2 masks dated within minutes of command execution
- Age span reduced to 0 (all masks generated in single batch)
- All masks based on current stitched images (Dec 12 for 20251125)

### Build03 Regeneration (9 experiments)

**Before:**
- 9 experiments flagged with stale BF snips
- Age spans: 10-44 days between oldest and newest snips
- Oldest snips predate SAM2 CSV by 10-44 days

**After:**
- All BF snips dated within minutes of command execution
- Age span reduced to 0 (all snips extracted in single batch)
- All snips based on current SAM2 masks (Dec 14 dates)

---

## Code Flow Summary

### SAM2 with --force

```
CLI --force flag
  ↓
force_detection=True + force_mask_export=True
  ↓
run_sam2.py:
  1. Delete gdino_detections_{exp}.json
  2. Delete grounded_sam_segmentations_{exp}.json
  3. Run 03_gdino_detection.py (fresh detections)
  4. Run 04_sam2_video_processing.py (fresh segmentations)
  5. Run 06_export_masks.py --overwrite (overwrite mask PNGs)
  6. Generate sam2_metadata_{exp}.csv (fresh metadata)
  ↓
All SAM2 outputs regenerated from current stitched images
```

### Build03 with --force

```
CLI --force flag
  ↓
overwrite=True
  ↓
run_build03.py:
  1. Skip "output exists" check
  2. Load fresh sam2_metadata_{exp}.csv
  3. Run segment_wells_sam2_csv()
  4. Run compile_embryo_stats_sam2()
  5. Overwrite expr_embryo_metadata_{exp}.csv
  6. Run extract_embryo_snips() with overwrite_flag=True
  ↓
All Build03 outputs regenerated from current SAM2 masks
```

---

## Conclusion

✅ **Both commands will fully regenerate stale files:**

1. **SAM2 command** deletes intermediate files (detections, segmentations) before regenerating, ensuring no stale data persists. Mask export uses `--overwrite` to replace existing PNGs.

2. **Build03 command** bypasses existence check when `overwrite=True`, forcing complete reprocessing including snip regeneration with `overwrite_flag=True`.

**No manual cleanup needed** - the `--force` flag triggers deletion/overwrite at each stage.

**Safe to execute** - commands are idempotent and experiment-scoped (won't affect other experiments).

---

## Commands Ready to Run

```bash
# Terminal session
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq

# 1. SAM2 regeneration (11 experiments, ~2-3 hours)
python -m src.run_morphseq_pipeline.cli pipeline \
  --data-root morphseq_playground \
  --experiments 20250305,20250501,20251017_part1,20251017_part2,20251020,20251104,20251106,20251113,20251119,20251121,20251125 \
  --action sam2 --force

# 2. Build03 regeneration (9 experiments, ~30-60 minutes)
python -m src.run_morphseq_pipeline.cli pipeline \
  --data-root morphseq_playground \
  --experiments 20250501,20250912,20251020,20251104,20251106,20251113,20251119,20251121,20251125 \
  --action build03 --force

# 3. Verify success
python detect_mixed_staleness.py
# Expected: "No stale data detected!"
```

---

**Last Updated:** December 24, 2025  
**Verified by:** Code inspection of CLI, run_sam2.py, run_build03.py
