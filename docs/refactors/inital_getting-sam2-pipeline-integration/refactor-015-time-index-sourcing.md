# Refactor-015: Time Index Sourcing Alignment

Status: notes in progress

## Motivation

Recent Build03 runs surfaced two related regressions:
- yolk masks fail to load (`Legacy yolk mask not found`) because the snip/yolk stub `t####` no longer lines up with disk filenames.
- `snip_id` values derived downstream no longer match the stitched image IDs or mask filenames, risking downstream QC mismatches.

Both issues originate from diverging definitions of `time_int` after the SAM2 bridge refactor. This document records observations and requirements while we trace the indexing pipeline and plan a fix.

## Current Findings

### Build01 Sources (legacy Keyence & YX1)
- `build01A_compile_keyence_images.py` assigns `time_string = 'T' + time_dir[-4:]` and `time_int = int(time_dir[-4:])` for time-series folders; static snapshots get `time_int = 0`. Disk outputs under `stitched_FF_images` therefore use 1-based `t####` segments mirroring the original folder names.
- `build01B_compile_yx1_images_torch.py` enumerates ND2 frames with `time_int_list = np.arange(0, n_t)` per well. Filenames under `stitched_FF_images` follow the same zero-based order (`{well}_t{t:04}_...`).
- Conclusion: legacy Build01 introduces two conventions—Keyence files start at 1, YX1 files start at 0. The metadata CSV carries both `time_string` and `time_int` so downstream code can track which convention applies for each row.

### SAM2 Bridge (`segment_wells_sam2_csv`)
- The SAM2 CSV already preserves the authoritative `image_id` (`20230525_A03_ch00_t0060`) and columns `time_string`, `time_int` (1-based), and zero-based `frame_index`.
- Current refactor overwrites (`clobbers`) the 1-based `time_int` with `frame_index` (`exp_df['time_int'] = exp_df['frame_index']`). That change silently drops the legacy numbering while keeping `image_id` untouched.
- Downstream `snip_id = embryo_id + '_t' + time_int.zfill(4)` now emits `_t0059` for the example above, breaking lookups for yolk masks and stitched FF images that still live at `_t0060`.

### Mask/Yolk Glob Logic (`export_embryo_snips`)
- Yolk glob uses `stub = f"{well}_t{time_int:04}*"`. When `time_int` is zero-based the glob shifts by one and returns empty for Keyence experiments; YX1 still happens to work because its files were zero-based originally.
- Mask glob relies on `snip_id` (derived from the clobbered `time_int`) to build filenames under `training_data`. Any future integrations that expect `snip_id` to mirror `image_id` will misalign across microscopes.

## Requirements Emerging

1. Preserve the authoritative index from acquisition. SAM2 should propagate `time_string`/`time_int` untouched; if a zero-based value is needed, store it in a separate column (e.g., `frame_index_0`).
2. Disk lookups must prefer `time_string` when available; fallback to `time_int` only if the string is missing. This keeps compatibility with both Keyence (1-based) and YX1 (0-based) exports.
3. `snip_id` must be recomputed using the preserved 1-based value so all downstream artifacts share the same stub as the original `image_id`.
4. Add guards: raise if both `time_string` and `time_int` are absent; log a warning if `time_int == 0` while `time_string` encodes a non-zero frame (signals CSV drift).
5. Ensure SAM2-to-legacy migrations do not invent new numbering. If we need to derive values (e.g., for propagated frames), we should base them on the source `image_id` rather than re-enumerating on load.

## Next Investigation Steps

- Verify GDINO and SAM2 mask exports continue using the original `image_id` for filenames; confirm no additional reindexing happens during mask export.
- Trace Build04/Build06 joins to ensure any cached `time_int` columns still behave once we restore the 1-based values.
- Design regression checks: pick a small Keyence experiment, run the pipeline after the fix, and assert yolk glob hits `/segmentation/yolk*/.../A03_t0060*` when the CSV row reports `time_string = T0060`.

## Open Questions

- Are there experiments where the stitched FF images start at 0 even for Keyence (e.g., partial time series)? Need to sample multiple dates to confirm assumptions before hard-coding validation rules.
- Do SAM2 CSVs ever omit `time_string`? If yes, confirm how to reconstruct it (likely from `image_id` suffix).
- How do legacy Build03 archives behave? Ensure updating `time_int` doesn’t break backward compatibility with previously exported metadata.

