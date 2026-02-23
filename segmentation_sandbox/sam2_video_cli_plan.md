# SAM2 Video CLI Plan

This plan outlines the approach for adding a command-line utility to generate
quality-control videos from SAM2 segmentation outputs.

## Goals
- Quickly create MP4 videos visualizing SAM2 masks for one or more embryos.
- Accept both **embryo IDs** and **video IDs** and resolve them to the
  corresponding video paths using existing parsing utilities.
- Provide a clean **CLI** and an importable function so the tool can be used in
  scripts or interactive notebooks.
- Offer optional overlays:
  - Segmentation masks (enabled by default)
  - Bounding boxes
  - Embryo labels with ID and simple metrics
- Support output directory selection and optional filename suffixes to help
  distinguish different video batches.

## Implementation Notes
- Reuse `VideoGenerator` and `OverlayManager` from
  `scripts/utils/video_generation` for overlay logic and consistent
  styling. These utilities rely on `VideoConfig` defaults for fonts, colors,
  and transparency (e.g., `FONT_SCALE=2`, semi-transparent mask overlays).
- Use functions in `parsing_utils` to normalize input IDs and extract
  `experiment_id` / `video_id` pairs.
- Collect unique `(experiment_id, video_id)` pairs from the provided IDs and
  generate one video per pair.
- Provide flags to toggle overlays:
  - `--bbox` to draw bounding boxes
  - `--no-mask` to hide segmentation masks
  - `--no-labels` to hide embryo labels and metrics
- Add options for
  - `--results-json` path (default: `data/segmentation/grounded_sam_segmentations.json`)
  - `--output-dir` for saving videos (defaults to current working directory)
  - `--suffix` to append before `.mp4` in output filenames

## Testing
- Run `pre-commit` on the new files.
- Execute `python -m py_compile` on the CLI script to ensure it imports
  correctly.
- Run the parsing utilities test (`test_channel_parsing.py`) as a basic
  regression check for ID parsing used by the script.

