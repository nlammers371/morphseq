# Phase 3 – Segmentation & UNet Auxiliary Masks

Goal: consume stitched FF images + aligned metadata to produce embryo
detections, SAM2 tracking outputs, mask exports, and UNet auxiliary
quality-control masks. All Stage 3 modules must rely on the manifest and
`identifiers/parsing.py` so downstream phases receive canonical IDs.

---

## Inputs

- `experiment_image_manifest.json` (Phase 2b) – authoritative list of
  wells, channels, frames.
- `built_image_data/{exp}/stitched_ff_images/…` (Phase 2a) – normalized
  stitched imagery.
- Aligned metadata (Phase 1) for calibration (`scope_and_plate_metadata`).

---

## Outputs

- `segmentation/{exp}/gdino_detections.json` – GroundingDINO seed boxes.
- `segmentation/{exp}/sam2_raw_output.json` – SAM2 propagation results
  (nested JSON).
- `segmentation/{exp}/mask_images/{image_id}_masks.png` – exported mask
  PNGs (integer labeled).
- `segmentation/{exp}/segmentation_tracking.csv` – schema-backed tracking
  table (`REQUIRED_COLUMNS_SEGMENTATION_TRACKING`).
- `segmentation/{exp}/unet_masks/{mask_type}/{image_id}_{mask_type}.png`
  – UNet auxiliary masks (via=viability, yolk, focus, bubble, mask).
- Optional diagnostics (propagation stats, seed frame choices) – treated
  as human-facing logs; no strict schema.

---

## Modules to Populate

### `segmentation/grounded_sam2/gdino_detection.py`

- Responsibility: run GroundingDINO per well using manifest frame lists
  to produce seed detections.
- Must record detections in manifest order; derive IDs via
  `identifiers/parsing`.

### `segmentation/grounded_sam2/propagation.py`

- Responsibility: bidirectional SAM2 propagation from seed frames to
  generate per-embryo masks across time.
- Uses `frame_organization` helpers internally (no separate Snakemake
  rule).
- Emits nested JSON and feeds mask export / CSV formatter.
- **Pain point:** Converting GroundingDINO bbox format to SAM2 input
  format requires a utility to manage expected outputs from each model
  and ensure proper input format conversion (coordinate systems, box
  ordering, normalization).


**Minimal functional interface (no classes):**

```python
@contextmanager
def sam2_frame_dir(frame_paths: list[Path]):
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        for idx, src in enumerate(frame_paths):
            (tmp_dir / f"{idx:05d}.jpg").symlink_to(src)
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)

def propagate_forward(predictor, temp_dir: Path, start_index: int,
                      seed_boxes, embryo_ids, verbose=False):
    state = predictor.init_state(video_path=str(temp_dir))
    predictor.add_new_points_or_box(state, frame_idx=0, box=seed_boxes, obj_id=..., ...)
    raw = {}
    for frame_offset, obj_ids, mask_logits in predictor.propagate_in_video(state):
        raw[start_index + frame_offset] = decode_masks(obj_ids, mask_logits, embryo_ids)
    return raw

def propagate_bidirectional(predictor, frame_paths: list[Path], seed_idx: int,
                            seed_boxes, embryo_ids, verbose=False):
    with sam2_frame_dir(frame_paths[seed_idx:]) as tmp_fwd:
        forward = propagate_forward(
            predictor, tmp_fwd,
            start_index=seed_idx,
            seed_boxes=seed_boxes,
            embryo_ids=embryo_ids,
            verbose=verbose,
        )

    if seed_idx == 0:
        return forward

    with sam2_frame_dir(frame_paths[:seed_idx + 1][::-1]) as tmp_rev:
        backward_raw = propagate_forward(
            predictor, tmp_rev,
            start_index=0,
            seed_boxes=seed_boxes,
            embryo_ids=embryo_ids,
            verbose=verbose,
        )

    backward = {seed_idx - offset: data for offset, data in backward_raw.items()}
    return merge_results(forward, backward)  # prefers forward when both exist
```

`propagate_forward` always receives a `start_index`, so offset `0` from SAM2 maps back to the real frame index (the seed frame in the forward pass, the seed frame in reversed space for backward). When we do the reverse slice, we remap offsets with `seed_idx - offset` before merging, keeping everything keyed by true chronology.



### `segmentation/grounded_sam2/mask_export.py`

- Responsibility: convert SAM2 masks to integer-labeled PNGs under
  `mask_images/`.
- Must follow `image_id` naming from parsing helpers.

### `segmentation/grounded_sam2/csv_formatter.py`

- Responsibility: flatten SAM2 JSON + metadata into
  `segmentation_tracking.csv`, enforcing
  `REQUIRED_COLUMNS_SEGMENTATION_TRACKING`.
- Joins aligned metadata to add calibration fields (`well_id`,
  `micrometers_per_pixel`, etc.).

### `segmentation/unet/inference.py` & `model_loader.py`

- Responsibility: generate auxiliary UNet masks using stitched FF inputs.
- Unified inference pipeline with 5 model checkpoints: `mask_v0_0100`,
  `via_v1_0100`, `yolk_v1_0050`, `focus_v0_0100`, `bubble_v0_0100`.
- Outputs to `segmentation/{exp}/unet_masks/{mask_type}/` (via, yolk,
  focus, bubble, mask subdirectories).
- Viability masks (`via/`) feed Phase 6 `embryo_death_qc` (only death source).

---

## Contracts & Validation

- **Manifest-driven:** all modules read frames/wells from the manifest,
  never globbing stitched images directly.
- **ID consistency:** use `identifiers/parsing` to derive `image_id`,
  `snip_id`, `embryo_id`, etc., so Phase 4 reuses the same identifiers.
- **Schema enforcement:** `segmentation_tracking.csv` must call the
  schema validator; auxiliary mask outputs follow directory conventions.
- **Diagnostics:** any propagation statistics or seed-frame logs are
  optional and considered human-readable (no formal schema, similar to
  Phase 2 diagnostics).

---

## Handoff

- Phase 3 outputs feed directly into snip extraction (Phase 4),
  feature/QC modules (Phase 5-6), and the embryo death QC (via UNet
  viability masks).
- Ensure masking outputs and tracking table align with the manifest so
  downstream joins remain deterministic.
