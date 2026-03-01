from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_pipeline.segmentation.grounded_sam2.propagation import encode_mask_to_rle

from ..raw_types import RawMask, RawTrack


def ingest_propagation(
    results: dict[int, dict[str, dict]],
    *,
    image_id_by_frame_index: dict[int, str],
    seed_frame_index: int,
    seed_image_id: str,
    well_id: str,
    channel_id: str,
) -> list[RawTrack]:
    tracks: list[RawTrack] = []
    for frame_index, frame_data in results.items():
        image_id = image_id_by_frame_index.get(int(frame_index))
        if image_id is None:
            continue
        for embryo_id, emb in frame_data.items():
            mask = emb["mask"]
            bbox = [float(x) for x in emb.get("bbox", [0, 0, 0, 0])]
            area = float(emb.get("area", 0.0))
            conf = float(emb.get("confidence", 0.0))
            embryo_local_id = str(embryo_id)
            embryo_global_id = f"{str(well_id)}_{embryo_local_id}"
            tracks.append(
                RawTrack(
                    frame_index=int(frame_index),
                    image_id=str(image_id),
                    embryo_id=str(embryo_global_id),
                    embryo_local_id=str(embryo_local_id),
                    channel_id=str(channel_id),
                    mask=mask,
                    bbox_xyxy_abs=bbox,
                    area_px=area,
                    confidence=conf,
                    is_seed_frame=(str(image_id) == str(seed_image_id) and int(frame_index) == int(seed_frame_index)),
                )
            )
    return tracks


def _centroid_xy(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask.astype(bool))
    if len(xs) == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def tracks_to_raw_masks(
    tracks: list[RawTrack],
    *,
    source_image_path_by_image_id: dict[str, str],
    exported_mask_dir: Path,
    exported_mask_rel_prefix: str,
    mask_type: str = "sam",
) -> list[RawMask]:
    exported_mask_dir = Path(exported_mask_dir)
    exported_mask_dir.mkdir(parents=True, exist_ok=True)

    mask_head = f"{mask_type}_mask"
    masks: list[RawMask] = []
    for t in tracks:
        mask_arr = np.asarray(t.mask)
        # SAM2 outputs can be (1,H,W); normalize to (H,W) for centroid + export consistency.
        if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
            mask_arr = mask_arr[0]
        elif mask_arr.ndim == 3 and mask_arr.shape[-1] == 1:
            mask_arr = mask_arr[..., 0]
        else:
            mask_arr = np.squeeze(mask_arr)
        rle = encode_mask_to_rle(mask_arr)
        cx, cy = _centroid_xy(mask_arr)
        bbox = [float(x) for x in t.bbox_xyxy_abs]
        source_image_path = str(source_image_path_by_image_id.get(t.image_id, ""))

        snip_id = f"{str(t.embryo_id)}_{str(t.channel_id)}_f{int(t.frame_index):04d}"
        exported_rel = f"{exported_mask_rel_prefix}/{mask_head}/{snip_id}_mask.png"
        exported_abs = exported_mask_dir / f"{snip_id}_mask.png"

        # Export a simple binary PNG (uint8).
        try:
            import imageio.v2 as imageio  # type: ignore
        except Exception:  # pragma: no cover
            import imageio  # type: ignore
        imageio.imwrite(exported_abs, (mask_arr.astype(np.uint8) * 255))

        masks.append(
            RawMask(
                frame_index=int(t.frame_index),
                image_id=str(t.image_id),
                embryo_id=str(t.embryo_id),
                embryo_local_id=str(t.embryo_local_id),
                channel_id=str(t.channel_id),
                mask_type=str(mask_type),
                mask_rle=rle,
                area_px=float(t.area_px),
                bbox_xyxy_abs=bbox,
                centroid_x_px=float(cx),
                centroid_y_px=float(cy),
                confidence=float(t.confidence),
                is_seed_frame=bool(t.is_seed_frame),
                exported_mask_path=str(exported_rel),
                source_image_path=str(source_image_path),
            )
        )

        # Free memory: the heavy mask array is no longer needed after encoding/export.
        t.mask = None

    return masks
