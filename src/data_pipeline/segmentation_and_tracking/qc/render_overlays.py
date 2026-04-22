from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


from data_pipeline.segmentation.video_generation.video_config import COLORBLIND_PALETTE, VideoConfig


def _resolve(root: Path, path_str: str) -> Path:
    p = Path(str(path_str))
    return p if p.is_absolute() else (root / p)

def _color_for_key(key: str) -> tuple[int, int, int]:
    # Deterministic palette selection for stable QC visuals across runs.
    palette = list(COLORBLIND_PALETTE.values())
    if not palette:
        return (0, 255, 0)
    idx = abs(hash(str(key))) % len(palette)
    return palette[idx]


def _draw_label(frame: np.ndarray, text: str, x: int, y: int) -> None:
    # Caller controls scale/thickness; defaults are set below via cfg.
    cv2.putText(
        frame,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

def _draw_label_cfg(frame: np.ndarray, text: str, x: int, y: int, *, scale: float, thickness: int) -> None:
    cv2.putText(
        frame,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        (255, 255, 255),
        int(thickness),
        cv2.LINE_AA,
    )


def _draw_top_banner(frame: np.ndarray, text: str) -> None:
    """Draw a readable top banner label (image_id) for QC videos."""
    if frame is None or frame.size == 0:
        return
    x, y = 16, 36
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 8
    x0, y0 = x - pad, y - th - pad
    x1, y1 = x + tw + pad, y + baseline + pad
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(frame.shape[1] - 1, x1)
    y1 = min(frame.shape[0] - 1, y1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    # alpha blend rectangle for readability
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def _draw_top_banner_cfg(frame: np.ndarray, text: str, *, scale: float, thickness: int) -> None:
    if frame is None or frame.size == 0:
        return
    x, y = 16, 36
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = float(scale)
    thickness = int(thickness)
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 8
    x0, y0 = x - pad, y - th - pad
    x1, y1 = x + tw + pad, y + baseline + pad
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(frame.shape[1] - 1, x1)
    y1 = min(frame.shape[0] - 1, y1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_overlays_from_segmentation_tracking(
    tracking_csv: Path,
    *,
    output_root: Path,
    out_dir: Path,
    out_name: str = "overlay.mp4",
    cfg: VideoConfig,
    mask_type: str | None = None,
    frame_suffix: str | None = None,
) -> dict[str, Path]:
    """
    Render per-frame overlay images and an MP4 video from `segmentation_tracking.csv`.

    Uses `exported_mask_path` PNGs (fast, avoids RLE decode).
    """
    tracking_csv = Path(tracking_csv)
    output_root = Path(output_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tracking_csv)
    if len(df) == 0:
        return {"out_dir": out_dir}

    required = {"image_id", "time_int", "source_image_path", "exported_mask_path", "embryo_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"segmentation_tracking missing columns required for overlays: {sorted(missing)}")

    if mask_type is not None:
        if "mask_type" not in df.columns:
            raise ValueError("mask_type filter requested but segmentation_tracking has no `mask_type` column")
        df = df[df["mask_type"].astype(str) == str(mask_type)].copy()
        if len(df) == 0:
            return {"out_dir": out_dir}

    frames_dir = out_dir / "frames"
    if bool(getattr(cfg, "WRITE_FRAMES", True)):
        frames_dir.mkdir(parents=True, exist_ok=True)

    suffix = frame_suffix
    if suffix is None:
        # Make standalone frame filenames self-describing even if copied out of the directory.
        # Directory already encodes mask head, but the suffix helps avoid confusion.
        suffix = f"_{mask_type}_mask_overlay" if mask_type is not None else ""

    # Ensure deterministic order.
    sort_cols = ["time_int", "image_id"]
    if "embryo_local_id" in df.columns:
        sort_cols.append("embryo_local_id")
    else:
        sort_cols.append("embryo_id")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    out_scale = float(getattr(cfg, "OUTPUT_SCALE", 1.0))
    label_scale = float(getattr(cfg, "LABEL_FONT_SCALE", getattr(cfg, "FONT_SCALE", 0.8)))
    label_thickness = int(getattr(cfg, "LABEL_THICKNESS", getattr(cfg, "FONT_THICKNESS", 2)))
    banner_scale = float(getattr(cfg, "BANNER_FONT_SCALE", 1.0))
    banner_thickness = int(getattr(cfg, "BANNER_THICKNESS", 2))

    writer = None
    video_path = out_dir / str(out_name)

    produced_frames: list[Path] = []

    for image_id, g in df.groupby("image_id", sort=False):
        # Base image
        source_rel = str(g["source_image_path"].iloc[0])
        source_path = _resolve(output_root, source_rel)
        base = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if base is None:
            raise FileNotFoundError(f"Could not read source image for overlay: {source_path}")

        overlay = base.copy()
        h, w = overlay.shape[:2]

        alpha = float(getattr(cfg, "MASK_ALPHA", 0.45))
        for _, row in g.iterrows():
            # Prefer a short tracker-local ID for overlay readability.
            if "embryo_local_id" in row and pd.notna(row["embryo_local_id"]):
                emb = str(row["embryo_local_id"])
            else:
                emb = str(row["embryo_id"])
            mask_rel = str(row["exported_mask_path"])
            mask_path = _resolve(output_root, mask_rel)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask_bin = (mask > 0)
            if mask_bin.shape[:2] != overlay.shape[:2]:
                # Best-effort resize (should not happen if masks are exported correctly).
                mask_bin = cv2.resize(mask_bin.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

            color = np.asarray(_color_for_key(emb), dtype=np.float32)
            overlay[mask_bin] = (
                overlay[mask_bin].astype(np.float32) * (1.0 - alpha)
                + color * alpha
            ).astype(np.uint8)

            # BBox + label (best-effort; may be missing/zero).
            x1 = int(round(float(row.get("bbox_x_min", 0.0))))
            y1 = int(round(float(row.get("bbox_y_min", 0.0))))
            x2 = int(round(float(row.get("bbox_x_max", 0.0))))
            y2 = int(round(float(row.get("bbox_y_max", 0.0))))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), tuple(int(c) for c in color), 2)
                _draw_label_cfg(overlay, emb, x1 + 4, max(20, y1 - 8), scale=label_scale, thickness=label_thickness)

        # Add image_id to the top of the frame so QC videos are easy to follow.
        _draw_top_banner_cfg(overlay, str(image_id), scale=banner_scale, thickness=banner_thickness)

        # Optional output up-scaling for easier viewing.
        if out_scale != 1.0:
            if out_scale <= 0:
                raise ValueError(f"OUTPUT_SCALE must be > 0, got {out_scale}")
            h0, w0 = overlay.shape[:2]
            overlay = cv2.resize(
                overlay,
                (int(round(w0 * out_scale)), int(round(h0 * out_scale))),
                interpolation=cv2.INTER_LINEAR,
            )

        out_frame_path = frames_dir / f"{image_id}{suffix}.jpg"
        if bool(getattr(cfg, "WRITE_FRAMES", True)):
            cv2.imwrite(str(out_frame_path), overlay)
        produced_frames.append(out_frame_path)

        if bool(getattr(cfg, "WRITE_VIDEO", True)):
            if writer is None:
                h, w = overlay.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = float(getattr(cfg, "FPS", 10))
                writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
            writer.write(overlay)

    if writer is not None:
        writer.release()

    out = {"out_dir": out_dir}
    if bool(getattr(cfg, "WRITE_FRAMES", True)):
        out["frames_dir"] = frames_dir
    if bool(getattr(cfg, "WRITE_VIDEO", True)):
        out["video_path"] = video_path
    return out
