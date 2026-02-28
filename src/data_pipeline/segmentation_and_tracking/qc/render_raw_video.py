from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class RawVideoConfig:
    fps: int = 10
    codec: str = "mp4v"
    write_video: bool = True
    write_frames: bool = False


def _draw_top_banner(frame: np.ndarray, text: str) -> None:
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
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_raw_video(
    *,
    frames: list[tuple[str, Path]],
    out_dir: Path,
    out_name: str,
    cfg: RawVideoConfig,
) -> dict[str, Path]:
    """
    Render a raw (no masks) QC video from a sequence of frames.

    `frames`: list of (image_id, abs_path) in chronological order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not frames:
        return {"out_dir": out_dir}

    frames_dir = out_dir / "frames"
    if cfg.write_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    video_path = out_dir / out_name
    if cfg.write_video:
        # We'll initialize writer after reading the first frame.
        pass

    for image_id, src in frames:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read frame for raw video: {src}")

        frame = img.copy()
        _draw_top_banner(frame, str(image_id))

        if cfg.write_frames:
            cv2.imwrite(str(frames_dir / f"{image_id}.jpg"), frame)

        if cfg.write_video:
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*str(cfg.codec))
                writer = cv2.VideoWriter(str(video_path), fourcc, float(cfg.fps), (w, h))
            writer.write(frame)

    if writer is not None:
        writer.release()

    out = {"out_dir": out_dir}
    if cfg.write_frames:
        out["frames_dir"] = frames_dir
    if cfg.write_video:
        out["video_path"] = video_path
    return out

