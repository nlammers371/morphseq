from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass(frozen=True)
class RawFramesResult:
    raw_frames_dir: Path
    sam2_frames_dir: Path | None


def _link_or_copy(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown raw_frames.mode={mode!r} (expected 'symlink' or 'copy')")


def materialize_raw_frames(
    *,
    frames: list[tuple[str, Path]],
    out_dir: Path,
    mode: str = "symlink",
    write_sam2_seq_dir: bool = True,
) -> RawFramesResult:
    """
    Persist a well's input frames in a stable location.

    - raw_frames/: files named by image_id (easy to browse)
    - sam2_frames/: optional sequential 00000.jpg, 00001.jpg... for SAM2 predictors

    `frames`: list of (image_id, abs_path) in chronological order.
    """
    out_dir = Path(out_dir)
    raw_frames_dir = out_dir / "raw_frames"
    raw_frames_dir.mkdir(parents=True, exist_ok=True)

    sam2_frames_dir: Path | None = None
    if write_sam2_seq_dir:
        sam2_frames_dir = out_dir / "sam2_frames"
        sam2_frames_dir.mkdir(parents=True, exist_ok=True)

    for seq_idx, (image_id, src_abs) in enumerate(frames):
        src_abs = Path(src_abs)
        if not src_abs.exists():
            raise FileNotFoundError(f"Frame not found: {src_abs}")

        # Preserve extension (usually .jpg)
        ext = src_abs.suffix or ".jpg"
        dst_raw = raw_frames_dir / f"{image_id}{ext}"
        _link_or_copy(src_abs, dst_raw, mode=mode)

        if sam2_frames_dir is not None:
            dst_seq = sam2_frames_dir / f"{seq_idx:05d}{ext}"
            _link_or_copy(src_abs, dst_seq, mode=mode)

    return RawFramesResult(raw_frames_dir=raw_frames_dir, sam2_frames_dir=sam2_frames_dir)

