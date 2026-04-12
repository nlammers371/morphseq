from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

# Lightweight, import-safe helpers for pipeline state checks.
# These are intentionally dependency-free (no torch/cv2/skimage) so that
# ExperimentManager status/discovery can run quickly on login/compute nodes.


# Patterns to use for file checking (used by Experiment._sync_with_disk)
PATTERNS = {
    "raw": ("W*", "XY*", "*.nd2"),
    "ff": ("*.jpg", "*.png", "ff_*"),
    "stitch": ("*_stitch.jpg",),
    "stitch_z": ("*_stack.tif",),
    "segment": ("*.jpg",),
    "snips": ("*.jpg",),
}


def _match_files(folder: Union[Path, str], patterns: Iterable[str]) -> List[Path]:
    """
    Return a list with *at most one* matching file in `folder`
    (empty list ↔ no match). Stops after the first hit.
    """
    if isinstance(patterns, str):
        patterns = [patterns]

    folder = Path(folder)
    if not folder.is_dir():
        return []

    for pattern in patterns:
        hit = next(folder.glob(pattern), None)
        if hit is not None:
            return [hit]

    return []


def has_output(path: Optional[Path], patterns: Sequence[str]) -> bool:
    """Does *path* contain at least one matching file?"""
    if not path:
        return False
    return bool(_match_files(Path(path), patterns))


def newest_mtime(path: Optional[Path], patterns: Sequence[str]) -> float:
    """mtime of the newest file that matches *patterns*. Returns 0 if nothing matches."""
    if not path:
        return 0.0

    files = _match_files(Path(path), patterns)
    if not files:
        return 0.0

    return max(f.stat().st_mtime for f in files)


def _mod_time(path: Optional[Path]) -> float:
    return path.stat().st_mtime if path and path.exists() else 0.0

