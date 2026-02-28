"""SAM2 loader wrapper.

We do not assume `sam2` is pip-installed. Instead, we support loading from a
checked-out models root directory that contains a `sam2/` package directory.

SAM2 import is a little sensitive to working directory in some environments, so we
mirror the known-good approach: add models root to sys.path and temporarily chdir
into the `sam2` package directory before importing/building the predictor.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_sam2_pkg_dir(sam2_models_root: Path) -> Path:
    sam2_models_root = Path(sam2_models_root)
    if not sam2_models_root.exists():
        raise FileNotFoundError(f"SAM2 models root not found: {sam2_models_root}")

    # Typical layout: <sam2_models_root>/sam2/<python package>
    candidate = sam2_models_root / "sam2"
    if candidate.is_dir():
        return candidate

    # Alternate: the configured root might already be the package dir.
    if (sam2_models_root / "build_sam.py").exists():
        return sam2_models_root

    raise FileNotFoundError(
        "Could not locate SAM2 package directory. "
        f"Expected either {candidate} or a package-like directory at {sam2_models_root}."
    )


def load_sam2_video_predictor(
    *,
    sam2_models_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    device: str = "cuda",
):
    sam2_models_root = Path(sam2_models_root)
    sam2_pkg_dir = _resolve_sam2_pkg_dir(sam2_models_root)
    sam2_models_root = sam2_models_root.resolve()
    sam2_pkg_dir = sam2_pkg_dir.resolve()

    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)

    # Resolve relative paths:
    # - config is typically under sam2_pkg_dir (e.g. sam2/configs/*.yaml)
    # - checkpoint is typically under sam2_models_root (e.g. checkpoints/*.pt)
    if not config_path.is_absolute():
        config_path = sam2_pkg_dir / config_path
    if not checkpoint_path.is_absolute():
        checkpoint_path = sam2_models_root / checkpoint_path

    if not config_path.exists():
        raise FileNotFoundError(f"SAM2 config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")

    # Make `import sam2...` work:
    # - if pkg dir is <root>/sam2 then sys.path should include <root>
    # - if pkg dir is <root> itself, sys.path should include parent(<root>)
    if sam2_pkg_dir.name == "sam2":
        sys_path_root = sam2_pkg_dir.parent
    else:
        sys_path_root = sam2_pkg_dir.parent

    root_str = str(sys_path_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    original_cwd = os.getcwd()
    try:
        os.chdir(str(sam2_pkg_dir))
        try:
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Failed to import SAM2. "
                f"sam2_models_root={sam2_models_root}, sam2_pkg_dir={sam2_pkg_dir}."
            ) from e
        # Hydra expects config_name relative to the sam2 package config search path.
        # Pass a relative path like "configs/sam2/sam2_hiera_l.yaml" (NOT an absolute path).
        cfg_name = str(config_path.relative_to(sam2_pkg_dir))
        return build_sam2_video_predictor(cfg_name, str(checkpoint_path), device=device)
    finally:
        os.chdir(original_cwd)
