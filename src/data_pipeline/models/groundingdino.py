"""GroundingDINO loader wrapper.

We do not assume `groundingdino` is pip-installed. Instead, we support loading from a
checked-out repo directory (e.g. under DATA_ROOT/models/GroundingDINO).
"""

from __future__ import annotations

import sys
from pathlib import Path

from data_pipeline.utils.cuda_diagnostics import resolve_device


def load_groundingdino_model(
    *,
    repo_dir: Path,
    config_path: Path,
    weights_path: Path,
    device: str = "cuda",
):
    device = resolve_device(device)
    repo_dir = Path(repo_dir)
    config_path = Path(config_path)
    weights_path = Path(weights_path)

    if not repo_dir.exists():
        raise FileNotFoundError(f"GroundingDINO repo_dir not found: {repo_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"GroundingDINO config not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"GroundingDINO weights not found: {weights_path}")

    # Make repo importable so `import groundingdino...` works.
    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    try:
        from groundingdino.util.inference import load_model  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Failed to import GroundingDINO from repo_dir. "
            f"repo_dir={repo_dir} (expected to contain python package `groundingdino`)."
        ) from e

    # Try the repo's stock loader first. If it fails due to PyTorch 2.6+'s
    # `torch.load(weights_only=True)` default, fall back to a safe, explicit
    # load path that allowlists the minimal globals needed and keeps
    # `weights_only=True`.
    #
    # This is required for some fine-tuned checkpoints that store extra Python
    # objects (e.g. argparse.Namespace) in the serialized file. Only use this
    # with trusted checkpoints.
    try:
        return load_model(str(config_path), str(weights_path), device=device)
    except Exception:
        import argparse
        import torch  # local import to keep module import cheap

        from groundingdino.models import build_model  # type: ignore
        from groundingdino.util.misc import clean_state_dict  # type: ignore
        from groundingdino.util.slconfig import SLConfig  # type: ignore

        args = SLConfig.fromfile(str(config_path))
        args.device = str(device)
        model = build_model(args)
        # Allowlist common innocuous globals found in training checkpoints.
        torch.serialization.add_safe_globals([argparse.Namespace])
        # `mmap=True` reduces peak RSS for large training checkpoints.
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=True, mmap=True)

        state = ckpt.get("model") if isinstance(ckpt, dict) else None
        if state is None:
            raise ValueError("Unexpected GroundingDINO checkpoint format: missing `model` key")
        # Drop optimizer/etc as early as possible.
        if isinstance(ckpt, dict):
            for k in list(ckpt.keys()):
                if k != "model":
                    ckpt.pop(k, None)

        model.load_state_dict(clean_state_dict(state), strict=False)
        model.eval()
        return model
