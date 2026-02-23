from __future__ import annotations
from pathlib import Path

try:
    # Use the legacy, proven combine function from Archive for now
    from src.build._Archive.build03A_process_embryos_main_par import build_well_metadata_master as _combine
except Exception:  # pragma: no cover
    _combine = None


def run_combine_metadata(root: str | Path) -> None:
    if _combine is None:
        raise SystemExit("Combine metadata not available: build_well_metadata_master import failed")
    _combine(root=Path(root))
    print("✔️  Combined master well metadata written.")
