from __future__ import annotations
from pathlib import Path

from src.build.build05_make_training_snips import make_image_snips


def run_build05(
    root: str | Path,
    train_name: str,
    label_var: str | None = None,
    rs_factor: float = 1.0,
    overwrite: bool = False,
) -> None:
    make_image_snips(root=str(Path(root)), train_name=train_name, label_var=label_var,
                     rs_factor=rs_factor, overwrite_flag=overwrite)
    print("✔️  Build05 complete.")
