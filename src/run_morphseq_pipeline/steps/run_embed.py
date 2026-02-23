from __future__ import annotations
from typing import Union, Optional

from pathlib import Path

from src.legacy.vae.auxiliary_scripts.embed_training_snips import embed_snips


def run_embed(
    root: Union[str, Path],
    train_name: str,
    model_dir: Optional[Union[str, Path]] = None,
    out_csv: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    simulate: bool = False,
    latent_dim: int = 16,
    seed: int = 0,
) -> None:
    embed_snips(
        root=root,
        train_name=train_name,
        model_dir=model_dir,
        out_csv=out_csv,
        batch_size=batch_size,
        simulate=simulate,
        latent_dim=latent_dim,
        seed=seed,
    )
    print("✔️  Embedding step complete.")

