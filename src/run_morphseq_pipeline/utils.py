from __future__ import annotations
from pathlib import Path
import pandas as pd


def ensure_parent(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def sample_tracked_df(
    df: pd.DataFrame,
    by_embryo: int | None = None,
    frames_per_embryo: int | None = None,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Deterministically sample dataframe by embryo and frames.

    - If by_embryo is set: keep first N distinct embryos.
    - If frames_per_embryo is set: keep first M frames per embryo.
    - If max_samples is set: cap rows at first K rows.
    """
    out = df
    if by_embryo:
        keep_embryos = (
            out["embryo_id"].drop_duplicates().iloc[:by_embryo].tolist()
        )
        out = out[out["embryo_id"].isin(keep_embryos)].copy()
    if frames_per_embryo:
        out = (
            out.sort_values(["embryo_id", "time_int"])  # assume time_int is int
               .groupby("embryo_id", as_index=False, group_keys=False)
               .apply(lambda g: g.head(frames_per_embryo))
        )
    if max_samples:
        out = out.head(max_samples).copy()
    return out.reset_index(drop=True)

