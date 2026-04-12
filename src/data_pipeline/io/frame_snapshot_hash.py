from __future__ import annotations

import hashlib

import pandas as pd
import numpy as np


SNAPSHOT_COLUMNS_ORDER: list[str] = [
    "image_id",
    "source_image_path",
    "source_micrometers_per_pixel",
    "channel_id",
    "image_width_px",
    "image_height_px",
]


def compute_frame_snapshot_hash(snapshot_df: pd.DataFrame) -> str:
    """
    Compute a deterministic snapshot hash for the per-image physical inventory fields
    that downstream stages (snip processing) rely on.

    Determinism rules (do not change without bumping schema_version):
    - fixed column order: SNAPSHOT_COLUMNS_ORDER
    - rows sorted by image_id (string)
    - delimiter: '|'
    - newline: '\\n'
    - floats formatted as '%.8f'
    - no index column

    Returns:
        16-hex-character sha256 prefix.
    """
    missing = [c for c in SNAPSHOT_COLUMNS_ORDER if c not in snapshot_df.columns]
    if missing:
        raise ValueError(f"snapshot_df missing required columns for hashing: {missing}")

    df = snapshot_df.loc[:, SNAPSHOT_COLUMNS_ORDER].copy()

    # Fail fast on nulls: hashing nulls as "nan"/"None" makes drift detection ambiguous.
    null_cols = [c for c in SNAPSHOT_COLUMNS_ORDER if df[c].isna().any()]
    if null_cols:
        counts = {c: int(df[c].isna().sum()) for c in null_cols}
        raise ValueError(f"snapshot_df contains nulls in required columns: {counts}")

    # Canonicalize types.
    df["image_id"] = df["image_id"].astype(str)
    df["source_image_path"] = df["source_image_path"].astype(str)
    df["channel_id"] = df["channel_id"].astype(str)

    w = pd.to_numeric(df["image_width_px"], errors="raise").astype(float)
    h = pd.to_numeric(df["image_height_px"], errors="raise").astype(float)
    if not np.isfinite(w).all() or not np.isfinite(h).all():
        raise ValueError("image_width_px/image_height_px must be finite")
    if not (np.equal(w, np.floor(w)).all() and np.equal(h, np.floor(h)).all()):
        raise ValueError("image_width_px/image_height_px must be integer-like")
    df["image_width_px"] = w.astype(int)
    df["image_height_px"] = h.astype(int)

    um = pd.to_numeric(df["source_micrometers_per_pixel"], errors="raise").astype(float)
    if not np.isfinite(um).all():
        raise ValueError("source_micrometers_per_pixel must be finite")
    df["source_micrometers_per_pixel"] = um

    df = df.sort_values(["image_id", "source_image_path"], kind="mergesort").reset_index(drop=True)

    def _esc(s: str) -> str:
        # Prevent ambiguous serialization if a field contains our delimiter/newline.
        return (
            s.replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )

    # Deterministic line format.
    lines: list[str] = []
    for row in df.itertuples(index=False):
        # row order matches SNAPSHOT_COLUMNS_ORDER
        image_id, path, um_per_px, channel_id, w, h = row
        lines.append(
            f"{_esc(str(image_id))}|{_esc(str(path))}|{float(um_per_px):.8f}|{_esc(str(channel_id))}|{int(w)}|{int(h)}"
        )

    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
