from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.io.frame_snapshot_hash import compute_frame_snapshot_hash


def test_frame_snapshot_hash_is_deterministic_and_escapes() -> None:
    df = pd.DataFrame(
        {
            "image_id": ["i2", "i1"],
            "source_image_path": ["/a|b\nc", "/x\\y"],
            "source_micrometers_per_pixel": [0.6500000001, 0.65],
            "channel_id": ["BF", "BF"],
            "image_width_px": [10.0, 10],
            "image_height_px": [20, 20.0],
        }
    )
    h1 = compute_frame_snapshot_hash(df)
    h2 = compute_frame_snapshot_hash(df.copy())
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 16


def test_frame_snapshot_hash_fails_on_nulls() -> None:
    df = pd.DataFrame(
        {
            "image_id": ["i1"],
            "source_image_path": [None],
            "source_micrometers_per_pixel": [0.65],
            "channel_id": ["BF"],
            "image_width_px": [10],
            "image_height_px": [20],
        }
    )
    with pytest.raises(ValueError):
        compute_frame_snapshot_hash(df)

