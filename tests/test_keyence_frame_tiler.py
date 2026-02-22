from __future__ import annotations

import numpy as np

from src.data_pipeline.image_building.utils.frame_tiler import FrameTilingConfig
from src.data_pipeline.image_building.utils.frame_tiler import TileSpec
from src.data_pipeline.image_building.utils.frame_tiler import legacy_canvas_shape
from src.data_pipeline.image_building.utils.frame_tiler import stitch_frame_tiles


def test_single_tile_short_circuit_preserves_shape_and_inverts() -> None:
    image = np.full((4, 5), 10, dtype=np.uint8)
    result = stitch_frame_tiles(
        tile_specs=[TileSpec(tile_id="tile_a", image=image)],
        config=FrameTilingConfig(
            orientation="horizontal",
            mode="auto",
            fallback_policy=("concat",),
            enable_alignment=False,
            use_legacy_canvas=True,
            compat_postprocess=True,
            invert_intensity=True,
        ),
    )

    assert result.fallback_used == "none"
    assert result.stitched.shape == (4, 5)
    assert np.all(result.stitched == 245)


def test_concat_fallback_is_deterministic_and_sorted_by_tile_id() -> None:
    tiles = [
        TileSpec(tile_id="b", image=np.full((2, 3), 10, dtype=np.uint8)),
        TileSpec(tile_id="a", image=np.full((2, 3), 20, dtype=np.uint8)),
        TileSpec(tile_id="c", image=np.full((2, 3), 30, dtype=np.uint8)),
    ]
    result = stitch_frame_tiles(
        tile_specs=tiles,
        config=FrameTilingConfig(
            orientation="horizontal",
            mode="auto",
            fallback_policy=("concat",),
            enable_alignment=False,
            use_legacy_canvas=False,
            compat_postprocess=True,
            invert_intensity=True,
        ),
    )

    assert result.fallback_used == "concat"
    assert result.stitched.shape == (9, 2)  # concat width then transpose
    # Sorted order is a, b, c -> inverted intensities 235, 245, 225 by column blocks.
    assert np.all(result.stitched[0:3, :] == 235)
    assert np.all(result.stitched[3:6, :] == 245)
    assert np.all(result.stitched[6:9, :] == 225)
    assert result.tile_transforms["a"].dx_px == 0.0
    assert result.tile_transforms["b"].dx_px == 3.0
    assert result.tile_transforms["c"].dx_px == 6.0


def test_legacy_canvas_shape_matches_scaled_reference() -> None:
    assert legacy_canvas_shape(3, "horizontal", tile_shape=(1440, 1920)) == (3420, 1440)
