from __future__ import annotations

from pathlib import Path

from src.data_pipeline.metadata_ingest.scope.keyence_scope_metadata import _extract_time_int_from_path
from src.data_pipeline.metadata_ingest.stitched_index.materialize_stitched_images import (
    _keyence_canvas_shape,
    _infer_keyence_stack_lookup,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_lookup_xy_without_t_uses_filename_token_as_tile_and_time0(tmp_path: Path) -> None:
    raw_dir = tmp_path / "20240509_24hpf"
    for tile in (1, 2, 3):
        for z in (1, 2):
            _touch(raw_dir / "XY16" / f"embryo__XY16_{tile:05d}_Z{z:03d}_CH1.tif")

    lookup = _infer_keyence_stack_lookup(raw_dir)
    key = ("B04", 0)
    assert key in lookup
    assert set(lookup[key].keys()) == {1, 2, 3}
    assert all(len(lookup[key][tile]) == 2 for tile in (1, 2, 3))
    assert ("B04", 1) not in lookup
    assert ("B04", 2) not in lookup


def test_lookup_with_p_and_t_dirs_uses_dir_time_and_p_tile(tmp_path: Path) -> None:
    raw_dir = tmp_path / "20230613"
    for tile in (1, 2):
        for z in (1, 2):
            _touch(
                raw_dir
                / "W057"
                / f"P{tile:05d}"
                / "T0034"
                / f"img_T0034_Z{z:03d}_CH1.tif"
            )

    lookup = _infer_keyence_stack_lookup(raw_dir)
    key = ("E09", 33)
    assert key in lookup
    assert set(lookup[key].keys()) == {1, 2}
    assert all(len(lookup[key][tile]) == 2 for tile in (1, 2))


def test_scope_time_parser_matches_legacy_semantics() -> None:
    assert _extract_time_int_from_path(Path("XY16/embryo__XY16_00003_Z001_CH1.tif")) == 0
    assert _extract_time_int_from_path(Path("W057/P00001/T0034/img_Z001_CH1.tif")) == 33
    assert _extract_time_int_from_path(Path("W057/P00001/img_T0007_Z001_CH1.tif")) == 6


def test_canvas_shape_scales_to_native_tile_size() -> None:
    assert _keyence_canvas_shape(3, "horizontal", tile_shape=(1440, 1920)) == (3420, 1440)
