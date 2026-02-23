import os
from pathlib import Path

import numpy as np
import pytest
import imageio.v2 as iio

# Guard against pandas ABI mismatch (e.g., NumPy 2.0 with older pandas builds)
try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    pytest.skip(f"Skipping Build03A integration tests due to pandas import error: {e}", allow_module_level=True)

# Import Build03A module lazily and skip these tests if optional dependencies
# (e.g., stitch2d) are not available in the environment.
_build03_mod = pytest.importorskip(
    "src.build.build03A_process_images",
    reason="Build03A integration tests require optional deps (e.g., stitch2d)",
)
get_embryo_stats = _build03_mod.get_embryo_stats


def _write_png(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(path), arr.astype(np.uint8))


def _make_sam2_mask(tmp_root: Path, date: str, mask_filename: str, label_coords):
    base = tmp_root / date / "masks"
    arr = np.zeros((7, 7), dtype=np.uint8)
    for y, x in label_coords:
        arr[y, x] = 1  # integer label 1
    _write_png(base / mask_filename, arr)
    return base / mask_filename


def _make_build02_mask(tmp_root: Path, date: str, kind: str, stub: str, coords):
    # dir name containing keyword, emulating `<root>/segmentation/{kind}_modelX/{date}`
    folder = tmp_root / "segmentation" / f"{kind}_modelX" / date
    arr = np.zeros((7, 7), dtype=np.uint8)
    for y, x in coords:
        arr[y, x] = 255  # legacy binary
    _write_png(folder / f"foo_{stub}_bar.tif", arr)


def _row(date: str, well: str, t: int, mask_fname: str):
    # Minimal row with required fields for get_embryo_stats()
    return {
        "experiment_date": date,
        "well": well,
        "time_int": t,
        "region_label": 1,
        "exported_mask_path": mask_fname,
        "Height (um)": 10.0,
        "Height (px)": 10.0,
        "xpos": 3.0,
        "ypos": 3.0,
        "Time Rel (s)": 0.0,
        "area_px": 9.0,
    }


def test_get_embryo_stats_sam2_only_fraction_alive_nan(tmp_path, monkeypatch):
    date = "20240101"
    well = "A01"
    t = 0
    mask_fname = f"{date}_{well}_ch00_t{t:04d}_masks_emnum_1.png"

    # Point SAM2 mask base to tmp dir
    sam2_base = tmp_path / "sam2_masks"
    monkeypatch.setenv("MORPHSEQ_SANDBOX_MASKS_DIR", str(sam2_base))

    # Create a small integer-labeled SAM2 mask (label 1 at a few pixels)
    _make_sam2_mask(sam2_base, date, mask_fname, label_coords=[(3, 3), (3, 4), (4, 3)])

    df = pd.DataFrame([_row(date, well, t, mask_fname)])
    out = get_embryo_stats(0, root=tmp_path, embryo_metadata_df=df, qc_scale_um=150, ld_rat_thresh=0.9)

    val = out.loc[out.index[0], "fraction_alive"]
    assert not np.isfinite(val)  # NaN when via mask is absent
    assert out.loc[out.index[0], "dead_flag"] is False


def test_get_embryo_stats_with_via_mask(tmp_path, monkeypatch):
    date = "20240101"
    well = "A01"
    t = 0
    mask_fname = f"{date}_{well}_ch00_t{t:04d}_masks_emnum_1.png"

    # Point SAM2 mask base to tmp dir
    sam2_base = tmp_path / "sam2_masks"
    monkeypatch.setenv("MORPHSEQ_SANDBOX_MASKS_DIR", str(sam2_base))

    # SAM2 embryo label pixels (3 pixels)
    _make_sam2_mask(sam2_base, date, mask_fname, label_coords=[(3, 3), (3, 4), (4, 3)])

    # Build02 via mask overlapping 1 of those pixels => fraction_alive = 2/3 < 0.9 => dead_flag True
    stub = f"{well}_t{t:04d}"
    _make_build02_mask(tmp_path, date, kind="via", stub=stub, coords=[(3, 3)])

    df = pd.DataFrame([_row(date, well, t, mask_fname)])
    out = get_embryo_stats(0, root=tmp_path, embryo_metadata_df=df, qc_scale_um=150, ld_rat_thresh=0.9)

    frac = out.loc[out.index[0], "fraction_alive"]
    assert np.isfinite(frac)
    assert 0.0 <= frac <= 1.0
    assert out.loc[out.index[0], "dead_flag"] is True
