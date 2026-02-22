"""Pure utility for stitching Keyence frame tiles with deterministic fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import skimage.util as skutil


Orientation = Literal["vertical", "horizontal"]
TilingMode = Literal["auto", "prior_only", "align_only"]
FallbackStep = Literal["per_frame", "master", "concat"]
FallbackUsed = Literal["none", "per_frame", "master", "concat"]


@dataclass(frozen=True)
class TileSpec:
    tile_id: str
    image: np.ndarray


@dataclass(frozen=True)
class TileTransform:
    tile_id: str
    dx_px: float
    dy_px: float
    source: Literal["identity", "align", "fallback"]
    confidence: float | None = None


@dataclass(frozen=True)
class TilingQC:
    passed: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float]
    suggested_action: Literal["ok", "use_master", "concat", "fail"]


@dataclass(frozen=True)
class FrameTilingConfig:
    orientation: Orientation
    mode: TilingMode = "auto"
    fallback_policy: tuple[FallbackStep, ...] = ("per_frame", "master", "concat")
    max_abs_shift_px: float = 500.0
    enable_alignment: bool = True
    transpose_after_stitch: bool = True
    use_legacy_canvas: bool = True
    compat_postprocess: bool = True
    invert_intensity: bool = True


@dataclass(frozen=True)
class FallbackParams:
    master_params_path: Path | None = None
    per_frame_params_path: Path | None = None


@dataclass(frozen=True)
class FrameTileResult:
    stitched: np.ndarray
    tile_transforms: dict[str, TileTransform]
    canvas_shape: tuple[int, int]
    qc: TilingQC
    fallback_used: FallbackUsed


def legacy_canvas_shape(
    n_tiles: int,
    orientation: Orientation,
    tile_shape: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if n_tiles <= 1:
        return (0, 0)
    shape_map = {
        2: np.array([800, 630]),
        3: np.array([1140, 630]) if orientation == "vertical" else np.array([1140, 480]),
    }
    if n_tiles not in shape_map:
        return (0, 0)
    target = shape_map[n_tiles].astype(float)
    if tile_shape is not None:
        tile_width = float(tile_shape[1])
        size_factor = tile_width / 640.0 if tile_width > 0 else 1.0
        if np.isfinite(size_factor) and size_factor > 0:
            target = target * size_factor
    return tuple(np.round(target).astype(int))


def stitch_frame_tiles(
    tile_specs: Sequence[TileSpec],
    config: FrameTilingConfig,
    fallback: FallbackParams | None = None,
) -> FrameTileResult:
    tiles = _prepare_tile_specs(tile_specs)
    if not tiles:
        raise ValueError("No tile specs provided")

    if len(tiles) == 1:
        stitched = _finalize_image(
            tiles[0].image,
            config=config,
            n_tiles=1,
            tile_shape=tiles[0].image.shape[:2],
        )
        qc = TilingQC(
            passed=True,
            reasons=tuple(),
            metrics={"tile_count": 1.0, "max_abs_shift_px": 0.0},
            suggested_action="ok",
        )
        tr = {tiles[0].tile_id: TileTransform(tiles[0].tile_id, 0.0, 0.0, source="identity")}
        return FrameTileResult(
            stitched=stitched,
            tile_transforms=tr,
            canvas_shape=stitched.shape[:2],
            qc=qc,
            fallback_used="none",
        )

    fallback = fallback or FallbackParams()
    initial = _init_transforms_from_priors(tiles)
    initial_qc = _run_tiling_qc(initial, config)

    if config.mode != "prior_only" and config.enable_alignment:
        try:
            stitched_align, tr_align = _stitch_with_stitch2d(
                tiles=tiles,
                orientation=config.orientation,
                load_params_path=None,
                run_align=True,
            )
            qc_align = _run_tiling_qc(tr_align, config)
            if qc_align.passed or config.mode == "align_only":
                stitched = _finalize_image(
                    stitched_align,
                    config=config,
                    n_tiles=len(tiles),
                    tile_shape=tiles[0].image.shape[:2],
                )
                return FrameTileResult(
                    stitched=stitched,
                    tile_transforms=tr_align,
                    canvas_shape=stitched.shape[:2],
                    qc=qc_align,
                    fallback_used="none",
                )
            initial_qc = qc_align
        except Exception:
            initial_qc = TilingQC(
                passed=False,
                reasons=("alignment_failed",),
                metrics=initial_qc.metrics,
                suggested_action="use_master",
            )

    for step in config.fallback_policy:
        if step == "per_frame" and fallback.per_frame_params_path is not None and fallback.per_frame_params_path.exists():
            try:
                stitched_pf, tr_pf = _stitch_with_stitch2d(
                    tiles=tiles,
                    orientation=config.orientation,
                    load_params_path=fallback.per_frame_params_path,
                    run_align=False,
                )
                qc_pf = _run_tiling_qc(tr_pf, config)
                stitched = _finalize_image(
                    stitched_pf,
                    config=config,
                    n_tiles=len(tiles),
                    tile_shape=tiles[0].image.shape[:2],
                )
                return FrameTileResult(
                    stitched=stitched,
                    tile_transforms=tr_pf,
                    canvas_shape=stitched.shape[:2],
                    qc=qc_pf,
                    fallback_used="per_frame",
                )
            except Exception:
                pass

        if step == "master" and fallback.master_params_path is not None and fallback.master_params_path.exists():
            try:
                stitched_master, tr_master = _stitch_with_stitch2d(
                    tiles=tiles,
                    orientation=config.orientation,
                    load_params_path=fallback.master_params_path,
                    run_align=False,
                )
                qc_master = _run_tiling_qc(tr_master, config)
                stitched = _finalize_image(
                    stitched_master,
                    config=config,
                    n_tiles=len(tiles),
                    tile_shape=tiles[0].image.shape[:2],
                )
                return FrameTileResult(
                    stitched=stitched,
                    tile_transforms=tr_master,
                    canvas_shape=stitched.shape[:2],
                    qc=qc_master,
                    fallback_used="master",
                )
            except Exception:
                pass

        if step == "concat":
            stitched_concat, tr_concat = _concat_tiles(tiles=tiles, orientation=config.orientation)
            qc_concat = _run_tiling_qc(tr_concat, config, allow_zero_shift=False)
            stitched = _finalize_image(
                stitched_concat,
                config=config,
                n_tiles=len(tiles),
                tile_shape=tiles[0].image.shape[:2],
            )
            return FrameTileResult(
                stitched=stitched,
                tile_transforms=tr_concat,
                canvas_shape=stitched.shape[:2],
                qc=qc_concat,
                fallback_used="concat",
            )

    stitched_default, tr_default = _concat_tiles(tiles=tiles, orientation=config.orientation)
    stitched_default = _finalize_image(
        stitched_default,
        config=config,
        n_tiles=len(tiles),
        tile_shape=tiles[0].image.shape[:2],
    )
    qc_default = _run_tiling_qc(tr_default, config, allow_zero_shift=False)
    return FrameTileResult(
        stitched=stitched_default,
        tile_transforms=tr_default,
        canvas_shape=stitched_default.shape[:2],
        qc=qc_default,
        fallback_used="concat",
    )


def _prepare_tile_specs(tile_specs: Sequence[TileSpec]) -> list[TileSpec]:
    out: list[TileSpec] = []
    for spec in tile_specs:
        if not isinstance(spec.image, np.ndarray):
            raise TypeError(f"Tile '{spec.tile_id}' image must be numpy.ndarray")
        if spec.image.ndim not in (2, 3):
            raise ValueError(f"Tile '{spec.tile_id}' image must be 2D or 3D")
        image = spec.image if spec.image.dtype == np.uint8 else skutil.img_as_ubyte(spec.image)
        out.append(TileSpec(tile_id=str(spec.tile_id), image=image))
    out.sort(key=lambda item: item.tile_id)
    return out


def _init_transforms_from_priors(tiles: Sequence[TileSpec]) -> dict[str, TileTransform]:
    return {
        tile.tile_id: TileTransform(tile_id=tile.tile_id, dx_px=0.0, dy_px=0.0, source="identity")
        for tile in tiles
    }


def _coords_to_transforms(tiles: Sequence[TileSpec], coords: dict) -> dict[str, TileTransform]:
    out: dict[str, TileTransform] = {}
    for idx, tile in enumerate(tiles):
        value = coords.get(idx, coords.get(str(idx), (0.0, 0.0)))
        x_val = float(value[0]) if len(value) > 0 else 0.0
        y_val = float(value[1]) if len(value) > 1 else 0.0
        out[tile.tile_id] = TileTransform(
            tile_id=tile.tile_id,
            dx_px=x_val,
            dy_px=y_val,
            source="align",
        )
    return out


def _stitch_with_stitch2d(
    tiles: Sequence[TileSpec],
    orientation: Orientation,
    load_params_path: Path | None,
    run_align: bool,
) -> tuple[np.ndarray, dict[str, TileTransform]]:
    from stitch2d import StructuredMosaic
    from stitch2d.tile import OpenCVTile, Tile

    tile_images = [tile.image for tile in tiles]
    if load_params_path is not None:
        mosaic = StructuredMosaic(
            [Tile(img) for img in tile_images],
            dim=len(tile_images),
            origin="upper left",
            direction=orientation,
            pattern="raster",
        )
        mosaic.load_params(str(load_params_path))
    else:
        mosaic = StructuredMosaic(
            [OpenCVTile(img) for img in tile_images],
            dim=len(tile_images),
            origin="upper left",
            direction=orientation,
            pattern="raster",
        )
        if run_align:
            mosaic.align()

    coords = mosaic.params.get("coords", {})
    if len(coords) != len(tile_images):
        raise RuntimeError("incomplete tile alignment")
    transforms = _coords_to_transforms(tiles, coords)

    mosaic.reset_tiles()
    mosaic.smooth_seams()
    stitched = mosaic.stitch()
    return stitched, transforms


def _concat_tiles(
    tiles: Sequence[TileSpec],
    orientation: Orientation,
) -> tuple[np.ndarray, dict[str, TileTransform]]:
    concat_axis = 0 if orientation == "vertical" else 1
    stitched = np.concatenate([tile.image for tile in tiles], axis=concat_axis)

    transforms: dict[str, TileTransform] = {}
    cursor_x = 0.0
    cursor_y = 0.0
    for tile in tiles:
        transforms[tile.tile_id] = TileTransform(
            tile_id=tile.tile_id,
            dx_px=cursor_x,
            dy_px=cursor_y,
            source="fallback",
        )
        if concat_axis == 1:
            cursor_x += float(tile.image.shape[1])
        else:
            cursor_y += float(tile.image.shape[0])
    return stitched, transforms


def _run_tiling_qc(
    transforms: dict[str, TileTransform],
    config: FrameTilingConfig,
    allow_zero_shift: bool = True,
) -> TilingQC:
    max_abs_shift = 0.0
    all_zero = True
    for tr in transforms.values():
        local_max = max(abs(float(tr.dx_px)), abs(float(tr.dy_px)))
        max_abs_shift = max(max_abs_shift, local_max)
        if local_max > 0:
            all_zero = False

    reasons: list[str] = []
    if max_abs_shift > float(config.max_abs_shift_px):
        reasons.append("shift_exceeds_threshold")
    if not allow_zero_shift and all_zero:
        reasons.append("unresolved_transforms")

    passed = len(reasons) == 0
    suggested_action: Literal["ok", "use_master", "concat", "fail"] = "ok" if passed else "concat"
    metrics = {
        "max_abs_shift_px": float(max_abs_shift),
        "tile_count": float(len(transforms)),
    }
    return TilingQC(
        passed=passed,
        reasons=tuple(reasons),
        metrics=metrics,
        suggested_action=suggested_action,
    )


def _trim_to_shape(image: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    target_y, target_x = target
    image_y, image_x = image.shape[:2]

    pad_y = max(0, target_y - image_y)
    pad_x = max(0, target_x - image_x)
    if pad_y or pad_x:
        image = np.pad(
            image,
            (
                (pad_y // 2, pad_y - pad_y // 2),
                (pad_x // 2, pad_x - pad_x // 2),
            ),
            mode="constant",
        )

    start_y = (image.shape[0] - target_y) // 2
    start_x = (image.shape[1] - target_x) // 2
    return image[start_y : start_y + target_y, start_x : start_x + target_x]


def _finalize_image(
    image: np.ndarray,
    config: FrameTilingConfig,
    n_tiles: int,
    tile_shape: tuple[int, int],
) -> np.ndarray:
    out = image
    if n_tiles > 1 and config.orientation == "horizontal" and config.transpose_after_stitch:
        out = out.T

    if config.use_legacy_canvas:
        target = legacy_canvas_shape(
            n_tiles=n_tiles,
            orientation=config.orientation,
            tile_shape=tile_shape,
        )
        if target != (0, 0):
            out = _trim_to_shape(out, target)

    if config.compat_postprocess and config.invert_intensity:
        out = np.iinfo(out.dtype).max - out
    return out
