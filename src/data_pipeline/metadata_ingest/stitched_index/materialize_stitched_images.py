"""Materialize stitched images from raw inputs and emit stitched_image_index.csv."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable
import re

import nd2
import numpy as np
import pandas as pd
from skimage import exposure
import skimage.io as skio
import skimage.util as skutil
import torch

from data_pipeline.image_building.shared.log_focus import LoG_focus_stacker
from data_pipeline.image_building.shared.log_focus import im_rescale

log = logging.getLogger(__name__)


def _parse_selected_wells(selected_wells: Iterable[str] | None) -> set[str]:
    return {str(w).strip() for w in (selected_wells or []) if str(w).strip()}


def _well_to_xy_dir_name(well_index: str) -> str:
    row = ord(well_index[0].upper()) - ord("A")
    col = int(well_index[1:])
    xy_idx = row * 12 + col
    return f"XY{xy_idx:02d}"


def _normalize_extension(extension: str | None) -> str:
    ext = (extension or "jpg").strip().lower()
    if ext.startswith("."):
        ext = ext[1:]
    if ext not in {"jpg", "jpeg", "png", "tif", "tiff"}:
        raise ValueError(f"Unsupported output image extension: {extension}")
    return "jpg" if ext == "jpeg" else ("tif" if ext == "tiff" else ext)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    im = image.astype(np.float32)
    im_min = float(np.min(im))
    im_max = float(np.max(im))
    if im_max <= im_min:
        return np.zeros_like(im, dtype=np.uint8)
    scaled = (im - im_min) / (im_max - im_min)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _write_image(output_path: Path, image: np.ndarray, overwrite: bool, image_extension: str) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return "existing"

    if image_extension in {"jpg", "png"}:
        image = _to_uint8(image)
    elif image.dtype != np.uint16:
        image = image.astype(np.uint16)

    skio.imsave(output_path, image, check_contrast=False)
    return "materialized"


def _write_placeholder_image(
    output_path: Path, width: int, height: int, value: int, overwrite: bool, image_extension: str
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return "existing"

    if image_extension in {"jpg", "png"}:
        image = np.zeros((max(height, 1), max(width, 1)), dtype=np.uint8)
        image[0, 0] = np.uint8(value % 255)
    else:
        image = np.zeros((max(height, 1), max(width, 1)), dtype=np.uint16)
        image[0, 0] = np.uint16(value % 65535)

    skio.imsave(output_path, image, check_contrast=False)
    return "placeholder"


def _build_yx1_well_to_series_map(mapping_csv: Path | None) -> dict[str, int]:
    if mapping_csv is None or not mapping_csv.exists():
        return {}
    mapping_df = pd.read_csv(mapping_csv)
    out: dict[str, int] = {}
    for _, row in mapping_df.iterrows():
        series_number = row.get("series_number")
        well_index = row.get("well_index")
        if pd.isna(series_number) or pd.isna(well_index):
            continue
        # Convert to 0-based ND2 series index.
        out[str(well_index)] = int(series_number) - 1
    return out


def _select_yx1_channel_index(channel_names: list[str]) -> int:
    lower = [str(name).lower() for name in channel_names]
    if "bf" in lower:
        return lower.index("bf")
    for idx, name in enumerate(lower):
        if any(token in name for token in ("dia", "brightfield", "empty")):
            return idx
    return 0


def _materialize_yx1_image(
    nd: nd2.ND2File,
    dask_arr,
    time_int: int,
    series_index: int,
    channel_index: int,
    device: str,
) -> np.ndarray:
    if dask_arr.ndim == 6:
        stack = dask_arr[time_int, series_index, :, channel_index, :, :].compute()
    elif dask_arr.ndim == 5:
        stack = dask_arr[time_int, series_index, :, :, :].compute()
    else:
        raise ValueError(f"Unexpected ND2 dimensions: {dask_arr.ndim}")

    # Match legacy YX1 processing: percentile-rescale stack then LoG focus stack.
    norm, _, _ = im_rescale(stack)
    ff, _ = LoG_focus_stacker(norm.astype(np.float32), filter_size=3, device=device)
    ff_np = ff.detach().cpu().numpy() if torch.is_tensor(ff) else np.asarray(ff)
    ff_u16 = np.clip(ff_np, 0, 65535).astype(np.uint16)
    return skutil.img_as_ubyte(ff_u16)


def _well_from_w_index(raw: int) -> str:
    row = (raw - 1) // 12
    col = (raw - 1) % 12 + 1
    return f"{chr(65 + row)}{col:02d}"


def _extract_keyence_well_and_tile(path: Path) -> tuple[str | None, int]:
    tile_id: int | None = None

    for part in path.parts:
        p_match = re.fullmatch(r"P(\d+)", part, flags=re.IGNORECASE)
        if p_match:
            tile_id = int(p_match.group(1))
            break

    for part in path.parts:
        xy_match = re.fullmatch(r"XY(\d+)([A-Za-z]?)", part, flags=re.IGNORECASE)
        if xy_match:
            xy_raw = int(xy_match.group(1))
            suffix = xy_match.group(2)
            if suffix:
                well_index = f"{suffix.upper()}{xy_raw:02d}"
                if tile_id is None:
                    tile_id = max(ord(suffix.lower()) - 96, 1)
            else:
                well_index = _well_from_w_index(xy_raw)
            return well_index, tile_id or 1

    for part in path.parts:
        w_match = re.fullmatch(r"W0?(\d+)", part, flags=re.IGNORECASE)
        if w_match:
            return _well_from_w_index(int(w_match.group(1))), tile_id or 1

    name_match = re.search(r"([A-H](?:0[1-9]|1[0-2]))", path.name)
    if name_match:
        return name_match.group(1), tile_id or 1

    return None, tile_id or 1


def _parse_keyence_time_and_z(path: Path) -> tuple[int, int] | None:
    match = re.search(r"(?:_T|_)(\d+)_Z(\d+)_CH\d+", path.name, flags=re.IGNORECASE)
    if not match:
        return None
    time_raw = int(match.group(1))
    time_int = max(time_raw - 1, 0)
    z_index = int(match.group(2))
    return time_int, z_index


def _infer_keyence_stack_lookup(raw_images_dir: Path) -> dict[tuple[str, int], dict[int, list[Path]]]:
    lookup: dict[tuple[str, int], dict[int, list[tuple[int, Path]]]] = {}
    for path in raw_images_dir.rglob("*CH*.tif"):
        well_index, tile_id = _extract_keyence_well_and_tile(path)
        if well_index is None:
            continue

        parsed = _parse_keyence_time_and_z(path)
        if parsed is None:
            continue
        time_int, z_index = parsed
        key = (well_index, time_int)
        lookup.setdefault(key, {}).setdefault(tile_id, []).append((z_index, path))

    out: dict[tuple[str, int], dict[int, list[Path]]] = {}
    for key, tile_dict in lookup.items():
        out[key] = {}
        for tile_id, z_pairs in tile_dict.items():
            out[key][tile_id] = [path for _, path in sorted(z_pairs, key=lambda pair: pair[0])]
    return out


def _materialize_keyence_tile_projection(stack_paths: list[Path]) -> np.ndarray:
    frames = [skio.imread(path) for path in stack_paths]
    stack = np.stack(frames, axis=0)
    projected = np.max(stack, axis=0)
    if projected.dtype != np.uint16:
        projected = projected.astype(np.uint16)
    return projected


def _resolve_device(device_preference: str | None) -> str:
    pref = str(device_preference or "cuda").strip().lower()
    if pref.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _compute_joint_percentile_bounds(stacks: list[np.ndarray]) -> tuple[float, float]:
    if not stacks:
        return 0.0, 1.0

    max_samples = 1_000_000
    frac = 0.01
    total_vox = int(sum(stack.size for stack in stacks))
    n_want = int(min(max_samples, max(1, int(total_vox * frac))))
    per_stack = int(max(1, n_want // len(stacks)))

    rng = np.random.default_rng(42)
    samples = []
    for stack in stacks:
        n = int(stack.size)
        k = min(n, per_stack)
        if k >= n:
            samples.append(stack.reshape(-1))
        else:
            idx = rng.choice(n, k, replace=False)
            samples.append(stack.reshape(-1)[idx])

    all_samp = np.concatenate(samples)
    lo, hi = np.percentile(all_samp, [0.1, 99.9])
    if float(hi) <= float(lo):
        hi = lo + 1.0
    return float(lo), float(hi)


def _keyence_filter_size(micrometers_per_pixel: float, ff_filter_res_um: float) -> int:
    pixel_size = float(micrometers_per_pixel) if micrometers_per_pixel and micrometers_per_pixel > 0 else np.nan
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        return 3
    filter_rad = max(1, int(round(float(ff_filter_res_um) / pixel_size)))
    return int(2 * filter_rad + 1)


def _materialize_keyence_ff_tiles_with_log(
    tile_stacks: dict[int, list[Path]],
    device: str,
    filter_size: int,
) -> list[np.ndarray]:
    tile_ids = sorted(tile_stacks.keys())
    stacks = []
    for tile_id in tile_ids:
        frames = [skio.imread(path) for path in tile_stacks[tile_id]]
        stacks.append(np.stack(frames, axis=0))

    lo, hi = _compute_joint_percentile_bounds(stacks)
    ff_tiles: list[np.ndarray] = []
    for stack in stacks:
        norm = exposure.rescale_intensity(stack, in_range=(lo, hi), out_range=(0, 1)).astype(np.float32)
        ff, _ = LoG_focus_stacker(norm, filter_size=filter_size, device=device)
        ff_np = ff.detach().cpu().numpy() if torch.is_tensor(ff) else np.asarray(ff)
        ff_tiles.append(skutil.img_as_ubyte(np.clip(ff_np, 0.0, 1.0)))
    return ff_tiles


def _keyence_orientation(experiment: str) -> str:
    year_match = re.match(r"(\d{4})", experiment)
    if year_match and int(year_match.group(1)) < 2024:
        return "vertical"
    return "horizontal"


def _keyence_master_params_path(raw_images_dir: Path, experiment: str) -> Path | None:
    for parent in raw_images_dir.parents:
        if parent.name == "raw_image_data":
            candidate = parent.parent / "built_image_data" / "Keyence" / "FF_images" / experiment / "master_params.json"
            return candidate if candidate.exists() else None
    return None


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


def _keyence_canvas_shape(n_tiles: int, orientation: str) -> tuple[int, int]:
    shape_map = {
        1: np.array([480, 640]),
        2: np.array([800, 630]),
        3: np.array([1140, 630]) if orientation == "vertical" else np.array([1140, 480]),
    }
    if n_tiles in shape_map:
        return tuple(shape_map[n_tiles].astype(int))
    # Fallback for non-standard tile counts: preserve tile geometry by concatenation.
    return (0, 0)


def _stitch_keyence_tile_projections(
    tile_projections: list[np.ndarray],
    orientation: str,
    master_params_path: Path | None,
) -> np.ndarray:
    tile_images = [
        img if img.dtype == np.uint8 else skutil.img_as_ubyte(img)
        for img in tile_projections
    ]

    n_tiles = len(tile_images)
    if n_tiles == 1:
        stitched = tile_images[0]
    else:
        try:
            from stitch2d import StructuredMosaic
            from stitch2d.tile import OpenCVTile, Tile

            if master_params_path is not None and master_params_path.exists():
                mosaic = StructuredMosaic(
                    [Tile(img) for img in tile_images],
                    dim=len(tile_images),
                    origin="upper left",
                    direction=orientation,
                    pattern="raster",
                )
                mosaic.load_params(str(master_params_path))
            else:
                mosaic = StructuredMosaic(
                    [OpenCVTile(img) for img in tile_images],
                    dim=len(tile_images),
                    origin="upper left",
                    direction=orientation,
                    pattern="raster",
                )
                mosaic.align()
                if len(mosaic.params.get("coords", {})) != len(tile_images):
                    raise RuntimeError("incomplete keyence tile alignment")
            mosaic.reset_tiles()
            mosaic.smooth_seams()
            stitched = mosaic.stitch()
        except Exception as exc:  # pragma: no cover - exercised in integration with stitch2d
            log.warning("Falling back to deterministic Keyence tile concatenation: %s", exc)
            concat_axis = 0 if orientation == "vertical" else 1
            stitched = np.concatenate(tile_images, axis=concat_axis)

    if n_tiles > 1 and orientation == "horizontal":
        stitched = stitched.T

    target = _keyence_canvas_shape(n_tiles, orientation)
    if target != (0, 0):
        stitched = _trim_to_shape(stitched, target)

    # Legacy Keyence stitched_FF_images convention is intensity inversion.
    return np.iinfo(stitched.dtype).max - stitched


def materialize_stitched_images(
    experiment: str,
    microscope: str,
    raw_images_dir: Path,
    scope_csv: Path,
    mapping_csv: Path | None,
    output_root: Path,
    output_stitched_index_csv: Path,
    selected_wells: Iterable[str] | None,
    overwrite: bool,
    output_image_extension: str = "jpg",
    device_preference: str = "cuda",
    keyence_projection_method: str = "log",
    keyence_ff_filter_res_um: float = 3.0,
    done_flag: Path | None = None,
) -> pd.DataFrame:
    """Materialize stitched images for selected wells and emit stitched-image index."""
    scope_df = pd.read_csv(scope_csv)
    selected = _parse_selected_wells(selected_wells)
    if selected:
        scope_df = scope_df[scope_df["well_index"].astype(str).isin(selected)].copy()

    if scope_df.empty:
        raise ValueError(f"No scope rows available for experiment {experiment} after well filtering")

    if "channel_id" not in scope_df.columns:
        scope_df["channel_id"] = scope_df.get("channel", "BF").astype(str)

    scope_df = (
        scope_df.sort_values(["well_index", "channel_id", "time_int"])
        .drop_duplicates(subset=["experiment_id", "well_id", "well_index", "channel_id", "time_int"], keep="first")
        .copy()
    )

    stitched_root = output_root / experiment / "stitched_ff_images"
    data_root = output_root.parent

    yx1_nd2_files = sorted(raw_images_dir.glob("*.nd2")) if microscope == "YX1" else []
    yx1_source = yx1_nd2_files[0] if yx1_nd2_files else raw_images_dir
    keyence_lookup = _infer_keyence_stack_lookup(raw_images_dir) if microscope == "Keyence" else {}
    keyence_orientation = _keyence_orientation(experiment)
    keyence_master_params = _keyence_master_params_path(raw_images_dir, experiment) if microscope == "Keyence" else None
    keyence_device = _resolve_device(device_preference)
    yx1_device = _resolve_device(device_preference)
    yx1_series_map = _build_yx1_well_to_series_map(mapping_csv) if microscope == "YX1" else {}
    image_extension = _normalize_extension(output_image_extension)
    keyence_method = str(keyence_projection_method or "log").strip().lower()

    nd = None
    dask_arr = None
    channel_index = 0
    if microscope == "YX1" and yx1_nd2_files:
        nd = nd2.ND2File(yx1_nd2_files[0])
        dask_arr = nd.to_dask()
        channel_names = [ch.channel.name for ch in nd.frame_metadata(0).channels]
        channel_index = _select_yx1_channel_index(channel_names)

    rows = []
    try:
        for _, row in scope_df.iterrows():
            well_index = str(row["well_index"])
            channel_id = str(row.get("channel_id", row.get("channel", "BF")))
            well_id = str(row["well_id"])
            time_int = int(row["time_int"])
            frame_index = int(row.get("frame_index", time_int))

            image_id = row.get("image_id")
            if pd.isna(image_id) or not str(image_id):
                image_id = f"{well_id}_{channel_id}_t{frame_index:04d}"
            image_id = str(image_id)

            output_path = stitched_root / well_index / channel_id / f"{image_id}.{image_extension}"
            width = int(float(row.get("image_width_px", 512)))
            height = int(float(row.get("image_height_px", 512)))
            materialized_width = width
            materialized_height = height

            status = "placeholder"
            source_artifact: Path
            source_kind: str

            if microscope == "YX1" and nd is not None and dask_arr is not None:
                series_index = yx1_series_map.get(well_index)
                if series_index is not None:
                    image = _materialize_yx1_image(
                        nd=nd,
                        dask_arr=dask_arr,
                        time_int=time_int,
                        series_index=series_index,
                        channel_index=channel_index,
                        device=yx1_device,
                    )
                    materialized_height, materialized_width = image.shape[:2]
                    status = _write_image(output_path, image, overwrite=overwrite, image_extension=image_extension)
                else:
                    status = _write_placeholder_image(
                        output_path,
                        width,
                        height,
                        frame_index,
                        overwrite=overwrite,
                        image_extension=image_extension,
                    )
                source_artifact = yx1_source
                source_kind = "yx1_nd2_zstack"
            elif microscope == "Keyence":
                tile_stacks = keyence_lookup.get((well_index, time_int), {})
                if tile_stacks:
                    if keyence_method == "log":
                        mpp = float(row.get("micrometers_per_pixel", np.nan))
                        filter_size = _keyence_filter_size(
                            micrometers_per_pixel=mpp,
                            ff_filter_res_um=float(keyence_ff_filter_res_um),
                        )
                        tile_projections = _materialize_keyence_ff_tiles_with_log(
                            tile_stacks=tile_stacks,
                            device=keyence_device,
                            filter_size=filter_size,
                        )
                    else:
                        tile_projections = [
                            _materialize_keyence_tile_projection(tile_stacks[tile_id])
                            for tile_id in sorted(tile_stacks.keys())
                        ]
                    image = _stitch_keyence_tile_projections(
                        tile_projections=tile_projections,
                        orientation=keyence_orientation,
                        master_params_path=keyence_master_params,
                    )
                    materialized_height, materialized_width = image.shape[:2]
                    status = _write_image(output_path, image, overwrite=overwrite, image_extension=image_extension)
                    source_artifact = tile_stacks[sorted(tile_stacks.keys())[0]][0]
                    source_kind = (
                        "keyence_tiff_stitched_tiles_log"
                        if len(tile_stacks) > 1
                        else "keyence_tiff_single_tile_log"
                    )
                else:
                    status = _write_placeholder_image(
                        output_path,
                        width,
                        height,
                        frame_index,
                        overwrite=overwrite,
                        image_extension=image_extension,
                    )
                    source_artifact = raw_images_dir / _well_to_xy_dir_name(well_index)
                    source_kind = "keyence_tiff_zstack"
            else:
                status = _write_placeholder_image(
                    output_path,
                    width,
                    height,
                    frame_index,
                    overwrite=overwrite,
                    image_extension=image_extension,
                )
                source_artifact = raw_images_dir
                source_kind = "placeholder"

            stitched_rel = output_path.relative_to(data_root)
            rows.append(
                {
                    "experiment_id": experiment,
                    "microscope_id": microscope,
                    "well_id": well_id,
                    "well_index": well_index,
                    "channel_id": channel_id,
                    "time_int": time_int,
                    "frame_index": frame_index,
                    "image_id": image_id,
                    "stitched_image_path": str(stitched_rel),
                    "materialization_status": status,
                    "source_artifact_path": str(source_artifact),
                    "source_artifact_kind": source_kind,
                    "image_width_px": materialized_width,
                    "image_height_px": materialized_height,
                }
            )
    finally:
        if nd is not None:
            nd.close()

    stitched_index_df = pd.DataFrame(rows)
    output_stitched_index_csv.parent.mkdir(parents=True, exist_ok=True)
    stitched_index_df.to_csv(output_stitched_index_csv, index=False)

    if done_flag is not None:
        done_flag.parent.mkdir(parents=True, exist_ok=True)
        done_flag.write_text("done\n")

    return stitched_index_df
