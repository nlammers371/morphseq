from pathlib import Path
import sys

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

# Dependency simplification notes (comments only; no behavior change):
# - scikit-image: many usages (io, transform, exposure, filters, morphology) could be replaced by OpenCV (`cv2`),
#   imageio, and NumPy to slim the environment. Example: resize via `cv2.resize`, CLAHE via `cv2.createCLAHE`.
# - sklearn PCA: can be replaced by NumPy SVD/eig for principal axes to avoid scikit-learn in this module.
# - tqdm/process_map: can fallback to `concurrent.futures` with an optional tqdm progress wrapper if available.
# - warnings/scipy.ndimage: if SciPy is heavy for your env, basic binary hole-filling can be mimicked with OpenCV morphology.
# - Paths/CSV IO: consider `csv`/`pathlib` where pandas is only used for simple merges.

import os
import glob
import csv
import re
from datetime import datetime
from typing import Dict, Optional
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours
import skimage
import cv2
import pandas as pd
from src.functions.utilities import path_leaf
from src.functions.image_utils import crop_embryo_image, get_embryo_angle, process_masks
from skimage.morphology import disk, binary_closing, remove_small_objects
import warnings
import scipy
# from parfor import pmap
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
import skimage
from scipy.stats import truncnorm
import numpy as np
import skimage.io as io
import multiprocessing
from functools import partial
from tqdm.contrib.concurrent import process_map 
from skimage.transform import rescale, resize
from src.build.export_utils import trim_to_shape
from sklearn.decomposition import PCA
from src.build.qc_utils import compute_fraction_alive, compute_qc_flags, compute_speed
import warnings
from pathlib import Path
from typing import Sequence, List
from itertools import chain
import os
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask

def resolve_sandbox_embryo_mask(root: str | Path, date: str, well: str, time_int: int) -> Path:
    """Legacy mask resolver - DEPRECATED. Use resolve_sandbox_embryo_mask_from_csv instead.
    
    This function uses hardcoded patterns and may fail when multiple embryos exist per frame.
    """
    root = Path(root)
    base_override = os.environ.get("MORPHSEQ_SANDBOX_MASKS_DIR")
    base = Path(base_override) if base_override else (root / "segmentation_sandbox" / "data" / "exported_masks")
    mask_dir = base / str(date) / "masks"
    # Match the actual naming pattern: {date}_{well}_ch00_t{time:04}_masks_emnum_1.png
    stub = f"{date}_{well}_ch00_t{int(time_int):04}_masks_emnum_1.png"
    candidates = sorted(mask_dir.glob(stub))
    if not candidates:
        raise FileNotFoundError(f"Sandbox embryo mask not found for pattern: {mask_dir}/{stub}")
    return candidates[0]

def resolve_sandbox_embryo_mask_from_csv(root: str | Path, row) -> Path:
    """Resolve integer-labeled embryo mask path using CSV exported_mask_path.
    
    Uses the exact filename provided in the SAM2 CSV instead of pattern matching.
    This eliminates hardcoded pattern assumptions and works with multiple embryos per frame.
    
    Args:
        root: Project root directory
        row: CSV row with exported_mask_path column
        
    Returns:
        Path to the specific mask file for this embryo
        
    Raises:
        FileNotFoundError: If the mask file specified in CSV doesn't exist
    """
    root = Path(root)
    base_override = os.environ.get("MORPHSEQ_SANDBOX_MASKS_DIR")
    
    if not base_override:
        # Use data root SAM2 pipeline location instead of hardcoded segmentation_sandbox path
        base = root / "sam2_pipeline_files" / "exported_masks"
    else:
        base = Path(base_override)
    
    date = str(row["experiment_date"])
    mask_filename = row["exported_mask_path"]
    
    mask_path = base / date / "masks" / mask_filename
    
    if not mask_path.exists():
        raise FileNotFoundError(f"Sandbox embryo mask not found: {mask_path}")

    return mask_path


def validate_sam2_artifact_consistency(root: Path, exp_id: str) -> None:
    """
    Validate that mask_manifest and SAM2 JSON are consistent.

    This catches the issue where SAM2 pipeline stages ran at different times,
    causing the mask manifest to be out of sync with the SAM2 JSON source data.

    Checks:
    1. manifest.exports[image_id].embryo_count == len(sam2_json[...].embryos)
    2. manifest.exports[image_id].output_path exists on disk

    Args:
        root: Project root directory
        exp_id: Experiment ID (e.g., "20251121")

    Raises:
        RuntimeError: If artifacts are inconsistent, with actionable fix instructions
    """
    import json

    root = Path(root)

    # Locate SAM2 JSON
    sam2_json_path = root / "sam2_pipeline_files" / "segmentation" / f"grounded_sam_segmentations_{exp_id}.json"
    if not sam2_json_path.exists():
        # If SAM2 JSON doesn't exist, skip validation (SAM2 not run yet)
        return

    # Locate mask manifest
    manifest_path = root / "sam2_pipeline_files" / "exported_masks" / exp_id / f"mask_export_manifest_{exp_id}.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"❌ Mask manifest not found: {manifest_path}\n"
            f"SAM2 mask export (Stage 5) may not have completed.\n\n"
            f"FIX: Re-run SAM2 pipeline:\n"
            f"  python -m src.run_morphseq_pipeline.cli pipeline \\\n"
            f"    --experiments {exp_id} --action sam2 --force"
        )

    # Load artifacts
    with open(sam2_json_path, 'r') as f:
        sam2_data = json.load(f)

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Build embryo count map from SAM2 JSON
    # Structure: sam2_data["experiments"][exp]["videos"][video]["image_ids"][image]["embryos"]
    json_embryo_counts = {}
    experiments = sam2_data.get("experiments", {})
    for exp, exp_data in experiments.items():
        videos = exp_data.get("videos", {})
        for video_id, video_data in videos.items():
            image_ids = video_data.get("image_ids", {})
            for image_id, image_data in image_ids.items():
                embryos = image_data.get("embryos", {})
                json_embryo_counts[image_id] = len(embryos)

    # Compare with manifest
    manifest_exports = manifest.get("exports", {})
    mismatches = []

    for image_id, export_info in manifest_exports.items():
        manifest_count = export_info.get("embryo_count", 0)
        json_count = json_embryo_counts.get(image_id, 0)

        # Check 1: Embryo count mismatch
        if manifest_count != json_count:
            # Check if mask file exists to provide detailed diagnostic
            output_path = Path(export_info.get("output_path", ""))
            file_exists = output_path.exists() if output_path else False

            # Check for alternative emnum files
            if output_path and output_path.parent.exists():
                stem = output_path.stem.rsplit("_emnum_", 1)[0]
                alternatives = sorted(output_path.parent.glob(f"{stem}_emnum_*.png"))
            else:
                alternatives = []

            mismatches.append({
                "image_id": image_id,
                "sam2_count": json_count,
                "manifest_count": manifest_count,
                "expected_file": output_path.name if output_path else "unknown",
                "file_exists": file_exists,
                "alternatives": [a.name for a in alternatives]
            })
        else:
            # Check 2: Mask file existence (even if counts match)
            output_path = Path(export_info.get("output_path", ""))
            if output_path and not output_path.exists():
                mismatches.append({
                    "image_id": image_id,
                    "sam2_count": json_count,
                    "manifest_count": manifest_count,
                    "expected_file": output_path.name if output_path else "unknown",
                    "file_exists": False,
                    "alternatives": []
                })

    if mismatches:
        sample = mismatches[:5]
        msg = [
            f"❌ SAM2 ARTIFACT INCONSISTENCY DETECTED",
            "",
            f"Mask manifest is out of sync with SAM2 JSON for experiment {exp_id}.",
            f"Found {len(mismatches)} inconsistencies.",
            "",
            "Sample mismatches:"
        ]

        for m in sample:
            msg.append(f"  Image: {m['image_id']}")
            msg.append(f"    - SAM2 JSON: {m['sam2_count']} embryos")
            msg.append(f"    - Manifest:  {m['manifest_count']} embryos (emnum_{m['manifest_count']})")
            if m['sam2_count'] != m['manifest_count']:
                msg.append(f"    - Expected:  ...emnum_{m['sam2_count']}.png")
            if not m['file_exists']:
                msg.append(f"    - Status:    MISSING on disk")
            if m['alternatives']:
                msg.append(f"    - Found:     {m['alternatives']}")
            msg.append("")

        if len(mismatches) > 5:
            msg.append(f"  ... and {len(mismatches) - 5} more inconsistencies")
            msg.append("")

        msg.append("This happens when SAM2 JSON is updated but masks are not re-exported.")
        msg.append("")
        msg.append("FIX: Re-run SAM2 with --force to regenerate masks:")
        msg.append(f"  python -m src.run_morphseq_pipeline.cli pipeline \\")
        msg.append(f"    --experiments {exp_id} --action sam2 --force")

        raise RuntimeError("\n".join(msg))


def _extract_time_stub(row) -> str:
    """Return the acquisition time suffix (####) for a given metadata row."""

    def _from_value(value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        text = str(value).strip()
        if not text or text.lower() == 'nan':
            return ""
        if text[0].lower() == 't' and len(text) > 1:
            text = text[1:]
        if text.isdigit():
            return f"{int(text):04d}"
        return ""

    for key in ("time_key", "time_string", "time_int", "frame_index"):
        if key not in row:
            continue
        value = row[key]
        if key == "time_int" or key == "frame_index":
            try:
                return f"{int(value):04d}"
            except (TypeError, ValueError):
                continue
        stub = _from_value(value)
        if stub:
            return stub

    raise ValueError("Unable to derive time stub from row metadata")

def _load_build02_masks_for_row(root: Path, row, target_shape: tuple[int, int], is_sam2_pipeline: bool = False) -> dict:
    """Load Build02 masks (via/yolk/focus/bubble) for a row if available.

    Searches under `<root>/segmentation/*_<model>/<date>/*{well}_t####*`.
    Resizes masks (nearest-neighbor) to match `target_shape` when needed.
    Returns dict with present masks; missing keys omitted.
    """
    import skimage.io as io
    from skimage.transform import resize

    date = str(row.get("experiment_date"))
    well = row.get("well")
    time_stub = _extract_time_stub(row)
    stub = f"{well}_t{time_stub}"

    seg_root = Path(root) / "segmentation"
    if not seg_root.exists():
        return {}

    def _find_and_read(keyword: str):
        for p in seg_root.iterdir():
            if p.is_dir() and keyword in p.name:
                date_dir = p / date
                if date_dir.exists():
                    candidates = sorted(date_dir.glob(f"*{stub}*"))
                    if candidates:
                        arr_raw = io.imread(candidates[0])
                        if is_sam2_pipeline:
                            threshold = 127
                            arr = (arr_raw > threshold).astype(np.uint8)
                        else:
                            arr = (np.round(arr_raw / 255 * 2) - 1).astype(np.uint8)
                            arr = (arr > 0).astype(np.uint8)
                        if arr.shape != target_shape:
                            arr = resize(
                                arr.astype(float), target_shape, order=0, preserve_range=True, anti_aliasing=False
                            ).astype(np.uint8)
                        return arr
        return None

    out = {}
    for k in ("via", "yolk", "focus", "bubble"):
        m = _find_and_read(k)
        if m is not None:
            out[k] = m
    return out

# Suppress the specific warning from skimage
warnings.filterwarnings("ignore", message="Only one label was provided to `remove_small_objects`")

def estimate_image_background(root, embryo_metadata_df, bkg_seed=309, n_bkg_samples=100):

    np.random.seed(bkg_seed)
    bkg_sample_indices = np.random.choice(range(embryo_metadata_df.shape[0]), n_bkg_samples, replace=True)
    bkg_pixel_list = []

    for r in tqdm(range(len(bkg_sample_indices)), "Estimating background..."):
        sample_i = bkg_sample_indices[r]
        row = embryo_metadata_df.iloc[sample_i].copy()

        # set path to segmentation data
        ff_image_path = os.path.join(root, 'built_image_data', 'stitched_FF_images', '')
        segmentation_path = os.path.join(root, 'segmentation', '')
        segmentation_model_path = os.path.join(root, 'segmentation', 'segmentation_models', '')

         # get list of up-to-date models
        seg_mdl_list_raw = glob.glob(segmentation_model_path + "*")
        seg_mdl_list = [s for s in seg_mdl_list_raw if ~os.path.isdir(s)]
        emb_mdl_name = [path_leaf(m) for m in seg_mdl_list if "mask" in m][0]
        via_mdl_name = [path_leaf(m) for m in seg_mdl_list if "via" in m][0]

        # generate path and image name
        seg_dirs_raw = glob.glob(segmentation_path + "*")
        seg_dirs = [s for s in seg_dirs_raw if os.path.isdir(s)]

        emb_path = [m for m in seg_dirs if emb_mdl_name in m][0]
        via_path = [m for m in seg_dirs if via_mdl_name in m][0]

        well = row["well"]
        time_stub = _extract_time_stub(row)
        date = str(row["experiment_date"])

        ############
        # Load masks from segmentation
        ############
        stub_name = well + f"_t{time_stub}*"
        im_emb_path = glob.glob(os.path.join(emb_path, date, stub_name))[0]
        im_via_path = glob.glob(os.path.join(via_path, date, stub_name))[0]

        # load main embryo mask
        # im_emb_path = os.path.join(emb_path, date, lb_name)
        im_mask = io.imread(im_emb_path)
        im_mask = np.round(im_mask / 255 * 2 - 1).astype(int)

        im_via = io.imread(im_via_path)
        im_via = np.round(im_via / 255 * 2 - 1).astype(int)

        im_bkg = np.ones(im_mask.shape, dtype="uint8")
        im_bkg[np.where(im_mask == 1)] = 0
        im_bkg[np.where(im_via == 1)] = 0

        # load image
        im_ff_path = glob.glob(os.path.join(ff_image_path, date, stub_name))[0] #os.path.join(ff_image_path, date, im_name)
        im_ff = io.imread(im_ff_path)
        if im_ff.shape[0] < im_ff.shape[1]:
            im_ff = im_ff.transpose(1, 0)
        if im_ff.dtype != "uint8":
            im_ff = skimage.util.img_as_ubyte(im_ff)

        # extract background pixels
        bkg_pixels = im_ff[np.where(im_bkg == 1)].tolist()

        bkg_pixel_list += bkg_pixels

    # get mean and standard deviation
    px_mean = np.mean(bkg_pixel_list)
    px_std = np.std(bkg_pixel_list)

    return px_mean, px_std


def export_embryo_snips(r: int,
                        root: str | Path,
                        stats_df: pd.DataFrame,
                        dl_rad_um: int,
                        outscale: float,
                        outshape: List,
                        px_mean: float, px_std: float):
    """
    Refactored function to use SAM2 metadata and disable unavailable QC checks.

    This function is refactored to use pre-computed SAM2 metadata. It loads
    the embryo mask from the path specified in the SAM2 CSV. Since yolk masks
    are not available in the SAM2 pipeline output, a dummy yolk mask is used.
    """

    root = Path(root)
    row = stats_df.iloc[r].copy()

    # Extract key variables from row data
    well = row.get("well")
    time_stub = _extract_time_stub(row)
    date = str(row.get("experiment_date"))

    # --- Load embryo mask from segmentation_sandbox (integer-labeled) ---
    try:
        mask_path = resolve_sandbox_embryo_mask_from_csv(root, row)
        im_mask_int = io.imread(mask_path)
        lbi = int(row["region_label"])  # assumes present; MVP scope
        im_mask = ((im_mask_int == lbi) * 255).astype(np.uint8)
    except Exception as e:
        raise

    # --- Clean mask using 5-step pipeline ---
    try:
        im_mask, cleaning_stats = clean_embryo_mask(im_mask, verbose=False)
    except Exception as e:
        # Continue with original (uncleaned) mask on failure
        pass

    # Load yolk from Build02 segmentation (keep non-embryo masks unchanged)
    im_yolk = None
    seg_root = root / 'segmentation'
    if seg_root.exists():
        yolk_dirs = [p for p in seg_root.glob("*") if p.is_dir() and "yolk" in p.name]
        if yolk_dirs:
            stub = f"{well}_t{time_stub}*"
            candidates = sorted((yolk_dirs[0] / date).glob(stub))
            if candidates:
                im_yolk_raw = io.imread(candidates[0])
                im_yolk = resize(
                    (im_yolk_raw > (127 if im_yolk_raw.max() >= 255 else 0)).astype(np.uint8),
                    im_mask.shape,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(np.uint8)
    if im_yolk is None:
        warnings.warn("Legacy yolk mask not found; proceeding with empty yolk mask for 2D snips.")
        im_yolk = np.zeros_like(im_mask)

    # --- Load Full-Frame Image ---
    # Prefer stitched FF images by default (least compressed), then raw path if needed
    im_ff = None
    ff_image_path = root / 'built_image_data' / 'stitched_FF_images'

    # Try both full format and legacy format under stitched_FF_images
    full_stub = f"{row['image_id']}*"
    ff_image_paths = sorted((ff_image_path / date).glob(full_stub))

    if not ff_image_paths:
        # Try legacy format: extract well and time from image_id using regex
        # image_id format: "20250612_30hpf_ctrl_atf6_C12_ch00_t0000"
        # legacy format: "C12_t0000_stitch.jpg"
        import re
        image_id = row['image_id']
        match = re.search(r'_([A-H]\d{2})_.*_(t\d{4})$', image_id)
        if match:
            well_part, time_part = match.groups()
            legacy_stub = f"{well_part}_{time_part}*"
            ff_image_paths = sorted((ff_image_path / date).glob(legacy_stub))

    if ff_image_paths:
        im_ff = io.imread(ff_image_paths[0])
    else:
        # Fallback to raw stitched image path from metadata if available
        if 'raw_stitch_image_path' in row and row['raw_stitch_image_path']:
            raw_stitch_path = Path(row['raw_stitch_image_path'])
            if raw_stitch_path.exists():
                im_ff = io.imread(raw_stitch_path)
        if im_ff is None:
            warnings.warn(
                f"FF image not found under {ff_image_path / date} for stub '{full_stub}' and legacy pattern; "
                f"no raw_stitch_image_path usable.",
                stacklevel=2,
            )
            return True

    # --- Continue with legacy processing ---
    if 'Height (um)' in row and 'Height (px)' in row and row['Height (px)'] > 0:
        px_dim_raw = row["Height (um)"] / row["Height (px)"]
    else:
        px_dim_raw = 1.0

    # For single-embryo binary mask, set region_label=1 so downstream selection stays unchanged
    row_for_mask = row.copy()
    row_for_mask["region_label"] = 1
    im_mask_ft, im_yolk = process_masks(im_mask, im_yolk, row_for_mask)

    ff_transposed = False
    if im_ff.shape[0] < im_ff.shape[1]:
        # Keep FF image and masks in the same orientation before cropping.
        # 20260122 data are landscape; transposing only the FF image causes mask/image misalignment.
        im_ff = im_ff.transpose(1, 0)
        ff_transposed = True

    if im_ff.dtype != "uint8":
        im_ff_scaled = skimage.exposure.rescale_intensity(im_ff, in_range='image', out_range=(0, 255))
        im_ff = im_ff_scaled.astype(np.uint8)

    if ff_transposed:
        im_mask_ft = im_mask_ft.transpose(1, 0)
        im_yolk = im_yolk.transpose(1, 0)

    im_ff_rs = rescale(im_ff, (px_dim_raw / outscale, px_dim_raw / outscale), order=1, preserve_range=True)
    mask_emb_rs = resize(im_mask_ft.astype(float), im_ff_rs.shape, order=1)
    mask_yolk_rs = resize(im_yolk.astype(float), im_ff_rs.shape, order=1)

    angle_to_use = get_embryo_angle((mask_emb_rs > 0.5).astype(np.uint8),(mask_yolk_rs > 0.5).astype(np.uint8))

    im_ff_rotated = rotate_image(im_ff_rs, np.rad2deg(angle_to_use))
    emb_mask_rotated = rotate_image(mask_emb_rs, np.rad2deg(angle_to_use))
    im_yolk_rotated = rotate_image(mask_yolk_rs, np.rad2deg(angle_to_use))

    im_cropped, emb_mask_cropped, yolk_mask_cropped, out_of_frame_flag = crop_embryo_image(
        im_ff_rotated, emb_mask_rotated, im_yolk_rotated, outshape=outshape, return_metrics=True
    )

    emb_mask_cropped2 = scipy.ndimage.binary_fill_holes(emb_mask_cropped > 0.5).astype(np.uint8)
    yolk_mask_cropped = scipy.ndimage.binary_fill_holes(yolk_mask_cropped > 0.5).astype(np.uint8)

    noise_array_raw = np.reshape(truncnorm.rvs(-px_mean/px_std, 4, size=outshape[0]*outshape[1]), outshape)
    noise_array = noise_array_raw*px_std + px_mean
    noise_array[np.where(noise_array < 0)] = 0

    im_cropped = skimage.exposure.equalize_adapthist(im_cropped)*255
    mask_cropped_gauss = skimage.filters.gaussian(emb_mask_cropped2.astype(float), sigma=dl_rad_um / outscale)
    im_cropped_gauss = np.multiply(im_cropped.astype(float), mask_cropped_gauss) + np.multiply(noise_array, 1-mask_cropped_gauss)

    # out_of_frame_flag now computed by crop_embryo_image() based on mask area retention
    # Flags embryos where <98% of mask area is retained after cropping
   
    im_name = row["snip_id"]
    exp_date = str(row["experiment_date"])
    im_snip_dir = root / 'training_data' / 'bf_embryo_snips'
    mask_snip_dir = root / 'training_data' / 'bf_embryo_masks'
    
    ff_dir = im_snip_dir / exp_date
    ff_save_path = ff_dir / f"{im_name}.jpg"
    if not ff_dir.is_dir():
        ff_dir.mkdir(parents=True, exist_ok=True)

    ff_dir_uc = (im_snip_dir.parent / (im_snip_dir.name + "_uncropped")) / exp_date
    ff_save_path_uc = ff_dir_uc / f"{im_name}.jpg"
    if not ff_dir_uc.is_dir():
        ff_dir_uc.mkdir(parents=True, exist_ok=True)

    io.imsave(ff_save_path, im_cropped_gauss.astype(np.uint8), check_contrast=False)
    io.imsave(ff_save_path_uc, im_cropped.astype(np.uint8), check_contrast=False)
    io.imsave(mask_snip_dir / f"emb_{im_name}.jpg", emb_mask_cropped2, check_contrast=False)
    io.imsave(mask_snip_dir / f"yolk_{im_name}.jpg", yolk_mask_cropped, check_contrast=False)

    return out_of_frame_flag



def rotate_image(mat, angle):
    """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
    width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def make_well_names():
    row_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
    well_name_list = []
    for r in row_list:
        well_names = [r + f'{c+1:02}' for c in range(12)]
        well_name_list += well_names

    return well_name_list


def get_unprocessed_metadata(
    master_df: pd.DataFrame,
    existing_meta_path: str | Path,
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Return only the rows in master_df whose (well, experiment_date, time_int)
    are not in the existing_meta_path CSV (if present).  Drops duplicates on
    the three key columns before doing the comparison.
    """
    key_cols = ["well", "experiment_date", "time_int"]
    # 1) dedupe the master list
    df = master_df.drop_duplicates(subset=key_cols).copy()
    # 2) if overwriting or no existing file, we're done
    existing = Path(existing_meta_path)
    if overwrite or not existing.exists():
        return df

    # 3) read and dedupe the done list
    done = (
        pd.read_csv(existing, usecols=key_cols)
          .drop_duplicates(subset=key_cols)
    )
    # 4) merge with indicator
    merged = df.merge(
        done, on=key_cols, how="left", indicator=True
    )
    # 5) keep only those not in 'done'
    new_only = merged.loc[merged["_merge"] == "left_only", df.columns]
    return new_only.reset_index(drop=True)


# ─── helper: sample one JPG per date, warn & drop missing ────────────────────
def sample_experiment_shapes(exp_dir: Path):
    # shapes, missing = {}, []
    jpg = next(exp_dir.glob("*.jpg"), None)
    if jpg is None:
        raise Exception(f"FF images not found for {exp_dir.name}")
    else:
        shape = io.imread(jpg).shape[:2]  # (H, W)
        
    return shape

def process_mask_images(image_path):

    # load label image
    im = io.imread(image_path)
    im = np.asarray(im)
    if im.max() > 1 and im.max() < 255:
        # integer-labeled mask from SAM2; treat any non-zero as embryo for this helper
        im_mask = (im > 0).astype(np.uint8)
    else:
        # legacy binary {0,255} or {0,1}
        im_mask = (im > 127).astype(np.uint8) if im.max() >= 255 else (im > 0).astype(np.uint8)

    # load viability image
    seg_path = os.path.dirname(os.path.dirname(os.path.dirname(image_path)))
    date_dir = os.path.basename(os.path.dirname(image_path))
    im_stub = os.path.basename(image_path)[:9]
    via_dirs = glob.glob(os.path.join(seg_path, "via_*"))
    if via_dirs:
        via_path_list = glob.glob(os.path.join(via_dirs[0], date_dir, im_stub + "*"))
        if via_path_list:
            im_via = io.imread(via_path_list[0])
            im_via = (im_via > 127).astype(np.uint8) if im_via.max() >= 255 else (im_via > 0).astype(np.uint8)
        else:
            im_via = np.zeros_like(im_mask)
    else:
        im_via = np.zeros_like(im_mask)
    
    # make a combined mask
    cb_mask = np.ones_like(im_mask)
    cb_mask[np.where(im_mask == 1)] = 1  # alive
    cb_mask[np.where((im_mask == 1) & (im_via == 1))] = 2  # dead

    return im_mask, cb_mask

# ─── helper: find mask paths & valid row indices ─────────────────────────────
# def get_mask_paths_from_diff_old(df_diff, emb_dir, strict=False):
#     emb_dir = Path(emb_dir)
#     paths, idxs = [], []
#     for i, row in df_diff.iterrows():
#         folder = emb_dir / str(row["experiment_date"])
#         stub   = f"{row['well']}_t{int(row['time_int']):04}"
#         files  = list(folder.glob(f"{stub}*"))
#         if not files:
#             msg = f"segment_wells: no mask for {stub} in {folder}"
#             if strict:
#                 raise FileNotFoundError(msg)
#             warnings.warn(msg, stacklevel=2)
#             continue
#         paths.append(str(files[0]))
#         idxs.append(i)
#     return paths, idxs


def get_mask_paths_from_diff(df_diff, emb_dir, strict=False):
    # unprocessed_date_index = np.unique(df_diff["experiment_date"]).astype(str)

    well_id_list = df_diff["well"].values
    well_date_list = df_diff["experiment_date"].values.astype(str)
    well_time_list = df_diff["time_int"].values

    dates_u = np.unique(well_date_list)
    experiment_list = [emb_dir / d for d in dates_u]

    images_to_process = []
    valid_indices = []
    for _, experiment_path in enumerate(experiment_list):
        ename = path_leaf(experiment_path)
        date_indices = np.where(well_date_list == ename)[0]

        # get channel info
        image_list_full = sorted(glob.glob(os.path.join(experiment_path, "*.jpg")))

        if len(image_list_full) > 0:
            im_name_test = path_leaf(image_list_full[0])
            image_suffix = im_name_test[9:]

            # get list of image prefixes
            im_names = [os.path.join(experiment_path, well_id_list[i] + f"_t{well_time_list[i]:04}" + image_suffix) for i in date_indices]
            
            full_set = set(image_list_full)
            # find the (df_diff) indices whose im_names actually exist on disk
            valid = [(idx, name) 
                    for idx, name in zip(date_indices, im_names) 
                    if name in full_set]
            # unpack into two parallel lists
            vi, itp = map(list, zip(*valid)) if valid else ([], [])

            valid_indices += vi
            images_to_process += itp

    return images_to_process, np.asarray(valid_indices)


# DELETED: count_embryo_regions() function
# This legacy function has been replaced by SAM2 metadata bridge approach.
# SAM2 provides pre-computed embryo areas, centroids, and bounding boxes,
# eliminating the need for regionprops calculations.
# 
# Original function removed as part of refactor-003 Phase 2 implementation.
# See: docs/refactors/refactor-003-segmentation-pipeline-integration-prd.md
    pass  # Function body removed - replaced by SAM2 bridge approach



# DELETED: do_embryo_tracking() function  
# This legacy function has been replaced by SAM2 metadata bridge approach.
# SAM2 provides inherent temporal tracking with stable embryo IDs across frames,
# eliminating the need for Hungarian algorithm tracking.
# 
# Original function removed as part of refactor-003 Phase 2 implementation.
# See: docs/refactors/refactor-003-segmentation-pipeline-integration-prd.md
def do_embryo_tracking_DELETED(
    well_id: str,
    df_update: pd.DataFrame
) -> pd.DataFrame:
    """DELETED - Function replaced by SAM2 bridge approach."""
    # slice out just this well’s rows, in time‐order
    sub = df_update[df_update["well_id"] == well_id].copy().reset_index(drop=True)
    n_obs = sub["n_embryos_observed"].astype(int).values
    max_n = n_obs.max()

    # if zero embryos ever seen, nothing to emit
    if max_n == 0:
        return pd.DataFrame([], columns=list(sub.columns) + 
                            ["xpos","ypos","fraction_alive","region_label","embryo_id"])

    # the columns we carry forward
    temp_cols = list(sub.columns)

    # simple‐case: exactly one embryo in every frame => no assignment needed
    if max_n == 1:
        out = sub[temp_cols].copy()
        out["xpos"]           = sub["e0_x"].values
        out["ypos"]           = sub["e0_y"].values
        out["fraction_alive"] = sub["e0_frac_alive"].values
        out["region_label"]   = sub["e0_label"].values
        out["embryo_id"]      = well_id + "_e00"
        return out

    # --- otherwise multiple embryos may appear/disappear, we do a linear‐assignment over time ---
    T = len(sub)
    # preallocate assignment array: (T × max_n)
    ids = np.full((T, max_n), np.nan)
    # initialize at first time‐point that saw any embryos
    first = np.argmax(n_obs > 0)
    ids[first, : n_obs[first]] = np.arange(n_obs[first], dtype=float)

    # keep track of last positions so we can compute cost‐matrix
    last_pos = np.zeros((max_n, 2), dtype=float)
    for e in range(n_obs[first]):
        last_pos[e, 0] = sub.loc[first, f"e{e}_x"]
        last_pos[e, 1] = sub.loc[first, f"e{e}_y"]

    # step through subsequent frames
    for t in range(first + 1, T):
        k = n_obs[t]
        if k == 0:
            # carry forward previous ids but no new detections
            continue
        # build current positions
        curr = np.vstack([
            [sub.loc[t, f"e{j}_x"], sub.loc[t, f"e{j}_y"]]
            for j in range(k)
        ])
        # cost‐matrix between last_pos and curr
        D = pairwise_distances(last_pos, curr)
        # assign old→new
        row_ind, col_ind = linear_sum_assignment(D)
        # record assignments
        ids[t, row_ind] = col_ind
        # update last_pos for those that moved
        for r, c in zip(row_ind, col_ind):
            last_pos[r, :] = curr[c]

    # build output rows by walking each track separately
    out_rows: List[pd.DataFrame] = []
    for track in range(max_n):
        # select time‐points where track was assigned
        times, cols = np.where(~np.isnan(ids[:, track][:, None]))
        if len(times) == 0:
            continue
        subtrk = sub.iloc[times].copy().reset_index(drop=True)
        subtrk["xpos"]           = [subtrk.loc[i, f"e{int(ids[times[i], track])}_x"] for i in range(len(times))]
        subtrk["ypos"]           = [subtrk.loc[i, f"e{int(ids[times[i], track])}_y"] for i in range(len(times))]
        subtrk["fraction_alive"] = [subtrk.loc[i, f"e{int(ids[times[i], track])}_frac_alive"] for i in range(len(times))]
        subtrk["region_label"]   = [subtrk.loc[i, f"e{int(ids[times[i], track])}_label"] for i in range(len(times))]
        subtrk["embryo_id"]      = well_id + f"_e{track:02}"
        out_rows.append(subtrk[temp_cols + 
            ["xpos","ypos","fraction_alive","region_label","embryo_id"]
        ])

    if not out_rows:
        # nothing survived—return empty but typed
        return pd.DataFrame([], columns=temp_cols + 
                            ["xpos","ypos","fraction_alive","region_label","embryo_id"])
    
    return pd.concat(out_rows, ignore_index=True)


def get_embryo_stats(index: int,
                     root: str | Path,
                     embryo_metadata_df: pd.DataFrame,
                     qc_scale_um: int,
                     ld_rat_thresh: float):
    """
    Refactored function to use SAM2 metadata and disable unavailable QC checks.
    
    This function is refactored as per refactor-003-prd to use pre-computed
    SAM2 metadata. It loads the embryo mask from the path specified in the
    SAM2 CSV. QC checks depending on bubble, focus, and yolk masks are
    disabled as these masks are not available in the SAM2 pipeline output.
    """
    
    root = Path(root)
    row = embryo_metadata_df.loc[index].copy()

    # --- Load embryo mask from segmentation_sandbox (integer-labeled) ---
    mask_path = resolve_sandbox_embryo_mask_from_csv(root, row)
    if not Path(mask_path).exists():
        warnings.warn(f"Sandbox mask not found: {mask_path}", stacklevel=2)
        # Set frame_flag=True to mark this row as unusable due to missing mask
        row.loc["frame_flag"] = True
        for c in ["dead_flag","no_yolk_flag","focus_flag","bubble_flag"]:
            row.loc[c] = False
        return pd.DataFrame(row).transpose()

    im_mask_int = io.imread(mask_path)
    lbi = int(row["region_label"])  # assumes present; MVP scope
    im_mask_lb = ((im_mask_int == lbi) * 255).astype(np.uint8)

    # --- Perform calculations and QC checks ---
    # Calculate actual pixel dimension from CSV metadata
    # Use Height dimensions; Width should be equivalent for square pixels
    px_dim = row["Height (um)"] / row["Height (px)"]

    # Surface area using area_px if provided, else count mask pixels
    row.loc["surface_area_um"] = row.get("area_px", float((im_mask_lb > 0).sum())) * (px_dim ** 2)

    # Geometry (length/width) via PCA on embryo pixels
    yy, xx = np.indices(im_mask_lb.shape)
    mask_coords = np.c_[xx[im_mask_lb == 1], yy[im_mask_lb == 1]]
    if mask_coords.shape[0] > 1:
        pca = PCA(n_components=2)
        mask_coords_rot = pca.fit_transform(mask_coords)
        row.loc["length_um"], row.loc["width_um"] = (
            (np.max(mask_coords_rot, axis=0) - np.min(mask_coords_rot, axis=0)) * px_dim
        )
    else:
        row.loc["length_um"], row.loc["width_um"] = 0.0, 0.0

    # Load Build02 auxiliary masks (best-effort) and compute fraction_alive + QC flags
    well = row.get("well")
    
    # Detect if we're using SAM2 pipeline (check if we loaded SAM2 mask)
    is_sam2_pipeline = 'exported_mask_path' in row and pd.notna(row.get('exported_mask_path'))
    # print(f"DEBUG: Detected pipeline type: {'SAM2' if is_sam2_pipeline else 'Legacy Build02'}")
    
    aux = _load_build02_masks_for_row(Path(root), row, target_shape=im_mask_lb.shape, is_sam2_pipeline=is_sam2_pipeline)
    # print(f"DEBUG: aux masks loaded: {list(aux.keys())}")
    
    via_mask = aux.get("via")
    # if via_mask is not None:
    #     # print(f"DEBUG: Via mask found - shape: {via_mask.shape}, unique values: {np.unique(via_mask)}, sum: {via_mask.sum()}")
    # else:
    #     # print(f"DEBUG: Via mask is None")
    
    emb_mask_binary = (im_mask_lb > 0).astype(np.uint8)
    # print(f"DEBUG: Embryo mask - shape: {emb_mask_binary.shape}, unique values: {np.unique(emb_mask_binary)}, sum: {emb_mask_binary.sum()}")
    
    frac_alive = compute_fraction_alive(emb_mask_binary, via_mask)
    # print(f"DEBUG: Raw fraction_alive result: {frac_alive} (type: {type(frac_alive)})")
    
    row.loc["fraction_alive"] = frac_alive
    # print(f"DEBUG: ld_rat_thresh value: {ld_rat_thresh}")
    
    if np.isfinite(frac_alive):
        comparison_result = frac_alive < ld_rat_thresh
        # print(f"DEBUG: frac_alive < ld_rat_thresh: {frac_alive} < {ld_rat_thresh} = {comparison_result}")
        row.loc["dead_flag"] = bool(comparison_result)
        # print(f"DEBUG: dead_flag set to: {bool(comparison_result)}")
    else:
        # print(f"DEBUG: frac_alive is not finite: {frac_alive}")
        row.loc["dead_flag"] = False
        # print(f"DEBUG: dead_flag set to False (non-finite frac_alive)")
    
    # print(f"DEBUG: Final values - fraction_alive: {row.loc['fraction_alive']}, dead_flag: {row.loc['dead_flag']}")
    # print(f"DEBUG: --- End processing well {well} ---")

    flags = compute_qc_flags(
        (im_mask_lb > 0).astype(np.uint8),
        px_dim_um=px_dim,
        qc_scale_um=qc_scale_um,
        yolk_mask=aux.get("yolk"),
        focus_mask=aux.get("focus"),
        bubble_mask=aux.get("bubble"),
    )
    for k, v in flags.items():
        row.loc[k] = v

    # Speed (µm/s) if previous row has time/position
    prev_xy = prev_t = None
    if index > 0:
        try:
            prev_xy = (
                float(embryo_metadata_df.loc[index - 1, "xpos"]),
                float(embryo_metadata_df.loc[index - 1, "ypos"]),
            )
            prev_t = float(embryo_metadata_df.loc[index - 1, "Time Rel (s)"])
        except Exception:
            prev_xy, prev_t = None, None
    curr_xy = (float(row.get("xpos", np.nan)), float(row.get("ypos", np.nan)))
    curr_t = float(row.get("Time Rel (s)", np.nan)) if np.isfinite(row.get("Time Rel (s)", np.nan)) else None
    row.loc["speed"] = compute_speed(prev_xy, prev_t, curr_xy, curr_t, px_dim)

    row_out = pd.DataFrame(row).transpose()
    
    return row_out

####################
# Main process function 2
####################

def _merge_with_build01_metadata(sam2_df: pd.DataFrame, build01_metadata: pd.DataFrame, exp_name: str) -> pd.DataFrame:
    """
    Merge SAM2 dataframe with fresh Build01 metadata to update well-level information.

    This function automatically merges ALL columns from Build01 metadata (except merge keys),
    allowing Build03 to pick up metadata changes without re-running SAM2 segmentation.

    Args:
        sam2_df: DataFrame from SAM2 CSV with mask/tracking data
        build01_metadata: Fresh metadata from Build01
        exp_name: Experiment name for logging

    Returns:
        DataFrame with updated metadata from Build01
    """
    print(f"🔄 Merging SAM2 data with fresh Build01 metadata for {exp_name}...")

    # SAM2 CSV has 'well' and 'time_int' columns (from export_sam2_metadata_to_csv.py)
    # Build01 has 'well' or 'well_id' and 'time_int'
    # We merge on: well + time_int (matching the original SAM2 export logic)

    # Ensure SAM2 has 'well' column
    if 'well' not in sam2_df.columns:
        print("⚠️  Warning: SAM2 data missing 'well' column - cannot merge")
        return sam2_df

    # Ensure Build01 has 'well' column (might be 'well_id' instead)
    if 'well' not in build01_metadata.columns:
        if 'well_id' in build01_metadata.columns:
            build01_metadata = build01_metadata.copy()
            build01_metadata['well'] = build01_metadata['well_id']
        else:
            print("⚠️  Warning: Build01 metadata missing 'well' or 'well_id' column - cannot merge")
            return sam2_df

    # Check for time_int in both
    if 'time_int' not in sam2_df.columns or 'time_int' not in build01_metadata.columns:
        print(f"⚠️  Warning: Missing 'time_int' column (SAM2: {'time_int' in sam2_df.columns}, Build01: {'time_int' in build01_metadata.columns})")
        return sam2_df

    # Get ALL columns from Build01 metadata EXCEPT merge keys (automatically includes everything: pair, notes, etc.)
    merge_keys = ['well', 'time_int']
    all_build01_cols = [col for col in build01_metadata.columns if col not in merge_keys]
    merge_cols = merge_keys + all_build01_cols
    build01_merge = build01_metadata[merge_cols].copy()

    # Perform left merge: keep all SAM2 rows, update metadata where available
    result_df = sam2_df.merge(
        build01_merge,
        on=merge_keys,
        how='left',
        suffixes=('_sam2', '_build01')
    )

    # For each metadata column, prefer Build01 value if available (handles duplicates)
    updated_cols = []
    for col in all_build01_cols:
        build01_col = f"{col}_build01"
        sam2_col = f"{col}_sam2"

        if build01_col in result_df.columns:
            # Update with Build01 values, keeping SAM2 values where Build01 is missing
            if sam2_col in result_df.columns:
                result_df[col] = result_df[build01_col].fillna(result_df[sam2_col])
                result_df.drop(columns=[sam2_col, build01_col], inplace=True)
            else:
                result_df[col] = result_df[build01_col]
                result_df.drop(columns=[build01_col], inplace=True)
            updated_cols.append(col)

    if updated_cols:
        print(f"✅ Merged {len(updated_cols)} metadata columns from Build01: {', '.join(updated_cols[:10])}{'...' if len(updated_cols) > 10 else ''}")
    else:
        print("ℹ️  No duplicate columns to merge (SAM2 already has all Build01 metadata)")

    return result_df


def segment_wells_sam2_csv(
    root: str | Path,
    exp_name: str,
    sam2_csv_path: str | Path = None,
    min_sa_um: float = 250_000,
    max_sa_um: float = 2_000_000,
    par_flag: bool = False,
    overwrite_well_stats: bool = False,
):
    """
    SAM2-based well segmentation using pre-computed metadata bridge CSV.
    
    This function replaces the legacy segment_wells() function that used
    regionprops calculations and Hungarian algorithm tracking. Instead,
    it loads pre-computed embryo metadata from SAM2 bridge CSV.
    
    Args:
        root: Project root directory
        exp_name: Experiment name (e.g., "20240418")
        sam2_csv_path: Path to SAM2 metadata CSV (if None, looks for sam2_metadata_{exp_name}.csv)
        min_sa_um: Minimum surface area filter (deprecated - handled by SAM2)
        max_sa_um: Maximum surface area filter (deprecated - handled by SAM2) 
        par_flag: Parallel processing flag (deprecated)
        overwrite_well_stats: Overwrite existing stats flag
        
    Returns:
        DataFrame with embryo tracking data compatible with legacy format
    """
    
    root = Path(root)
    
    # Determine CSV path
    if sam2_csv_path is None:
        sam2_csv_path = root / f"sam2_metadata_{exp_name}.csv"
    else:
        sam2_csv_path = Path(sam2_csv_path)
    
    if not sam2_csv_path.exists():
        raise FileNotFoundError(f"SAM2 metadata CSV not found: {sam2_csv_path}")
    
    print(f"🔄 Loading SAM2 metadata from {sam2_csv_path}")
    
    # Load SAM2 CSV with pre-computed metadata
    sam2_df = pd.read_csv(sam2_csv_path, dtype={'experiment_id': str})
    
    # Filter by experiment if needed
    if exp_name not in sam2_df['experiment_id'].values:
        raise ValueError(f"Experiment {exp_name} not found in SAM2 CSV")
    
    exp_df = sam2_df[sam2_df['experiment_id'] == exp_name].copy()

    print(f"📊 SAM2 data loaded: {len(exp_df)} snips from experiment {exp_name}")

    # Always merge with Build01 metadata to ensure all columns propagate through
    build01_metadata_path = root / "metadata" / "built_metadata_files" / f"{exp_name}_metadata.csv"

    if build01_metadata_path.exists():
        # Always merge with Build01 metadata to ensure ALL columns (pair, notes, etc.) propagate through
        print(f"🔄 Merging with Build01 metadata to ensure all columns propagate...")
        build01_metadata = pd.read_csv(build01_metadata_path, low_memory=False)
        exp_df = _merge_with_build01_metadata(exp_df, build01_metadata, exp_name)
    else:
        print(f"ℹ️  No Build01 metadata found at {build01_metadata_path} - using SAM2 CSV metadata as-is")

    has_time_string = 'time_string' in exp_df.columns
    has_time_int = 'time_int' in exp_df.columns
    if not (has_time_string or has_time_int):
        raise ValueError(
            "SAM2 CSV is missing both 'time_string' and 'time_int'."
            " Cannot continue without acquisition indices."
        )

    def _normalize_time_string(value: pd.Series) -> pd.Series:
        def _convert(x):
            if pd.isna(x):
                return pd.NA
            s = str(x).strip()
            if not s or s.lower() == 'nan':
                return pd.NA
            if s[0].lower() == 't':
                s = s[1:]
            try:
                return f"T{int(s):04d}"
            except ValueError:
                return pd.NA
        return value.apply(_convert)

    if has_time_string:
        exp_df['time_string'] = _normalize_time_string(exp_df['time_string'])
    else:
        exp_df['time_string'] = pd.Series([pd.NA] * len(exp_df), dtype="string")

    if has_time_int:
        exp_df['time_int'] = pd.to_numeric(exp_df['time_int'], errors='coerce').astype('Int64')
    else:
        exp_df['time_int'] = pd.Series([pd.NA] * len(exp_df), dtype='Int64')

    missing_str_mask = exp_df['time_string'].isna() & exp_df['time_int'].notna()
    exp_df.loc[missing_str_mask, 'time_string'] = exp_df.loc[missing_str_mask, 'time_int'].astype(int).apply(lambda x: f"T{x:04d}")

    missing_int_mask = exp_df['time_int'].isna() & exp_df['time_string'].notna()
    exp_df.loc[missing_int_mask, 'time_int'] = exp_df.loc[missing_int_mask, 'time_string'].str[1:].astype(int)

    if exp_df['time_string'].isna().any() or exp_df['time_int'].isna().any():
        unresolved = exp_df.loc[exp_df['time_string'].isna() | exp_df['time_int'].isna(), ['image_id', 'frame_index']]
        raise ValueError(
            "Unable to resolve acquisition indices for the following rows:\n"
            + unresolved.head().to_string(index=False)
        )

    mismatch_nonzero = (
        exp_df['time_int'] == 0
    ) & (
        exp_df['time_string'].str[1:].astype(int) != 0
    )
    if mismatch_nonzero.any():
        sample = exp_df.loc[mismatch_nonzero, ['image_id', 'time_int', 'time_string']].head()
        print(
            "⚠️ Warning: Detected rows where time_int == 0 but time_string indicates non-zero frame."
            " Review the first mismatches below:\n" + sample.to_string(index=False)
        )

    exp_df['time_key'] = exp_df['time_string'].str[1:]
    exp_df['frame_index_0'] = exp_df['frame_index']

    # Add SAM2 QC flag processing (Refactor-011-B)
    if 'sam2_qc_flags' in exp_df.columns:
        exp_df['sam2_qc_flag'] = exp_df['sam2_qc_flags'].apply(
            lambda x: len(str(x).strip()) > 0 if pd.notna(x) else False
        )
        flagged_count = exp_df['sam2_qc_flag'].sum()
        print(f"🚩 SAM2 QC flags detected: {flagged_count} snips flagged for quality issues")
    else:
        exp_df['sam2_qc_flag'] = False
        print(f"ℹ️ No SAM2 QC flags column found - skipping QC flag processing")
    
    # Transform SAM2 CSV format to legacy format expected by compile_embryo_stats
    # Key transformation: CSV has one row per snip, legacy expects one row per embryo per well per time
    
    # Extract position from bounding box center
    exp_df['xpos'] = (exp_df['bbox_x_min'] + exp_df['bbox_x_max']) / 2
    exp_df['ypos'] = (exp_df['bbox_y_min'] + exp_df['bbox_y_max']) / 2
    
    # Add required columns for legacy compatibility
    exp_df['fraction_alive'] = 1.0  # Placeholder - SAM2 doesn't compute this
    exp_df['region_label'] = exp_df['embryo_id'].str.extract(r'_e(\d+)$')[0].astype(int)
    
    # Add experiment metadata columns that legacy system expects
    exp_df['experiment_date'] = exp_name
    exp_df['well_id'] = exp_df['video_id'].str.extract(r'_([A-H]\d{2})$')[0]
    exp_df['well'] = exp_df['well_id']  # Add 'well' column for legacy compatibility

    # Calculate predicted developmental stage using legacy formula (Kimmel et al 1995)
    # Formula: predicted_stage_hpf = start_age_hpf + time_hours * (0.055 * temperature - 0.57)
    # Handle both 'temperature' and 'temperature_c' column names
    temp_col = 'temperature' if 'temperature' in exp_df.columns else 'temperature_c'
    exp_df['predicted_stage_hpf'] = exp_df['start_age_hpf'] + \
        (exp_df['Time Rel (s)'] / 3600.0) * (0.055 * exp_df[temp_col] - 0.57)

    print(f"✅ SAM2 data transformed to legacy format: {len(exp_df)} rows ready")
    
    return exp_df

def segment_wells(
    root: str | Path,
    exp_name: str, 
    min_sa_um: float = 250_000,
    max_sa_um: float = 2_000_000,
    par_flag: bool = False,
    overwrite_well_stats: bool = False,
):
    """
    LEGACY FUNCTION - Replaced by SAM2 bridge approach.
    
    This function has been replaced by segment_wells_sam2_csv() which uses
    pre-computed SAM2 metadata instead of image processing.
    
    For new workflows, use the SAM2 bridge CSV approach instead.
    """
    print("⚠️ WARNING: Using legacy segment_wells function")
    print("   Consider migrating to segment_wells_sam2_csv() with SAM2 bridge CSV")
    print("   See: docs/refactors/refactor-003-segmentation-pipeline-integration-prd.md")
    
    # Keep original function for backwards compatibility during transition
    
    n_workers = np.ceil(os.cpu_count()/4).astype(int)

    root     = Path(root)
    meta_dir = root / "metadata" / "built_metadata_files"
    out_dir     = meta_dir / "embryo_metadata_df_tracked.csv"

    # 1) load + diff
    meta_df     = pd.read_csv(meta_dir / f"{exp_name}_metadata.csv", low_memory=False)
    df_to_process = get_unprocessed_metadata(meta_df, out_dir, overwrite_well_stats)
    df_to_process["experiment_date"] = df_to_process["experiment_date"].astype(str)
    
    # 2) sample shapes, drop missing dates
    # dates   = df_to_process["experiment_date"].unique().tolist()
    ff_dir = root/"built_image_data"/"stitched_FF_images"/exp_name
    shape = sample_experiment_shapes(ff_dir)

    # if missing_dates:
    #     df_to_process = df_to_process.loc[
    #         ~df_to_process["experiment_date"].isin(missing_dates)
    #     ].reset_index(drop=True)

    # 3) build shape_arr (N×2)
    # image_shape_array = np.vstack([
    #     shapes[dt] for dt in df_to_process["experiment_date"]
    # ])

    # 4) find mask folder & collect mask paths, drop missing
    seg_root     = root / "segmentation"
    mask_root    = next(p for p in seg_root.iterdir() if p.is_dir() and "mask" in p.name)
    images_to_process, valid_idx = get_mask_paths_from_diff(df_to_process, mask_root, strict=False)

    if len(valid_idx) < len(df_to_process):
        missing = set(df_to_process.index) - set(valid_idx)
        warnings.warn(
            f"segment_wells: skipping {len(missing)} rows with no mask (indices {sorted(missing)})",
            stacklevel=2
        )
        df_to_process = df_to_process.loc[valid_idx].reset_index(drop=True)
        # image_shape_array     = image_shape_array[valid_idx]

    if df_to_process.empty:
        print("✅ segment_wells: nothing new to process.")
        return {}

    else:
        # initialize empty columns to store embryo information
        df_to_process = df_to_process.copy()
        # remove nuisance columns 
        drop_cols = [col for col in df_to_process.columns if "Unnamed" in col]
        df_to_process = df_to_process.drop(labels=drop_cols, axis=1)

        df_to_process["n_embryos_observed"] = np.nan
        df_to_process["FOV_size_px"] = shape[0] * shape[1]
        df_to_process["FOV_height_px"] = shape[0]
        df_to_process["FOV_width_px"] = shape[1]
        for n in range(4):  # allow for a maximum of 4 embryos per well
            for suffix in ("x", "y", "label", "frac_alive"):
                col_name = f"e{n}_{suffix}"
                if suffix in ("x", "y", "frac_alive"):
                    df_to_process[col_name] = pd.Series([pd.NA] * len(df_to_process), dtype="Float32")
                else:
                    df_to_process[col_name] = pd.Series([pd.NA] * len(df_to_process), dtype="Int8")


        ##########################
        # extract position and live/dead status of each embryo in each well
        ##########################

        meta_lookup   = { (row.well, row.experiment_date, row.time_int): idx
                  for idx, row in df_to_process.iterrows() }
        
        # initialize function
        run_count_embryos = partial(count_embryo_regions, meta_lookup=meta_lookup, image_list=images_to_process, master_df_update=df_to_process,
                                              max_sa_um=max_sa_um, min_sa_um=min_sa_um)
        
        if par_flag:
            raw = process_map(run_count_embryos, range(len(images_to_process)), max_workers=n_workers, chunksize=4)
        else:
            raw = [
                run_count_embryos(i)
                for i in tqdm(range(len(images_to_process)), desc="Counting embryo regions…")
            ]

        # 7) scatter results back
        local_idxs, rows = zip(*raw)
        result_df = pd.concat(rows, ignore_index=True)
        df_to_process.loc[list(local_idxs), result_df.columns] = result_df.values

        # Next, iterate through the extracted positions and use rudimentary tracking to assign embryo instances to stable
        # embryo_id that persists over time

        # get list of unique well instances
        if np.any(np.isnan(df_to_process["n_embryos_observed"].values.astype(float))):
            raise Exception("Missing rows found in metadata df")

        well_id_list = df_to_process["well_id"].unique()
        track_df_list = []
        # print("Performing embryo tracking...")
        for well_id in tqdm(well_id_list, "Doing embryo tracking..."):
            track_df_list.append(do_embryo_tracking(well_id, df_to_process))

        track_df_list = [df for df in track_df_list if isinstance(df, pd.DataFrame)]
        tracked = pd.concat(track_df_list, ignore_index=True)
        # embryo_metadata_df = embryo_metadata_df.iloc[:, 1:]

        # if track_ckpt.exists() and not overwrite_well_stats:
        #     prev = pd.read_csv(track_ckpt, low_memory=False)
        #     tracked  = pd.concat([prev, tracked], ignore_index=True)
        
        # tracked.to_csv(track_ckpt, index=False)

        # print(f"✔️  wrote {track_ckpt}")

        tracked = tracked.dropna(subset=["region_label"])

        return tracked


def compile_embryo_stats(root: str, 
                         tracked_df: pd.DataFrame, 
                         overwrite_flag: bool=False,
                         ld_rat_thresh: float=0.9, 
                         qc_scale_um: int=150, 
                         n_workers: int=1):

    par_flag = n_workers > 1

    # meta_root = os.path.join(root, 'metadata', "built_metadata_files", '')
    # segmentation_path = os.path.join(root, 'built_image_data', 'segmentation', '')

    # track_path = (os.path.join(meta_root, "embryo_metadata_df_tracked.csv"))
    # embryo_metadata_df = pd.read_csv(track_path, index_col=0)

    ######
    # Add key embryo characteristics and flag QC issues
    ######
    # initialize new variables
    new_cols = ["surface_area_um", "surface_area_um", "length_um", "width_um", "bubble_flag",
                "focus_flag", "frame_flag", "dead_flag", "no_yolk_flag"]
    tracked_df["surface_area_um"] = pd.Series([pd.NA] * len(tracked_df), dtype="Float32")
    tracked_df["length_um"] = pd.Series([pd.NA] * len(tracked_df), dtype="Float32")
    tracked_df["width_um"] = pd.Series([pd.NA] * len(tracked_df), dtype="Float32")
    tracked_df["bubble_flag"] = False
    tracked_df["focus_flag"] = False
    tracked_df["frame_flag"] = False  
    tracked_df["dead_flag"] = False
    tracked_df["no_yolk_flag"] = False

    # make stable embryo ID
    if 'time_key' in tracked_df.columns:
        time_component = tracked_df['time_key'].astype(str)
    elif 'time_string' in tracked_df.columns:
        time_component = tracked_df['time_string'].astype(str).str.replace(r'^[Tt]', '', regex=True)
    else:
        time_component = tracked_df['time_int'].astype(int).astype(str)
    tracked_df["snip_id"] = tracked_df["embryo_id"] + "_t" + time_component.str.zfill(4)

    # check for existing embryo metadata
    # if os.path.isfile(os.path.join(meta_root, "embryo_metadata_df.csv")) and not overwrite_flag:
    #     embryo_metadata_df_prev = pd.read_csv(os.path.join(meta_root, "embryo_metadata_df.csv"))
    #     # First, check to see if there are new embryo-well-time entries (rows)
    #     merge_skel = embryo_metadata_df.loc[:, ["embryo_id", "time_int"]]
    #     df_all = merge_skel.merge(embryo_metadata_df_prev.drop_duplicates(subset=["embryo_id", "time_int"]), on=["embryo_id", "time_int"],
    #                              how='left', indicator=True)
    #     diff_indices = np.where(df_all['_merge'].values == 'left_only')[0].tolist()

    #     # second, check to see if some or all of the stat columns are already filled in prev table
    #     for nc in new_cols:
    #         embryo_metadata_df[nc] = df_all.loc[:, nc]
    #         sa_nan = np.isnan(embryo_metadata_df["surface_area_um"].values.astype(float))
    #         bb_nan = embryo_metadata_df["bubble_flag"].astype(str) == "nan"
    #         ff_nan = embryo_metadata_df["focus_flag"].astype(str) == "nan"

    #     nan_indices = np.where(sa_nan | ff_nan | bb_nan)[0].tolist()
    #     indices_to_process = np.unique(diff_indices + nan_indices).tolist()
    # else:
    indices_to_process = range(tracked_df.shape[0])

    tracked_df.reset_index(inplace=True, drop=True)

    run_embryo_stats = partial(get_embryo_stats, root=root, embryo_metadata_df=tracked_df,
                                            qc_scale_um=qc_scale_um, ld_rat_thresh=ld_rat_thresh)
    if par_flag:
        emb_df_list = process_map(run_embryo_stats, indices_to_process, max_workers=n_workers, chunksize=4)
        
    else:
        emb_df_list = []
        for index in tqdm(indices_to_process, "Extracting embryo stats..."):
            df_temp = run_embryo_stats(index) #, root, embryo_metadata_df, qc_scale_um, ld_rat_thresh)
            emb_df_list.append(df_temp)

    # update metadata
    emb_df = pd.concat(emb_df_list, axis=0, ignore_index=True)
    emb_df = emb_df.loc[:, ["snip_id"] + new_cols]

    snip1 = emb_df["snip_id"].to_numpy()
    snip2 = tracked_df.loc[indices_to_process, "snip_id"].to_numpy()
    assert np.all(snip1==snip2)
    tracked_df.loc[indices_to_process, new_cols] = emb_df.loc[:, new_cols].values

    # Build03 does NOT compute use_embryo_flag - that happens in Build04
    # Individual QC flags are computed and stored for Build04 to use

    # tracked_df.to_csv(os.path.join(meta_root, "embryo_metadata_df.csv"))

    return tracked_df


def extract_embryo_snips(root: str | Path,
                         stats_df: pd.DataFrame,
                         outscale: float=7.8,
                         n_workers: int=1,
                         overwrite_flag: bool=False, outshape=None, dl_rad_um=75):
    """
    Extract embryo snips from full-frame images.

    Parameters
    ----------
    root : Path
        Project root directory
    stats_df : pd.DataFrame
        Metadata dataframe with embryo information
    outscale : float, default 7.8
        Target resolution in μm/pixel. Default 7.8 μm/px provides 4.5mm × 2.0mm
        capture window (576×256 px), ensuring large embryos (~3.5mm) fill max 75%
        of frame with margin for rotation. Previous default (6.5 μm/px) caused
        clipping for large embryos.
    n_workers : int, default 1
        Number of parallel workers
    overwrite_flag : bool, default False
        Whether to overwrite existing snips
    outshape : list, optional
        Output shape [height, width] in pixels. Default [576, 256]
    dl_rad_um : float, default 75
        Radius for Gaussian smoothing in micrometers
    """

    par_flag = n_workers > 1
    root = Path(root)

    if outshape == None:
        outshape = [576, 256]

    dates = stats_df["experiment_id"].unique()
    if len(dates) > 1:
        raise Exception(f"Detected multiple dates in input dataset: {dates}")

    exp_id = str(dates[0])

    # read in metadata
    meta_root = root / 'metadata' / "embryo_metadata_files"
    os.makedirs(meta_root, exist_ok=True)
    stats_df = stats_df.drop_duplicates(subset=["snip_id"])

    # Validate SAM2 artifact consistency before processing (fail-fast on pipeline sync issues)
    validate_sam2_artifact_consistency(root, exp_id)

    # make directory for embryo snips
    im_snip_dir = root / 'training_data' / 'bf_embryo_snips'
    mask_snip_dir = root / 'training_data' / 'bf_embryo_masks'
    mask_snip_dir.mkdir(parents=True, exist_ok=True)
    #embryo_metadata_df["embryo_id"] + "_" + embryo_metadata_df["time_int"].astype(str)

    if overwrite_flag or not im_snip_dir.is_dir():
        im_snip_dir.mkdir(parents=True, exist_ok=True)
        export_indices = range(stats_df.shape[0])
        stats_df["out_of_frame_flag"] = False
        stats_df["snip_um_per_pixel"] = outscale
    else:
        # get list of exported images
        dates = stats_df["experiment_id"].unique()
        # BUG FIX: Use glob (non-recursive) instead of rglob to search only in the specific date directory
        extant_images = list(chain.from_iterable(
                                sorted((im_snip_dir / str(d)).glob(f"{d}*.jpg")) for d in dates
                            ))
        extant_snip_array = np.asarray([e.name.replace(".jpg", "") for e in extant_images])

        # Find which snip_ids need to be exported (not in extant_snip_array)
        # BUG FIX: Map missing snip IDs back to DataFrame row indices
        missing_snips = set(stats_df["snip_id"]) - set(extant_snip_array)
        export_indices = stats_df[stats_df["snip_id"].isin(missing_snips)].index.tolist()

        stats_df.loc[export_indices, "out_of_frame_flag"] = False
        stats_df.loc[export_indices, "snip_um_per_pixel"] = outscale


    stats_df["time_int"] = stats_df["time_int"].astype(int)

    # draw random sample to estimate background
    # print("Estimating background...")
    px_mean, px_std = 10, 5 #estimate_image_background(root, stats_df, bkg_seed=309, n_bkg_samples=100)

    # Normalize boolean dtypes to avoid pandas bitwise errors
    if "out_of_frame_flag" in stats_df.columns and stats_df["out_of_frame_flag"].dtype != bool:
        _o = stats_df["out_of_frame_flag"].astype(str).str.strip().str.lower()
        stats_df["out_of_frame_flag"] = _o.map({
            "true": True, "false": False,
            "1": True, "0": False,
            "yes": True, "no": False
        }).fillna(False).astype(bool)

    # extract snips
    out_of_frame_flags = []

    # DEBUG: Track export attempts for problem embryos
    debug_snip_prefixes = ["20250305_H04_e01", "20250305_A11_e01"]
    debug_export_attempts = {prefix: [] for prefix in debug_snip_prefixes}
    debug_export_failures = {prefix: [] for prefix in debug_snip_prefixes}

    run_export_snips = partial(export_embryo_snips,root=root, stats_df=stats_df,
                                      dl_rad_um=dl_rad_um, outscale=outscale, outshape=outshape,
                                      px_mean=px_mean, px_std=px_std)
    if par_flag:
        out_of_frame_flags = process_map(run_export_snips, export_indices, max_workers=n_workers, chunksize=4)

    else:
        for r in tqdm(export_indices, "Exporting snips..."):
            # DEBUG: Track attempts for problem embryos
            snip_id = stats_df.iloc[r]["snip_id"]
            for prefix in debug_snip_prefixes:
                if snip_id.startswith(prefix):
                    debug_export_attempts[prefix].append(snip_id)

            try:
                oof = run_export_snips(r)
                out_of_frame_flags.append(oof)
            except Exception as e:
                # Track failures
                for prefix in debug_snip_prefixes:
                    if snip_id.startswith(prefix):
                        debug_export_failures[prefix].append((snip_id, str(e)))
                # Re-raise to maintain original behavior
                raise
        
    out_of_frame_flags = np.asarray(out_of_frame_flags)

    # DEBUG: Print export summary for problem embryos
    print("\n" + "="*80)
    print("📊 DEBUG: Export Summary for Problem Embryos")
    print("="*80)
    for prefix in debug_snip_prefixes:
        attempted = len(debug_export_attempts[prefix])
        failed = len(debug_export_failures[prefix])
        if attempted > 0 or failed > 0:
            print(f"\n{prefix}:")
            print(f"   • Export attempts: {attempted}")
            print(f"   • Export failures: {failed}")
            if failed > 0:
                print(f"   • Failed snips:")
                for snip_id, error in debug_export_failures[prefix][:5]:  # Show first 5
                    print(f"      - {snip_id}: {error}")
            if attempted > 0:
                print(f"   • Sample attempted: {debug_export_attempts[prefix][:3]}")
    print("="*80 + "\n")

    # add oof flag
    stats_df.loc[export_indices, "out_of_frame_flag"] = out_of_frame_flags
    # Note: use_embryo_flag is NOT set in Build03 - Build04 will compute it based on all QC flags

    # save
    exp_name = dates[0]
    stats_df.to_csv(meta_root / f"{exp_name}_embryo_metadata.csv", index=False)
    
    # Also write Build04-compatible format at expected path
    build04_meta_root = root / 'metadata' / "combined_metadata_files"
    os.makedirs(build04_meta_root, exist_ok=True)
    stats_df.to_csv(build04_meta_root / "embryo_metadata_df01.csv", index=False)


# ========================================================================================
# SAM2 Integration Functions (Moved from run_build03.py for QC Restoration)
# ========================================================================================

def _ensure_predicted_stage_hpf(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Add `predicted_stage_hpf` using the legacy Kimmel-style formula if missing.

    Formula (hours post fertilization):
      start_age_hpf + (Time Rel (s)/3600) * (0.055*temperature - 0.57)

    Only applies if the requisite columns exist. Otherwise, the input is
    returned unchanged.

    This is the exact same logic from the legacy build03A_process_images.py.
    Handles both 'temperature' and 'temperature_c' column names.
    """
    # Check for temperature column (either 'temperature' or 'temperature_c')
    temp_col = 'temperature' if 'temperature' in df.columns else 'temperature_c'
    needed = {"start_age_hpf", "Time Rel (s)", temp_col}

    if verbose:
        print(f"      • DataFrame columns: {list(df.columns)}")
        print(f"      • Needed columns: {needed}")
        print(f"      • Columns present: {needed.intersection(set(df.columns))}")
        print(f"      • All needed present: {needed.issubset(df.columns)}")
        print(f"      • predicted_stage_hpf already exists: {'predicted_stage_hpf' in df.columns}")

    if needed.issubset(df.columns):
        try:
            df = df.copy()

            if verbose:
                print(f"      • Sample values:")
                for col in needed:
                    sample_vals = df[col].head(3).tolist()
                    print(f"        - {col}: {sample_vals}")

            df["predicted_stage_hpf"] = (
                df["start_age_hpf"].astype(float)
                + (df["Time Rel (s)"].astype(float) / 3600.0)
                  * (0.055 * df[temp_col].astype(float) - 0.57)
            )

            if verbose:
                print(f"      • Calculated predicted_stage_hpf: {df['predicted_stage_hpf'].head(3).tolist()}")

        except Exception as e:
            # Leave silently unchanged if types are malformed; downstream code will not rely on this.
            if verbose:
                print(f"      • ERROR in calculation: {e}")
            pass
    return df


def _log(verbose: bool, msg: str):
    """Helper function for conditional logging."""
    if verbose:
        print(msg)


def _parse_embryo_number(embryo_id: str) -> Optional[int]:
    """Parse embryo number from embryo_id (e.g., 'embryo_id_e01' -> 1)."""
    m = re.search(r"_e(\d+)$", embryo_id)
    if not m:
        return None
    return int(m.group(1).lstrip("0") or "0")


def _collect_rows_from_sam2_csv(csv_path: Path, exp: str, verbose: bool = False) -> list[Dict[str, str]]:
    """Parse SAM2 CSV and generate row data with QC flag detection.

    Note: Does NOT compute use_embryo_flag - that's handled by Build04
    using determine_use_embryo_flag() from src.build.qc.embryo_flags.
    """
    rows: list[Dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        need_any = {"image_id", "embryo_id"}
        if not need_any.issubset(header):
            raise ValueError(f"CSV missing required columns: {need_any - header}")
        for r in reader:
            image_id = r.get("image_id", "").strip()
            embryo_id = r.get("embryo_id", "").strip()
            if not image_id or not embryo_id:
                continue
                
            # Extract directly from CSV
            video_id = r.get("video_id", "").strip()
            well_id = r.get("well_id", "").strip()
            snip_id = r.get("snip_id", "").strip()

            raw_time_key = r.get("time_key", "").strip()
            time_stub = ""
            if raw_time_key:
                time_stub = f"{int(raw_time_key):04d}" if raw_time_key.isdigit() else raw_time_key
            else:
                time_string = r.get("time_string", "").strip()
                if time_string:
                    if time_string[0].lower() == 't':
                        time_string = time_string[1:]
                    if time_string.isdigit():
                        time_stub = f"{int(time_string):04d}"
                if not time_stub:
                    raw_time_int = r.get("time_int", "").strip()
                    if raw_time_int:
                        try:
                            time_stub = f"{int(raw_time_int):04d}"
                        except ValueError:
                            time_stub = ""
                if not time_stub:
                    frame_index = r.get("frame_index", "").strip()
                    if frame_index:
                        try:
                            time_stub = f"{int(frame_index):04d}"
                        except ValueError:
                            time_stub = ""

            # Generate snip_id if missing
            if not snip_id and time_stub:
                snip_id = f"{embryo_id}_t{time_stub}"
            
            out = {
                "exp_id": exp,
                "video_id": video_id,
                "well_id": well_id,
                "image_id": image_id,
                "embryo_id": embryo_id,
                "snip_id": snip_id,
                "time_int": time_stub,
                "time_key": time_stub,
                # Geometry placeholders (to be populated by _compute_row_geometry_and_qc)
                "area_px": "",
                "perimeter_px": "",
                "centroid_x_px": "",
                "centroid_y_px": "",
                "area_um2": "",
                "perimeter_um": "",
                "centroid_x_um": "",
                "centroid_y_um": "",
                # QC flags placeholders (to be populated by _compute_row_geometry_and_qc)
                "dead_flag": "",
                "no_yolk_flag": "",
                "focus_flag": "",
                "bubble_flag": "",
                "frame_flag": "",
                "fraction_alive": "",
                "speed": "",
                # Provenance
                "exported_mask_path": r.get("exported_mask_path", ""),
                "computed_at": datetime.now().isoformat(),
                # use_embryo_flag NOT computed in Build03 - Build04 will create it
                "predicted_stage_hpf": "",
                # Raw metadata for stage calculation
                "start_age_hpf": r.get("start_age_hpf", ""),
                "Time Rel (s)": r.get("Time Rel (s)", "") or r.get("relative_time_s", ""),
                "temperature": r.get("temperature", ""),
                # Pixel scale data (eliminates Build01 dependency)
                "width_um": r.get("width_um", ""),
                "width_px": r.get("width_px", ""), 
                "height_um": r.get("height_um", ""),
                "height_px": r.get("height_px", ""),
                # SAM2 QC flags for final decision logic
                "sam2_qc_flags": r.get("sam2_qc_flags", ""),
            }
            rows.append(out)
    _log(verbose, f"   • Collected {len(rows)} rows from {csv_path.name}")
    return rows


def _compute_row_geometry_and_qc(row: Dict[str, str], root: Path, verbose: bool = False) -> None:
    """Compute geometry and comprehensive QC for a row using SAM2 + Build02 masks.
    
    This function combines basic geometry computation with comprehensive QC analysis:
    1. Loads SAM2 labeled mask and computes geometry (area, perimeter, centroids)
    2. Loads Build02 auxiliary masks (via, yolk, focus, bubble) 
    3. Computes legacy QC flags using proven QC functions
    4. Updates row with all computed geometry and QC data
    
    Uses existing _load_build02_masks_for_row() with is_sam2_pipeline=True for proper
    mask processing that avoids the "all embryos flagged as dead" bug.
    """
    try:
        # Load SAM2 labeled mask using CSV path
        mask_path = resolve_sandbox_embryo_mask_from_csv(root, row)
        if not mask_path.exists():
            _log(verbose, f"⚠️ SAM2 mask not found: {mask_path}")
            row["frame_flag"] = "true"
            return
            
        img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            _log(verbose, f"⚠️ Could not read mask: {mask_path}")
            row["frame_flag"] = "true"
            return
            
        if img.ndim == 3:
            img = img[:, :, 0]  # Take first channel if RGB
            
        # Parse embryo number and select label
        embryo_num = _parse_embryo_number(row.get("embryo_id", ""))
        if embryo_num is None:
            _log(verbose, f"⚠️ Could not parse embryo number from {row.get('embryo_id')}")
            row["frame_flag"] = "true"
            return
            
        # Create binary mask for this specific embryo
        binary_mask = (img == embryo_num).astype("uint8")
        area_px = int(binary_mask.sum())
        
        if area_px == 0:
            _log(verbose, f"⚠️ No pixels found for embryo {embryo_num}")
            row["frame_flag"] = "true"
            return
            
        # Compute geometry using OpenCV
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            row["frame_flag"] = "true"
            return
            
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter_px = float(cv2.arcLength(largest_contour, True))
        
        # Compute centroids
        M = cv2.moments(binary_mask)
        if M["m00"] > 0:
            cx_px = float(M["m10"] / M["m00"])
            cy_px = float(M["m01"] / M["m00"])
        else:
            cx_px = cy_px = 0.0
            
        # Store pixel geometry
        row["area_px"] = str(area_px)
        row["perimeter_px"] = f"{perimeter_px:.2f}"
        row["centroid_x_px"] = f"{cx_px:.2f}"
        row["centroid_y_px"] = f"{cy_px:.2f}"
        
        # Get pixel scale from SAM2 CSV data
        sx, sy = None, None
        try:
            width_um = float(row.get("width_um", "") or 0)
            width_px = float(row.get("width_px", "") or 0)
            height_um = float(row.get("height_um", "") or 0)
            height_px = float(row.get("height_px", "") or 0)
            if width_um > 0 and width_px > 0 and height_um > 0 and height_px > 0:
                sx = width_um / width_px   # um per pixel in X
                sy = height_um / height_px # um per pixel in Y
        except (ValueError, TypeError, ZeroDivisionError):
            pass
            
        # Convert to microns if pixel scale available
        if sx and sy and sx > 0 and sy > 0:
            area_um2 = area_px * sx * sy
            per_um = perimeter_px * (sx + sy) / 2.0
            row["area_um2"] = f"{area_um2:.4f}"
            row["perimeter_um"] = f"{per_um:.4f}"
            row["centroid_x_um"] = f"{cx_px * sx:.4f}"
            row["centroid_y_um"] = f"{cy_px * sy:.4f}"
            px_dim_um = (sx + sy) / 2.0
        else:
            px_dim_um = 1.0
            
        # Load Build02 auxiliary masks using existing proven function
        # CRITICAL: Use is_sam2_pipeline=True for proper mask processing
        aux_masks = _load_build02_masks_for_row(
            root, row, target_shape=binary_mask.shape, is_sam2_pipeline=True
        )
        _log(verbose, f"   • Loaded auxiliary masks: {list(aux_masks.keys())}")
        
        # Compute comprehensive QC using existing proven functions
        via_mask = aux_masks.get("via")
        frac_alive = compute_fraction_alive(binary_mask, via_mask)
        row["fraction_alive"] = str(frac_alive) if np.isfinite(frac_alive) else ""
        
        # Compute QC flags using existing qc_utils functions
        qc_flags = compute_qc_flags(
            binary_mask,
            px_dim_um=px_dim_um,
            qc_scale_um=150,  # Standard QC scale
            yolk_mask=aux_masks.get("yolk"),
            focus_mask=aux_masks.get("focus"),
            bubble_mask=aux_masks.get("bubble"),
        )
        
        # Update row with QC flags
        for flag_name, flag_value in qc_flags.items():
            row[flag_name] = "true" if flag_value else "false"
            
        # Compute dead flag based on fraction_alive threshold
        if np.isfinite(frac_alive):
            row["dead_flag"] = "true" if frac_alive < 0.9 else "false"
        else:
            row["dead_flag"] = "false"
            
        _log(verbose, f"   • Computed geometry and QC for {row.get('embryo_id')}")
        
    except Exception as e:
        _log(verbose, f"⚠️ Error in geometry/QC computation: {e}")
        # Set frame_flag to mark as unusable but keep row valid
        row["frame_flag"] = "true"


# REMOVED: _set_final_use_embryo_flag() function
# Build03 no longer computes use_embryo_flag - that happens in Build04
# using determine_use_embryo_flag() from src.build.qc.embryo_flags


def compile_embryo_stats_sam2(root: str | Path, tracked_df: pd.DataFrame, n_workers: int = 1, skip_geometry_qc: bool = False) -> pd.DataFrame:
    """Compile comprehensive embryo statistics for SAM2 pipeline with full QC flag computation.

    This function provides the complete SAM2 integration workflow:
    1. Processes each row with geometry computation and comprehensive QC flag analysis

    Note: Does NOT compute use_embryo_flag - that happens in Build04.
    2. Computes individual QC flags (focus, bubble, frame, dead, no_yolk)
    3. Combines SAM2 QC flags with legacy Build02 mask-based QC flags

    This replaces the functionality lost in run_build03.py by consolidating all
    processing logic in build03A with comprehensive QC flag computation.

    Args:
        root: Project root directory
        tracked_df: DataFrame from segment_wells_sam2_csv()
        n_workers: Number of workers (currently single-threaded)
        skip_geometry_qc: If True, skip geometry computation and QC checks (only export snip IDs)

    Returns:
        DataFrame with complete geometry and QC data
    """
    root = Path(root)
    df = tracked_df.copy()

    # Check environment variable if not explicitly set
    if not skip_geometry_qc:
        skip_geometry_qc = os.environ.get("BUILD03_SKIP_GEOMETRY_QC", "0") == "1"

    if skip_geometry_qc:
        print(f"⚡ Fast mode: Skipping geometry QC computation (only exporting snip IDs)")
        # Set all QC flags to empty/false (no use_embryo_flag - Build04 will create it)
        df["area_px"] = ""
        df["perimeter_px"] = ""
        df["centroid_x_px"] = ""
        df["centroid_y_px"] = ""
        df["area_um2"] = ""
        df["perimeter_um"] = ""
        df["centroid_x_um"] = ""
        df["centroid_y_um"] = ""
        df["dead_flag"] = "false"
        df["no_yolk_flag"] = "false"
        df["focus_flag"] = "false"
        df["bubble_flag"] = "false"
        df["frame_flag"] = "false"
        df["fraction_alive"] = "1.0"
        df["speed"] = ""
        # use_embryo_flag NOT set in Build03 - Build04 will create it

        # Add predicted stage calculation at DataFrame level
        result_df = _ensure_predicted_stage_hpf(df, verbose=False)

        print(f"✅ Fast mode complete: {len(result_df)} snip IDs exported")
        return result_df

    print(f"🔄 Processing {len(df)} embryo records with comprehensive QC...")

    # Convert DataFrame to list of dicts for row-by-row processing
    rows = df.to_dict('records')

    # Process each row with geometry and QC computation
    for i, row in enumerate(tqdm(rows, desc="Computing geometry and QC")):
        _compute_row_geometry_and_qc(row, root, verbose=False)

        # Build03 does NOT compute use_embryo_flag - that happens in Build04
        # Individual QC flags (focus, bubble, frame, dead, no_yolk) are computed above

        rows[i] = row

    # Convert back to DataFrame
    result_df = pd.DataFrame(rows)

    # Add predicted stage calculation at DataFrame level
    result_df = _ensure_predicted_stage_hpf(result_df, verbose=False)

    # Summary statistics
    total_embryos = len(result_df)

    # Count QC flags using robust logic (exclude pandas NaN artifacts)
    def is_meaningful_qc_flag(val):
        if pd.isna(val):
            return False
        flag_str = str(val).strip().lower()
        return flag_str and flag_str not in ["", "nan", "none", "null"]

    # Auto-detect all columns with "flag" in the name
    flag_columns = [col for col in result_df.columns if "flag" in col.lower()]

    # Count SAM2 flags separately (if present)
    sam2_flagged = 0
    if "sam2_qc_flags" in result_df.columns:
        sam2_flagged = result_df["sam2_qc_flags"].apply(is_meaningful_qc_flag).sum()

    # Count embryos with any disqualifying flag
    flagged_mask = pd.Series([False] * len(result_df), index=result_df.index)
    for flag_col in flag_columns:
        if flag_col != "sam2_qc_flags":  # Skip sam2_qc_flags as it's counted separately
            flagged_mask |= (result_df[flag_col] == "true")

    legacy_flagged = flagged_mask.sum()
    usable_embryos = total_embryos - legacy_flagged

    print(f"✅ QC Processing Complete:")
    print(f"   • Total embryos: {total_embryos}")
    print(f"   • SAM2 QC flagged: {sam2_flagged}")
    print(f"   • Legacy QC flagged: {legacy_flagged}")
    print(f"   • Final usable: {usable_embryos} ({usable_embryos/total_embryos*100:.1f}%)")
    
    return result_df


if __name__ == "__main__":

    # The project root is now defined by the REPO_ROOT constant at the top of the file
    # to avoid hardcoded paths.
    root = REPO_ROOT
    
    # print('Compiling well metadata...')
    # build_well_metadata_master(root)

    # SAM2 Pipeline Integration - Phase 2 Implementation
    # Using bridge CSV instead of legacy image processing
    exp_name = "20250612_30hpf_ctrl_atf6"  # Production experiment with validated SAM2 data
    
    # The path to the CSV is now constructed relative to the project root.
    # The PRD specifies the CSV is in the project root.
    sam2_csv_path = root / f"sam2_metadata_{exp_name}.csv"

    # Option 1: Use new SAM2 CSV-based function (recommended)
    tracked_df = segment_wells_sam2_csv(root, exp_name=exp_name, sam2_csv_path=sam2_csv_path)

    # Production mode - process full dataset
    print(f"Production mode: processing full dataset of {len(tracked_df)} rows.")

    # Option 2: Use legacy function (fallback during transition)  
    # tracked_df = segment_wells(root, exp_name=exp_name)

    stats_df = compile_embryo_stats(root, tracked_df)

    # print('Extracting embryo snips...')
    extract_embryo_snips(root, stats_df=stats_df, outscale=6.5, dl_rad_um=50, overwrite_flag=False)
