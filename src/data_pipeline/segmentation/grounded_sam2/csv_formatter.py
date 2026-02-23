"""
SAM2 Metadata CSV Formatter

Flatten nested SAM2 JSON outputs into tabular CSV format for downstream analysis.
Transforms hierarchical experiment → video → frame → embryo structure into
one row per snip (embryo × frame combination).

Key Functions:
    - flatten_sam2_json_to_csv: Main conversion function
    - add_well_metadata: Merge well-level experimental metadata
    - encode_rle_for_csv: Prepare RLE masks for CSV storage
    - validate_csv_schema: Ensure output matches expected schema

Example Usage:
    ```python
    from pathlib import Path
    import json

    # Load SAM2 results
    with open("sam2_results.json") as f:
        sam2_data = json.load(f)

    # Flatten to CSV
    df = flatten_sam2_json_to_csv(
        sam2_data=sam2_data,
        output_csv=Path("segmentation_tracking.csv"),
        well_metadata_path=Path("well_metadata.csv")
    )

    # Result: DataFrame with columns:
    # image_id, embryo_id, snip_id, frame_index, area_px, bbox_*,
    # mask_confidence, mask_rle, exported_mask_path, well_id,
    # is_seed_frame, source_image_path, ...
    ```

Output Schema:
    Required columns for segmentation_tracking.csv:
    - image_id: Unique image identifier (e.g., "exp_A01_t0000")
    - embryo_id: Unique embryo identifier within video
    - snip_id: Unique snip identifier (embryo × frame)
    - frame_index: Temporal index
    - area_px: Mask area in pixels
    - bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max: Bounding box
    - mask_confidence: SAM2 confidence score
    - mask_rle: RLE-encoded mask
    - exported_mask_path: Path to PNG mask file
    - well_id: Well identifier (e.g., "A01")
    - is_seed_frame: Boolean indicating seed frame
    - source_image_path: Path to source image

Notes:
    - Adds well_id column for downstream filtering
    - Adds is_seed_frame boolean for tracking seed frames
    - Adds source_image_path for traceability
    - Adds mask_rle for compact mask storage
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Import centralized schema
from ...schemas.segmentation import REQUIRED_COLUMNS_SEGMENTATION_TRACKING

# Use centralized schema (more authoritative than local copy)
REQUIRED_CSV_COLUMNS = REQUIRED_COLUMNS_SEGMENTATION_TRACKING


def extract_well_index(well_id: str) -> int:
    """
    Extract well index from well identifier.

    Converts standard plate well format (A01-H12) to linear index (1-96).
    Format: Letter (A-H) + Number (01-12)

    Args:
        well_id: Well identifier (e.g., "A01", "H12")

    Returns:
        Linear well index (1-96)

    Example:
        >>> extract_well_index("A01")
        1
        >>> extract_well_index("B01")
        13
        >>> extract_well_index("H12")
        96
    """
    if not well_id or len(well_id) < 2:
        return 0

    row_letter = well_id[0].upper()
    col_str = well_id[1:]

    # Map letter to row (A=0, B=1, ..., H=7)
    row = ord(row_letter) - ord('A')
    if row < 0 or row > 7:
        return 0

    # Extract column number
    try:
        col = int(col_str)
        if col < 1 or col > 12:
            return 0
    except ValueError:
        return 0

    # Linear index (1-based): row * 12 + col
    return row * 12 + col


def extract_time_int(image_id: str) -> int:
    """
    Extract time index from image identifier.

    Typically image_id format is: "exp_well_tXXXX" where XXXX is time index.

    Args:
        image_id: Image identifier (e.g., "exp_A01_t0000")

    Returns:
        Time index as integer, or 0 if not found

    Example:
        >>> extract_time_int("exp_A01_t0000")
        0
        >>> extract_time_int("exp_A01_t0042")
        42
    """
    if '_t' not in image_id:
        return 0

    try:
        # Extract the part after 't'
        time_part = image_id.split('_t')[-1]
        # Convert to int (handle leading zeros)
        return int(time_part)
    except (ValueError, IndexError):
        return 0


def flatten_sam2_json_to_csv(
    sam2_data: Dict[str, Any],
    output_csv: Optional[Path] = None,
    well_metadata_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Flatten nested SAM2 JSON to CSV format.

    Args:
        sam2_data: SAM2 results dict with nested structure
        output_csv: Optional path to save CSV
        well_metadata_path: Optional path to well metadata for merging

    Returns:
        DataFrame with flattened segmentation tracking data

    Example:
        >>> sam2_data = {
        ...     "experiments": {
        ...         "exp1": {
        ...             "videos": {
        ...                 "exp1_A01": {
        ...                     "seed_frame_info": {"seed_frame": "exp1_A01_t0005"},
        ...                     "image_ids": {
        ...                         "exp1_A01_t0005": {
        ...                             "frame_index": 5,
        ...                             "embryos": {
        ...                                 "exp1_A01_e01": {
        ...                                     "snip_id": "exp1_A01_e01_t0005",
        ...                                     "segmentation": {
        ...                                         "area": 1234.5,
        ...                                         "bbox": [10, 20, 100, 120],
        ...                                         "rle": {"counts": "....", "size": [512, 512]}
        ...                                     },
        ...                                     "mask_confidence": 0.95
        ...                                 }
        ...                             }
        ...                         }
        ...                     }
        ...                 }
        ...             }
        ...         }
        ...     }
        ... }
        >>> df = flatten_sam2_json_to_csv(sam2_data)
        >>> len(df)
        1
        >>> df.columns.tolist()[:5]
        ['image_id', 'embryo_id', 'snip_id', 'frame_index', 'area_px']
    """
    rows = []

    experiments = sam2_data.get("experiments", {})

    for exp_id, exp_data in experiments.items():
        videos = exp_data.get("videos", {})

        for video_id, video_data in videos.items():
            # Get seed frame info
            seed_frame_info = video_data.get("seed_frame_info", {})
            seed_frame_id = seed_frame_info.get("seed_frame")

            # Extract well_id from video_id (last part after underscore)
            well_id = video_id.split("_")[-1] if "_" in video_id else video_id

            image_ids = video_data.get("image_ids", {})

            for image_id, image_data in image_ids.items():
                frame_index = image_data.get("frame_index", 0)
                is_seed_frame = (image_id == seed_frame_id)

                # Get source image path if available
                source_image_path = image_data.get("image_path", "")

                embryos = image_data.get("embryos", {})

                for embryo_id, embryo_data in embryos.items():
                    # Extract core fields
                    snip_id = embryo_data.get("snip_id", "")

                    segmentation = embryo_data.get("segmentation", {})
                    area_px = segmentation.get("area", 0.0)
                    bbox = segmentation.get("bbox", [0, 0, 0, 0])
                    mask_confidence = embryo_data.get("mask_confidence", 0.0)

                    # Encode RLE mask
                    mask_rle = encode_rle_for_csv(segmentation)

                    # Generate exported mask path
                    exported_mask_path = f"{image_id}_masks.png"

                    # Extract well_index from well_id (e.g., "A01" -> 1, "B01" -> 13)
                    # Standard 96-well plate: rows A-H, columns 01-12
                    well_index = extract_well_index(well_id)

                    # Extract time_int from image_id (last numerical component after 't')
                    time_int = extract_time_int(image_id)

                    # Calculate centroid from bounding box
                    centroid_x_px = (bbox[0] + bbox[2]) / 2.0 if len(bbox) >= 4 else 0.0
                    centroid_y_px = (bbox[1] + bbox[3]) / 2.0 if len(bbox) >= 4 else 0.0

                    # Build row with all required schema columns
                    row = {
                        # Core IDs
                        "experiment_id": exp_id,
                        "video_id": video_id,
                        "well_id": well_id,
                        "well_index": well_index,
                        "image_id": image_id,
                        "embryo_id": embryo_id,
                        "snip_id": snip_id,
                        "frame_index": frame_index,
                        "time_int": time_int,
                        # Mask data
                        "mask_rle": mask_rle,
                        "area_px": area_px,
                        "bbox_x_min": bbox[0] if len(bbox) > 0 else 0,
                        "bbox_y_min": bbox[1] if len(bbox) > 1 else 0,
                        "bbox_x_max": bbox[2] if len(bbox) > 2 else 0,
                        "bbox_y_max": bbox[3] if len(bbox) > 3 else 0,
                        "mask_confidence": mask_confidence,
                        # Geometry
                        "centroid_x_px": centroid_x_px,
                        "centroid_y_px": centroid_y_px,
                        # SAM2 metadata
                        "is_seed_frame": is_seed_frame,
                        # File references
                        "source_image_path": source_image_path,
                        "exported_mask_path": exported_mask_path,
                    }

                    rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Ensure column order matches schema
    if not df.empty:
        # Reorder columns to match schema
        present_columns = [col for col in REQUIRED_CSV_COLUMNS if col in df.columns]
        df = df[present_columns]

    # Merge with well metadata if provided
    if well_metadata_path is not None and well_metadata_path.exists():
        df = add_well_metadata(df, well_metadata_path)

    # Validate schema
    if not df.empty:
        validate_csv_schema(df)

    # Save to CSV if path provided
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

    return df


def encode_rle_for_csv(segmentation: Dict[str, Any]) -> str:
    """
    Encode RLE mask data for CSV storage.

    Args:
        segmentation: Segmentation dict containing RLE data

    Returns:
        JSON string of RLE or empty string if no RLE

    Example:
        >>> seg = {"rle": {"counts": "eNpj...", "size": [512, 512]}}
        >>> rle_str = encode_rle_for_csv(seg)
        >>> isinstance(rle_str, str)
        True
    """
    # Try different RLE key names
    rle_data = segmentation.get("rle") or segmentation.get("counts")

    if rle_data is None:
        return ""

    # If it's already a string, return it
    if isinstance(rle_data, str):
        return rle_data

    # If it's a dict, convert to JSON string
    if isinstance(rle_data, dict):
        return json.dumps(rle_data)

    return str(rle_data)


def add_well_metadata(
    df: pd.DataFrame,
    well_metadata_path: Path
) -> pd.DataFrame:
    """
    Merge well-level metadata into segmentation tracking dataframe.

    Args:
        df: Segmentation tracking dataframe
        well_metadata_path: Path to well metadata CSV

    Returns:
        DataFrame with merged well metadata

    Example:
        >>> df = pd.DataFrame({"well_id": ["A01", "A02"]})
        >>> # Assuming well_metadata.csv has well_id, genotype, treatment columns
        >>> merged = add_well_metadata(df, Path("well_metadata.csv"))
        >>> "genotype" in merged.columns
        True
    """
    if df.empty:
        return df

    try:
        well_metadata = pd.read_csv(well_metadata_path)

        # Merge on well_id
        if "well_id" in df.columns and "well_id" in well_metadata.columns:
            df = df.merge(
                well_metadata,
                on="well_id",
                how="left",
                suffixes=("", "_well")
            )

    except Exception as e:
        print(f"Warning: Could not merge well metadata: {e}")

    return df


def validate_csv_schema(df: pd.DataFrame) -> None:
    """
    Validate that DataFrame matches expected CSV schema.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If schema validation fails

    Example:
        >>> df = pd.DataFrame({
        ...     "image_id": ["img1"],
        ...     "embryo_id": ["e1"],
        ...     "snip_id": ["s1"],
        ...     "frame_index": [0],
        ...     "area_px": [100.0],
        ...     "bbox_x_min": [10],
        ...     "bbox_y_min": [20],
        ...     "bbox_x_max": [100],
        ...     "bbox_y_max": [120],
        ...     "mask_confidence": [0.9],
        ...     "mask_rle": ["{}"],
        ...     "exported_mask_path": ["mask.png"],
        ...     "well_id": ["A01"],
        ...     "is_seed_frame": [True],
        ...     "source_image_path": ["img.jpg"],
        ... })
        >>> validate_csv_schema(df)  # Should not raise
    """
    # Check for required columns
    missing_cols = set(REQUIRED_CSV_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for non-null values in critical columns
    critical_cols = ["image_id", "embryo_id", "snip_id"]
    for col in critical_cols:
        if df[col].isnull().any():
            raise ValueError(f"Column '{col}' contains null values")

    # Validate boolean column
    if "is_seed_frame" in df.columns:
        if df["is_seed_frame"].dtype != bool:
            # Try to convert
            df["is_seed_frame"] = df["is_seed_frame"].astype(bool)


def load_sam2_json(json_path: Path) -> Dict[str, Any]:
    """
    Load SAM2 results JSON file.

    Args:
        json_path: Path to SAM2 JSON file

    Returns:
        Loaded JSON data as dict

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON is malformed

    Example:
        >>> data = load_sam2_json(Path("results.json"))
        >>> "experiments" in data
        True
    """
    if not json_path.exists():
        raise FileNotFoundError(f"SAM2 JSON not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    return data


def export_sam2_to_csv(
    sam2_json_path: Path,
    output_csv_path: Path,
    well_metadata_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Complete pipeline: Load SAM2 JSON and export to CSV.

    Args:
        sam2_json_path: Path to SAM2 results JSON
        output_csv_path: Path for output CSV
        well_metadata_path: Optional well metadata for merging

    Returns:
        Flattened DataFrame

    Example:
        >>> df = export_sam2_to_csv(
        ...     Path("sam2_results.json"),
        ...     Path("segmentation_tracking.csv"),
        ...     Path("well_metadata.csv")
        ... )
        >>> df.shape[1] >= len(REQUIRED_CSV_COLUMNS)
        True
    """
    # Load JSON
    sam2_data = load_sam2_json(sam2_json_path)

    # Flatten to CSV
    df = flatten_sam2_json_to_csv(
        sam2_data=sam2_data,
        output_csv=output_csv_path,
        well_metadata_path=well_metadata_path
    )

    return df


def get_csv_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for segmentation tracking CSV.

    Args:
        df: Segmentation tracking dataframe

    Returns:
        Dict with summary statistics

    Example:
        >>> df = pd.DataFrame({
        ...     "image_id": ["img1", "img2"],
        ...     "well_id": ["A01", "A01"],
        ...     "is_seed_frame": [True, False]
        ... })
        >>> summary = get_csv_summary(df)
        >>> summary["total_frames"]
        2
    """
    summary = {
        "total_rows": len(df),
        "total_frames": df["image_id"].nunique() if "image_id" in df.columns else 0,
        "total_embryos": df["embryo_id"].nunique() if "embryo_id" in df.columns else 0,
        "total_wells": df["well_id"].nunique() if "well_id" in df.columns else 0,
        "seed_frames": df["is_seed_frame"].sum() if "is_seed_frame" in df.columns else 0,
    }

    return summary
