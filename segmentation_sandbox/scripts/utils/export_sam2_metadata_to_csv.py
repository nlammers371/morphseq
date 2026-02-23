#!/usr/bin/env python3
"""
Export SAM2 Metadata to CSV Bridge Script

Transforms GroundedSam2Annotations.json from rich nested format into flat CSV 
for downstream processing by legacy build scripts.

This script bridges the gap between SAM2's comprehensive segmentation output
and the simplified tabular format expected by the morphseq build pipeline.

Key Features:
- Flattens nested JSON to CSV with exact schema compliance
- Validates mask file existence with configurable error thresholds
- Progress tracking for large datasets (target: <30s for typical experiments)
- Robust error handling with actionable messages
- Uses existing parsing_utils.py for ID consistency
- Supports experiment filtering for selective processing

CSV Schema (Enhanced with Raw Metadata + Build03 Compatibility + QC Flags - 47 columns):
    Core SAM2 columns (18): image_id, embryo_id, snip_id, frame_index, area_px, bbox_x_min, bbox_y_min,
    bbox_x_max, bbox_y_max, mask_confidence, mask_rle, mask_height_px, mask_width_px, exported_mask_path,
    image_path, experiment_id, video_id, is_seed_frame

    Raw image metadata (16): Height (um), Height (px), Width (um), Width (px), BF Channel, Objective,
    Time (s), Time Rel (s), height_um, height_px, width_um, width_px, bf_channel, objective,
    raw_time_s, relative_time_s, microscope, nd2_series_num

    Well-level metadata (7): medium, genotype, chem_perturbation, start_age_hpf, embryos_per_well,
    temperature, well_qc_flag

    Build03 compatibility columns (3): well, time_int, time_string

    SAM2 QC flags (1): sam2_qc_flags

Input Requirements:
- GroundedSam2Annotations.json with standard SAM2 structure
- Optional: masks directory for file validation
- JSON must have 'experiments' and 'segmentation_format' fields

Output:
- CSV file with one row per snip (embryo × frame combination)
- Mask file paths follow convention: {image_id}_masks_emnum_{count}.png
- Boolean is_seed_frame indicates seed frames used for SAM2 initialization

Usage:
    # Basic usage with per-experiment metadata
    python export_sam2_metadata_to_csv.py annotations.json -o output.csv --metadata-json experiment_metadata_20240418.json
    
    # With mask validation  
    python export_sam2_metadata_to_csv.py annotations.json -o output.csv --masks-dir ./masks --metadata-json experiment_metadata_20240418.json
    
    # Filter specific experiments
    python export_sam2_metadata_to_csv.py annotations.json -o output.csv --experiment-filter 20240418 --metadata-json experiment_metadata_20240418.json
    
    # Verbose logging
    python export_sam2_metadata_to_csv.py annotations.json -o output.csv --metadata-json experiment_metadata_20240418.json -v

Error Handling:
- Graceful handling of malformed JSON with specific error messages
- File validation with configurable failure thresholds (50% missing = error)
- Schema validation ensuring CSV matches expected format
- ID parsing validation using segmentation_sandbox conventions

Performance:
- Optimized for large datasets with progress tracking
- Memory-efficient DataFrame operations
- Target: <30 seconds for typical experiment processing
- Tested with sample data: 8 snips in <0.01 seconds

Integration:
- Part of SAM2 pipeline integration (refactor-003)
- Replaces legacy regionprops calculations in build scripts
- Enables separation of segmentation and QC logic
- Compatible with existing export_utils.py for well metadata

Author: SAM2 Pipeline Integration
Version: 1.0.0
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Import parsing utilities from segmentation sandbox
try:
    from .parsing_utils import (
        extract_experiment_id,
        extract_video_id,
        parse_entity_id,
        normalize_frame_number,
        extract_frame_number
    )
except ImportError:
    # Handle direct execution case
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.parsing_utils import (
        extract_experiment_id,
        extract_video_id,
        parse_entity_id,
        normalize_frame_number,
        extract_frame_number
    )

# CSV Schema Constants - Enhanced with raw metadata
CSV_COLUMNS = [
    'image_id', 'embryo_id', 'snip_id', 'frame_index', 'area_px',
    'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max',
    'mask_confidence', 'mask_rle', 'mask_height_px', 'mask_width_px',
    'exported_mask_path', 'image_path', 'experiment_id',
    'video_id', 'is_seed_frame',

    # Raw image metadata from legacy CSV (both original names and aliases)
    'Height (um)', 'Height (px)', 'Width (um)', 'Width (px)',
    'BF Channel', 'Objective', 'Time (s)', 'Time Rel (s)',
    'height_um', 'height_px', 'width_um', 'width_px',
    'bf_channel', 'objective', 'raw_time_s', 'relative_time_s',
    'microscope', 'nd2_series_num',

    # Well-level metadata
    'medium', 'genotype', 'chem_perturbation', 'start_age_hpf',
    'embryos_per_well', 'temperature', 'well_qc_flag',

    # Build03 compatibility columns (extracted from SAM2 JSON)
    'well', 'time_int', 'time_string',

    # SAM2 QC flags (Refactor-011-B)
    'sam2_qc_flags'
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SAM2MetadataExporter:
    """
    Export SAM2 annotations to CSV format for legacy build script integration.
    
    Handles the transformation from SAM2's nested JSON structure to the flat
    CSV schema required by downstream processing steps.
    """
    
    def __init__(self, sam2_json_path: Path, masks_dir: Optional[Path] = None, metadata_json_path: Optional[Path] = None):
        """
        Initialize the exporter.
        
        Args:
            sam2_json_path: Path to GroundedSam2Annotations.json
            masks_dir: Directory containing exported mask PNG files
            metadata_json_path: Path to per-experiment metadata JSON file
        """
        self.sam2_json_path = Path(sam2_json_path)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.metadata_json_path = Path(metadata_json_path) if metadata_json_path else None
        self.sam2_data = None
        self.experiment_metadata = None
        self.build01_wells = set()
        
    def load_and_validate_json(self) -> Dict[str, Any]:
        """
        Load and validate SAM2 JSON file.
        
        Returns:
            Loaded JSON data
            
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            json.JSONDecodeError: If JSON is malformed
            ValueError: If JSON structure is invalid
        """
        if not self.sam2_json_path.exists():
            raise FileNotFoundError(f"SAM2 JSON file not found: {self.sam2_json_path}")
        
        logger.info(f"Loading SAM2 data from {self.sam2_json_path}")
        
        try:
            with open(self.sam2_json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON format in {self.sam2_json_path}: {e}")
        
        # Validate required structure
        self._validate_json_structure(data)
        
        self.sam2_data = data
        logger.info(f"Successfully loaded SAM2 data with {len(data.get('experiments', {}))} experiments")
        
        return data

    def load_experiment_metadata(self) -> None:
        """
        Load per-experiment metadata JSON if path provided.
        This provides pixel dimensions and other raw image data missing from SAM2 segmentation JSON.
        """
        if not self.metadata_json_path:
            logger.warning("No metadata JSON path provided - pixel dimensions will be missing")
            return
            
        if not self.metadata_json_path.exists():
            raise FileNotFoundError(f"Per-experiment metadata JSON not found: {self.metadata_json_path}")
        
        logger.info(f"Loading per-experiment metadata from {self.metadata_json_path}")
        try:
            with open(self.metadata_json_path, 'r') as f:
                self.experiment_metadata = json.load(f)
            logger.info(f"Successfully loaded per-experiment metadata")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON format in {self.metadata_json_path}: {e}")

    def load_build01_metadata(self, experiment_id: str) -> None:
        """
        Load Build01 metadata CSV for specific experiment to identify valid wells for filtering.
        This prevents orphaned SAM2 data from wells that Build01 didn't process.
        
        Args:
            experiment_id: Experiment ID (e.g., "20250529_30hpf_ctrl_atf6")
        """
        # Try to find Build01 metadata CSV matching this experiment
        sam2_dir = self.sam2_json_path.parent
        while sam2_dir.name and sam2_dir != sam2_dir.parent:
            metadata_dir = sam2_dir / "metadata" / "built_metadata_files"
            if metadata_dir.exists():
                # Look for CSV file matching this experiment
                experiment_csv = metadata_dir / f"{experiment_id}_metadata.csv"
                if experiment_csv.exists():
                    logger.info(f"Loading Build01 metadata for {experiment_id} from {experiment_csv}")
                    try:
                        df = pd.read_csv(experiment_csv)
                        if 'well_id' in df.columns:
                            # Extract well names (e.g., "A01" from "20250529_30hpf_ctrl_atf6_A01")
                            for well_id in df['well_id']:
                                if pd.notna(well_id) and isinstance(well_id, str):
                                    well_name = well_id.split('_')[-1]
                                    self.build01_wells.add(well_name)
                            
                            logger.info(f"Found {len(self.build01_wells)} valid wells in Build01 metadata for {experiment_id}")
                            return
                        else:
                            logger.warning(f"Build01 metadata missing 'well_id' column for {experiment_id}")
                    except Exception as e:
                        logger.warning(f"Failed to load Build01 metadata for {experiment_id}: {e}")
            sam2_dir = sam2_dir.parent
        
        logger.warning(f"No Build01 metadata found for experiment {experiment_id} - SAM2 export will include all wells (may cause downstream NaN issues)")
    
    def _validate_json_structure(self, data: Dict[str, Any]) -> None:
        """
        Validate that JSON has required SAM2 structure.
        
        Args:
            data: Loaded JSON data
            
        Raises:
            ValueError: If required fields are missing
        """
        required_fields = ['experiments', 'segmentation_format']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            raise ValueError(f"Missing required fields in SAM2 JSON: {missing_fields}")
            
        if data.get('segmentation_format') != 'rle':
            logger.warning(f"Unexpected segmentation format: {data.get('segmentation_format')}")
    
    def export_to_csv(self, 
                     output_path: Path, 
                     experiment_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Export SAM2 data to CSV format.
        
        Args:
            output_path: Path for output CSV file
            experiment_filter: List of experiment IDs to include (None for all)
            
        Returns:
            Generated DataFrame
            
        Raises:
            ValueError: If no data to export
        """
        if self.sam2_data is None:
            raise ValueError("No SAM2 data loaded. Call load_and_validate_json() first.")
        
        logger.info("Starting CSV export process")
        start_time = datetime.now()
        
        # Generate rows from nested JSON
        rows = self._generate_csv_rows(experiment_filter)
        
        if not rows:
            raise ValueError("No data to export. Check experiment filter or JSON content.")
        
        # Create DataFrame with proper schema
        df = pd.DataFrame(rows, columns=CSV_COLUMNS)
        
        # Sort by video_id (well) for better organization
        if 'video_id' in df.columns and not df.empty:
            df = df.sort_values('video_id').reset_index(drop=True)
            logger.info(f"Sorted {len(df)} rows by video_id (well)")
        
        # Validate schema
        self._validate_csv_schema(df)
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        duration = datetime.now() - start_time
        logger.info(f"CSV export completed: {len(df)} rows in {duration.total_seconds():.2f} seconds")
        logger.info(f"Output written to: {output_path}")
        
        return df
    
    def _extract_qc_flags_for_snip(self, snip_id: str) -> str:
        """
        Extract SAM2 QC flags from flag_overview compiled summary.
        
        Uses the flag_overview section which compiles all flagged snip_ids by flag type,
        making this a simple and efficient lookup operation.
        
        Args:
            snip_id: Snip ID to check for flags (e.g., "20250529_30hpf_ctrl_atf6_A01_e01_t0000")
            
        Returns:
            Comma-separated string of flag types, or empty string if no flags
            
        Example:
            "HIGH_SEGMENTATION_VAR_SNIP,MASK_ON_EDGE"
        """
        flag_overview = self.sam2_data.get("flags", {}).get("flag_overview", {})
        
        # Define which flag types we care about at snip level (from 05_sam2_qc_analysis.py)
        SNIP_LEVEL_FLAGS = [
            "HIGH_SEGMENTATION_VAR_SNIP",  # High area variance between frames
            "MASK_ON_EDGE",                # Mask touches image edges
            "LARGE_MASK",                  # Abnormally large segmentation
            "SMALL_MASK",                  # Abnormally small segmentation
            "DISCONTINUOUS_MASK"           # Fragmented segmentation
            # Note: OVERLAPPING_MASKS and DETECTION_FAILURE are image-level flags
        ]
        
        flags_for_snip = []
        for flag_type in SNIP_LEVEL_FLAGS:
            if flag_type in flag_overview:
                snip_ids_with_flag = flag_overview[flag_type].get("snip_ids", [])
                if snip_id in snip_ids_with_flag:
                    flags_for_snip.append(flag_type)
        
        return ",".join(flags_for_snip)
    
    def _generate_csv_rows(self, experiment_filter: Optional[List[str]] = None) -> List[List[Any]]:
        """
        Generate CSV rows from nested SAM2 JSON structure.
        
        Args:
            experiment_filter: List of experiment IDs to include
            
        Returns:
            List of rows, each row is a list of values matching CSV_COLUMNS
        """
        rows = []
        experiments = self.sam2_data.get('experiments', {})
        
        # Filter experiments if specified
        if experiment_filter:
            experiments = {k: v for k, v in experiments.items() if k in experiment_filter}
            logger.info(f"Filtering to experiments: {list(experiments.keys())}")
        
        # Load Build01 metadata for each experiment to filter valid wells
        for exp_id in experiments.keys():
            self.load_build01_metadata(exp_id)
        
        total_snips = 0
        
        # Pre-calculate total work for progress tracking
        total_images = sum(
            len(video_data.get('image_ids', {}))
            for exp_data in experiments.values() 
            for video_data in exp_data.get('videos', {}).values()
        )
        processed_images = 0
        
        logger.info(f"Processing {total_images} images across {len(experiments)} experiments")
        
        for exp_id, exp_data in experiments.items():
            logger.debug(f"Processing experiment: {exp_id}")
            
            videos = exp_data.get('videos', {})
            for video_id, video_data in videos.items():
                logger.debug(f"Processing video: {video_id}")
                
                # Filter wells based on Build01 metadata availability
                if self.build01_wells:
                    # Extract well name from video_id (e.g., "A01" from "20250529_30hpf_ctrl_atf6_A01")
                    well_name = video_id.split('_')[-1] if '_' in video_id else video_id
                    if well_name not in self.build01_wells:
                        logger.info(f"Skipping well {well_name} (video: {video_id}) - not found in Build01 metadata")
                        continue
                
                # Get seed frame info
                seed_frame_info = video_data.get('seed_frame_info', {})
                seed_frame_id = seed_frame_info.get('seed_frame')
                
                image_ids = video_data.get('image_ids', {})
                # Ensure deterministic temporal order when dict
                if isinstance(image_ids, dict):
                    iter_items = [(iid, image_ids[iid]) for iid in sorted(image_ids.keys())]
                else:
                    # legacy list form
                    iter_items = [(iid, {}) for iid in image_ids]

                for image_id, image_data in iter_items:
                    processed_images += 1
                    
                    # Progress tracking (log every 100 images or 10%)
                    if processed_images % max(1, total_images // 10) == 0 or processed_images % 100 == 0:
                        progress_pct = (processed_images / total_images) * 100 if total_images > 0 else 0
                        logger.info(f"Progress: {processed_images}/{total_images} images ({progress_pct:.1f}%)")
                    
                    logger.debug(f"Processing image: {image_id}")
                    
                    frame_index = image_data.get('frame_index', 0)
                    is_seed_frame = (image_id == seed_frame_id)
                    
                    embryos = image_data.get('embryos', {})
                    for embryo_id, embryo_data in embryos.items():
                        try:
                            # Extract core data with validation
                            snip_id = embryo_data.get('snip_id')
                            if not snip_id:
                                logger.warning(f"Missing snip_id for {embryo_id} in {image_id}")
                                continue
                            
                            # Segmentation data with validation
                            seg_data = embryo_data.get('segmentation', {})
                            if not seg_data:
                                logger.warning(f"Missing segmentation data for {embryo_id} in {image_id}")
                                continue
                                
                            area_px = seg_data.get('area', 0.0)
                            bbox = seg_data.get('bbox', [0, 0, 0, 0])
                            mask_confidence = embryo_data.get('mask_confidence', 0.0)
                            mask_rle = None
                            mask_height_px = None
                            mask_width_px = None
                            if isinstance(seg_data, dict):
                                counts = seg_data.get('counts') or seg_data.get('rle')
                                if counts is not None:
                                    mask_rle = counts if isinstance(counts, str) else json.dumps(counts)
                                size = seg_data.get('size') or seg_data.get('dimensions')
                                if isinstance(size, (list, tuple)) and len(size) >= 2:
                                    try:
                                        mask_height_px = int(size[0])
                                        mask_width_px = int(size[1])
                                    except (TypeError, ValueError):
                                        mask_height_px = size[0]
                                        mask_width_px = size[1]
                            
                            # Validate bbox format
                            if not isinstance(bbox, list) or len(bbox) != 4:
                                logger.warning(f"Invalid bbox format for {embryo_id} in {image_id}: {bbox}")
                                bbox = [0, 0, 0, 0]
                            
                            # Parse IDs using existing utilities with error handling
                            try:
                                experiment_id = extract_experiment_id(image_id)
                                video_id_parsed = extract_video_id(image_id)
                            except Exception as e:
                                logger.error(f"Failed to parse IDs for {image_id}: {e}")
                                continue
                            
                            # Generate mask file path
                            exported_mask_path = self._generate_mask_path(image_id, embryos)
                            
                            # Extract raw metadata from enhanced schema
                            raw_image_data = image_data.get('raw_image_data_info', {}) if isinstance(image_data, dict) else {}
                            
                            exp_metadata: Dict[str, Any] = {}
                            video_metadata: Dict[str, Any] = {}
                            image_metadata: Dict[str, Any] = {}
                            if self.experiment_metadata:
                                experiments_meta = self.experiment_metadata.get('experiments', {})
                                exp_metadata = experiments_meta.get(experiment_id, {}) if isinstance(experiments_meta, dict) else {}
                                if isinstance(exp_metadata, dict):
                                    videos_meta = exp_metadata.get('videos', {})
                                    video_metadata = videos_meta.get(video_id_parsed, {}) if isinstance(videos_meta, dict) else {}
                                    if isinstance(video_metadata, dict):
                                        image_ids_meta = video_metadata.get('image_ids', {})
                                        image_metadata = image_ids_meta.get(image_id, {}) if isinstance(image_ids_meta, dict) else {}
                            
                            if not raw_image_data and isinstance(image_metadata, dict):
                                raw_image_data = image_metadata.get('raw_image_data_info', {}) or {}
                            
                            if not isinstance(video_metadata, dict):
                                video_metadata = {}
                            
                            # Well-level metadata from experiment metadata (not SAM2 video_data)
                            well_metadata = {
                                'medium': video_metadata.get('medium', video_data.get('medium')),
                                'genotype': video_metadata.get('genotype', video_data.get('genotype')), 
                                'chem_perturbation': video_metadata.get('chem_perturbation', video_data.get('chem_perturbation')),
                                'start_age_hpf': video_metadata.get('start_age_hpf', video_data.get('start_age_hpf')),
                                'embryos_per_well': video_metadata.get('embryos_per_well', video_data.get('embryos_per_well')),
                                'temperature': video_metadata.get('temperature', video_data.get('temperature')),
                                'well_qc_flag': video_metadata.get('well_qc_flag', video_data.get('well_qc_flag'))
                            }

                            # Prefer processed image path; fall back to raw stitched or inferred location
                            image_path = None
                            if isinstance(image_metadata, dict):
                                image_path = image_metadata.get('processed_image_path') or image_metadata.get('raw_stitch_image_path')

                            if not image_path:
                                processed_dir = None
                                if isinstance(image_metadata, dict):
                                    processed_dir = image_metadata.get('processed_jpg_images_dir')
                                if not processed_dir and isinstance(video_metadata, dict):
                                    processed_dir = video_metadata.get('processed_jpg_images_dir')
                                if not processed_dir:
                                    processed_dir = video_data.get('processed_jpg_images_dir')
                                if processed_dir:
                                    suffix = None
                                    if isinstance(image_metadata, dict):
                                        proc_path = image_metadata.get('processed_image_path')
                                        raw_path = image_metadata.get('raw_stitch_image_path')
                                        if proc_path:
                                            suffix = Path(proc_path).suffix
                                        elif raw_path:
                                            suffix = Path(raw_path).suffix
                                    if not suffix:
                                        suffix = ".jpg"
                                    image_path = str(Path(processed_dir) / f"{image_id}{suffix}")
                            
                        except Exception as e:
                            logger.error(f"Error processing {embryo_id} in {image_id}: {e}")
                            continue
                        
                        # Extract SAM2 QC flags for this snip (Refactor-011-B)
                        sam2_qc_flags = self._extract_qc_flags_for_snip(snip_id)
                        
                        # Create row with enhanced metadata
                        row = [
                            image_id,                    # image_id
                            embryo_id,                   # embryo_id  
                            snip_id,                     # snip_id
                            frame_index,                 # frame_index
                            area_px,                     # area_px
                            bbox[0],                     # bbox_x_min
                            bbox[1],                     # bbox_y_min
                            bbox[2],                     # bbox_x_max
                            bbox[3],                     # bbox_y_max
                            mask_confidence,             # mask_confidence
                            mask_rle,                    # mask_rle
                            mask_height_px,              # mask_height_px
                            mask_width_px,               # mask_width_px
                            exported_mask_path,          # exported_mask_path
                            image_path,                  # image_path
                            experiment_id,               # experiment_id
                            video_id_parsed,             # video_id
                            is_seed_frame,               # is_seed_frame
                            
                            # Raw image metadata (original column names)
                            raw_image_data.get('Height (um)'),
                            raw_image_data.get('Height (px)'), 
                            raw_image_data.get('Width (um)'),
                            raw_image_data.get('Width (px)'),
                            # Accept either 'BF Channel' or fallback to legacy 'Channel'
                            (raw_image_data.get('BF Channel') if raw_image_data.get('BF Channel') is not None else raw_image_data.get('Channel')),
                            raw_image_data.get('Objective'),
                            raw_image_data.get('Time (s)'),
                            raw_image_data.get('Time Rel (s)'),
                            
                            # Raw image metadata (code-friendly aliases)
                            raw_image_data.get('height_um'),
                            raw_image_data.get('height_px'),
                            raw_image_data.get('width_um'), 
                            raw_image_data.get('width_px'),
                            raw_image_data.get('bf_channel'),
                            raw_image_data.get('objective'),
                            raw_image_data.get('raw_time_s'),
                            raw_image_data.get('relative_time_s'),
                            raw_image_data.get('microscope'),
                            raw_image_data.get('nd2_series_num'),
                            
                            # Well-level metadata
                            well_metadata['medium'],
                            well_metadata['genotype'],
                            well_metadata['chem_perturbation'],
                            well_metadata['start_age_hpf'],
                            well_metadata['embryos_per_well'],
                            well_metadata['temperature'],
                            well_metadata['well_qc_flag'],

                            # Build03 compatibility columns (preserve original time indexing from image_id)
                            video_data.get('well_id'),              # well
                            extract_frame_number(image_id),         # time_int
                            f"T{extract_frame_number(image_id):04d}",  # time_string

                            # SAM2 QC flags (Refactor-011-B)
                            sam2_qc_flags
                        ]
                        
                        rows.append(row)
                        total_snips += 1
        
        logger.info(f"Generated {total_snips} CSV rows from {len(experiments)} experiments")
        return rows
    
    def _generate_mask_path(self, image_id: str, embryos: Dict[str, Any]) -> str:
        """
        Generate exported mask file path following naming convention.

        Convention: {image_id}_masks_emnum_{embryo_count}.png
        
        Args:
            image_id: The image ID
            embryos: Dictionary of embryos for this image
            
        Returns:
            Generated mask file path (just filename, not full path)
        """
        embryo_count = len(embryos)
        mask_filename = f"{image_id}_masks_emnum_{embryo_count}.png"
        
        # Always return just the filename - path resolution handled in validation
        return mask_filename

    def _validate_csv_schema(self, df: pd.DataFrame) -> None:
        """
        Validate that DataFrame matches expected CSV schema.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If schema doesn't match
        """
        # Check columns
        if list(df.columns) != CSV_COLUMNS:
            raise ValueError(f"CSV columns don't match expected schema.\nExpected: {CSV_COLUMNS}\nActual: {list(df.columns)}")
        
        # Check for required fields
        required_non_null = ['image_id', 'embryo_id', 'snip_id']
        for col in required_non_null:
            if df[col].isnull().any():
                raise ValueError(f"Required column '{col}' contains null values")
        
        # Validate data types
        numeric_cols = ['frame_index', 'area_px', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'mask_confidence']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Column '{col}' is not numeric type: {df[col].dtype}")
        
        # Validate boolean column
        if not df['is_seed_frame'].dtype == bool:
            logger.warning(f"Column 'is_seed_frame' is not boolean type: {df['is_seed_frame'].dtype}")
        
        logger.info(f"CSV schema validation passed: {len(df)} rows × {len(df.columns)} columns")
    
    def validate_mask_files(self, df: pd.DataFrame) -> None:
        """
        Validate that exported mask files exist on filesystem.
        
        Args:
            df: DataFrame with exported_mask_path column
            
        Raises:
            FileNotFoundError: If critical mask files are missing
        """
        if self.masks_dir is None:
            logger.warning("No masks directory provided, skipping file validation")
            return
            
        logger.info(f"Validating mask file paths in {self.masks_dir}")
        
        # Get unique mask paths
        unique_mask_paths = df['exported_mask_path'].unique()
        
        missing_files = []
        existing_files = []
        
        for mask_path in unique_mask_paths:
            full_path = Path(mask_path)
            
            # Handle both absolute and relative paths
            if not full_path.is_absolute():
                full_path = self.masks_dir / mask_path
            
            if full_path.exists():
                existing_files.append(str(full_path))
            else:
                missing_files.append(str(full_path))
        
        # Report results
        logger.info(f"Mask file validation: {len(existing_files)} found, {len(missing_files)} missing")
        
        if missing_files:
            logger.warning(f"Missing mask files ({len(missing_files)}):")
            for missing_file in missing_files[:10]:  # Show first 10
                logger.warning(f"  - {missing_file}")
            if len(missing_files) > 10:
                logger.warning(f"  ... and {len(missing_files) - 10} more")
        
        # Only raise error if more than 50% of files are missing (indicating systematic issue)
        if len(missing_files) > len(unique_mask_paths) * 0.5:
            raise FileNotFoundError(
                f"Critical: {len(missing_files)}/{len(unique_mask_paths)} mask files missing. "
                f"Check masks directory path: {self.masks_dir}"
            )
        elif missing_files:
            logger.warning(f"Some mask files missing but continuing ({len(missing_files)} missing)")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export SAM2 metadata to CSV for legacy build script integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with per-experiment metadata
    python export_sam2_metadata_to_csv.py annotations.json -o output.csv --masks-dir ./masks --metadata-json experiment_metadata_20240418.json

    # Filter specific experiment  
    python export_sam2_metadata_to_csv.py annotations.json -o output.csv --masks-dir ./masks --experiment-filter 20240418 --metadata-json experiment_metadata_20240418.json
        """
    )
    
    parser.add_argument(
        'input_json',
        type=Path,
        help='Path to GroundedSam2Annotations.json file'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        required=True,
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--masks-dir',
        type=Path,
        help='Directory containing exported mask PNG files (for path validation)'
    )
    
    parser.add_argument(
        '--metadata-json',
        type=Path,
        help='Path to per-experiment metadata JSON file (e.g., experiment_metadata_{exp}.json)'
    )
    
    parser.add_argument(
        '--experiment-filter',
        nargs='+',
        help='Filter to specific experiment IDs (e.g., 20240418)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize exporter
        exporter = SAM2MetadataExporter(args.input_json, args.masks_dir, args.metadata_json)
        
        # Load and validate
        exporter.load_and_validate_json()
        exporter.load_experiment_metadata()
        
        # Export to CSV
        df = exporter.export_to_csv(args.output, args.experiment_filter)
        
        # Validate mask file paths if masks_dir provided
        if args.masks_dir:
            exporter.validate_mask_files(df)
        
        # Print summary
        print(f"✅ Successfully exported {len(df)} rows to {args.output}")
        if args.masks_dir:
            print(f"🔍 Validated mask file paths in {args.masks_dir}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
