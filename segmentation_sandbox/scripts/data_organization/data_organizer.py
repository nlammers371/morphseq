"""

## EntityIDTracker Integration:
EntityIDTracker serves as a PURE CONTAINER for entity validation and tracking.
- **Role**: Container for entity data, validator for entity relationships
- **Embedded Approach**: Tracker embedded in metadata JSON (not separate files)
- **Pipeline Context**: Since embedded, pipeline step is implicit (Module 0 = Module 0 entities)
- **Helper Functions**: Format-specific operations handled by static methods
- **Key Principle**: Keep EntityIDTracker simple and containerized

## File Structure Transformation:
INPUT: directory_with_experiments/20240411/A01_t0000_ch00_stitch.png
OUTPUT: raw_data_organized/20240411/images/20240411_A01/0000.jpg (NO 't' prefix on disk!)

## Key Implementation Points:

1. **Filename Parsing**: Use regex to extract well_id and frame from stitch files:
   - 'A01_t0000_ch00_stitch.png' → ('A01', '0000')  
   - 'B02_0123_stitch.tif' → ('B02', '0123')  # May not have 't' prefix

2. **Image Organization**: 
   - Group stitch files by well_id
   - Convert to JPEG (try pyvips, fallback to OpenCV)
   - Save as simple frame numbers: 0000.jpg, 0001.jpg (NO 't' prefix!)

3. **Video Creation**:
   - Create MP4 from JPEG sequences using OpenCV
   - Add frame number overlay to each frame
   - 5 FPS, mp4v codec

4. **Metadata Generation**:
   - Scan organized structure (don't track during processing)
   - Create image_ids WITH 't' prefix for JSON: "20240411_A01_t0000"
   - Structure: {"experiments": {exp_id: {"videos": {video_id: {...}}}}}

5. **Critical Naming Convention**:
   - DISK files: 0000.jpg (no 't' prefix)
   - JSON image_ids: "20240411_A01_t0000" (with 't' prefix)
   - This differentiation is crucial for later modules!

## Required Methods:
- process_experiments(source_dir, output_dir, experiment_names=None)
- find_experiment_directories(base_dir) 
- parse_stitch_filename(filename) → (well_id, frame)
- organize_experiment(exp_dir, output_dir, experiment_id)
- process_well(image_files, exp_output_dir, video_id)
- convert_to_jpeg(source_path, target_path, quality=90)
- create_video_from_jpegs(jpeg_paths, video_path, video_id)
- scan_organized_experiments(raw_data_dir) → metadata_dict
- scan_experiment_directory(exp_dir, experiment_id)
- scan_video_directory(video_id, video_path, images_dir)
- get_image_path_from_id(image_id, images_dir) → Path
- get_images_for_detection(metadata, experiment_ids=None) → List[Dict]

## Import Structure:
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import re
import json
import shutil
from datetime import datetime
from collections import defaultdict
import cv2
# Try pyvips import with fallback

from ..utils.parsing_utils import parse_entity_id, extract_experiment_id
from ..utils.base_file_handler import BaseFileHandler
from ..utils.entity_id_tracker import EntityIDTracker

## Dependencies:
- OpenCV (cv2) for video creation and image conversion fallback
- Optional: pyvips for better image conversion
- Uses Module 0 utilities for consistent parsing

## Success Criteria:
- Handles complex experiment IDs correctly
- Creates organized structure matching original 01_prepare_videos.py
- Generates lightweight metadata by scanning (not tracking during processing)
- Ready for downstream GDINO detection module
- Maintains critical disk vs JSON naming convention

IMPLEMENT the complete DataOrganizer class below this comment:
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple
import re
import json
import shutil
import sys
from datetime import datetime
from collections import defaultdict
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import os
import logging

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Try to import pyvips for fast image conversion
try:
    import pyvips
    PYVIPS_AVAILABLE = True
except ImportError:
    PYVIPS_AVAILABLE = False

# Import utilities
try:
    from ..utils.parsing_utils import parse_entity_id, extract_experiment_id, build_image_id
    from ..utils.entity_id_tracker import EntityIDTracker
except ImportError:
    # Fallback for when module is imported from different context
    import sys
    from pathlib import Path
    scripts_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(scripts_dir))
    from utils.parsing_utils import parse_entity_id, extract_experiment_id
    from utils.entity_id_tracker import EntityIDTracker

# Import new video generation utilities
try:
    from ..utils.video_generation import VideoGenerator, VideoConfig
except ImportError:
    # Fallback for when module is imported from different context
    try:
        from utils.video_generation import VideoGenerator, VideoConfig
    except ImportError:
        # If video generation is not available, we'll handle it in the method
        VideoGenerator = None
        VideoConfig = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataOrganizer:
    """
    Organizes raw stitched images into a standard structure and creates videos/metadata.
    
    Performance Notes:
    - Uses pyvips when available for faster image I/O (install with: pip install pyvips)
    - Falls back to OpenCV when pyvips is not available (slower)
    - Parallel processing with ThreadPoolExecutor for JPEG conversion
    - Smart skip logic to avoid reprocessing existing files
    """
    
    @staticmethod
    def convert_to_json_serializable(obj):
        """Convert pandas/numpy types to JSON serializable Python types"""
        import numpy as np
        
        if obj is None:
            return None
        elif isinstance(obj, dict):
            return {k: DataOrganizer.convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DataOrganizer.convert_to_json_serializable(v) for v in obj]
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'item'):  # numpy/pandas types
            return obj.item()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    @staticmethod
    def load_legacy_metadata_csv(experiment_name):
        """Load raw image metadata from legacy build scripts.

        Resolution order (first existing is used):
        1) Repo-relative: <repo_root>/metadata/built_metadata_files/{experiment_name}_metadata.csv
        2) Env var MORPHSEQ_METADATA_ROOT: $MORPHSEQ_METADATA_ROOT/{experiment_name}_metadata.csv
        3) Legacy absolute path (lab NFS) as last resort
        """
        candidates = []
        # 1) repo-relative
        try:
            repo_root = Path(__file__).resolve().parents[3]
            candidates.append(repo_root / "metadata" / "built_metadata_files" / f"{experiment_name}_metadata.csv")
        except Exception:
            pass
        # 2) env var
        env_root = os.getenv("MORPHSEQ_METADATA_ROOT")
        if env_root:
            candidates.append(Path(env_root) / f"{experiment_name}_metadata.csv")
        # 3) legacy absolute
        candidates.append(Path(f"/net/trapnell/vol1/home/nlammers/projects/data/morphseq/metadata/built_metadata_files/{experiment_name}_metadata.csv"))

        for cand in candidates:
            if cand and Path(cand).exists():
                try:
                    df = pd.read_csv(cand)
                    # logger.info(f"Loaded legacy metadata CSV: {cand}")
                    return df
                except Exception as e:
                    logger.warning(f"Failed to load CSV {cand}: {e}")
                    return None

        logger.warning(f"Legacy metadata CSV not found for {experiment_name} in any candidate path")
        return None
    
    @staticmethod
    def parse_video_id_for_metadata(video_id):
        """Extract experiment_name and well_id from video_id using existing parsing utils"""
        try:
            # Try existing parsing utils first
            # print(video_id)
            experiment_id = extract_experiment_id(video_id)
            # print(experiment_id)
            logger.debug(f"🔍 DEBUG: extract_experiment_id('{video_id}') returned: '{experiment_id}'")
            
            # Extract well_id - should be last part after splitting by '_'
            parts = video_id.split('_')
            well_id = parts[-1] if len(parts) > 0 else "unknown"
            
            logger.debug(f"Parsed video_id '{video_id}' as experiment_id='{experiment_id}', well_id='{well_id}'")
            return experiment_id, well_id, "ch00"
            
        except Exception as e:
            logger.warning(f"Error using extract_experiment_id: {e}")
            logger.debug(f"Falling back to manual parsing...")
            
            # Fallback to manual parsing
            parts = video_id.split('_')
            if len(parts) >= 2:
                well_id = parts[-1]  # Last part is always well_id (A01)
                experiment_name = '_'.join(parts[:-1])  # Everything else is experiment_name
                #logger.debug(f"Manual parse: experiment_name='{experiment_name}', well_id='{well_id}'")
                return experiment_name, well_id, "ch00"
            else:
                logger.error(f"Could not parse video_id: {video_id}")
                return video_id, "unknown", "ch00"

    @staticmethod
    def enhance_video_metadata_with_csv(video_data, csv_df, experiment_name, well_id, images_dir=None, source_root=None):
        """Transform video metadata to enhanced schema"""
        logger.debug(f"Starting schema enhancement for {experiment_name}_{well_id}")
        
        if csv_df is None:
            logger.warning(f"No CSV data available for enhancement")
            return video_data
        
        # 1. Add well-level metadata (constant per well)
        logger.debug(f"Looking for well rows matching '{well_id}' in CSV...")
        
        # Try different matching strategies
        if 'well_id' in csv_df.columns:
            well_rows = csv_df[csv_df['well_id'].str.contains(well_id, na=False)]
            logger.debug(f"Strategy 1 (contains {well_id}): Found {len(well_rows)} rows")
            
            if well_rows.empty:
                # Try exact match
                well_rows = csv_df[csv_df['well_id'] == well_id]
                logger.debug(f"Strategy 2 (exact {well_id}): Found {len(well_rows)} rows")
                
        elif 'well' in csv_df.columns:
            well_rows = csv_df[csv_df['well'] == well_id]
            logger.debug(f"Strategy 3 (well column {well_id}): Found {len(well_rows)} rows")
        else:
            well_rows = pd.DataFrame()  # Empty
            logger.warning(f"No well_id or well column found")
            
        if not well_rows.empty:
            first_row = well_rows.iloc[0]
            logger.debug(f"Adding well metadata from first matching row")
            
            # Convert pandas values to native Python types for JSON serialization
            def to_python_type(val):
                if pd.isna(val):
                    return None
                elif hasattr(val, 'item'):  # numpy types
                    return val.item()
                else:
                    return val
            
            video_data.update({
                'well_id': well_id,
                # Store a relative path hint; actual resolution happens at load-time
                'source_well_metadata_csv': f"metadata/built_metadata_files/{experiment_name}_metadata.csv",
                'medium': to_python_type(first_row.get('medium', None)),
                'genotype': to_python_type(first_row.get('genotype', None)),
                'chem_perturbation': to_python_type(first_row.get('chem_perturbation', None)),
                'start_age_hpf': to_python_type(first_row.get('start_age_hpf', None)),
                'embryos_per_well': to_python_type(first_row.get('embryos_per_well', None)),
                'temperature': to_python_type(first_row.get('temperature', None)),
                'well_qc_flag': to_python_type(first_row.get('well_qc_flag', None))
            })
        else:
            logger.warning(f"No matching rows found for well_id '{well_id}'")
        
        # 2. Transform image_ids from list to dictionary
        old_image_ids = video_data.get('image_ids', [])
        logger.debug(f"Converting {len(old_image_ids)} image_ids from list to dictionary")
        new_image_ids = {}
        
        for i, image_id in enumerate(old_image_ids):
            # Extract time_int from image_id 
            time_int = 0  # default
            if '_t' in image_id:
                try:
                    time_int = int(image_id.split('_t')[-1])
                except (ValueError, IndexError):
                    time_int = i  # fallback to frame index
            else:
                time_int = i
            
            logger.debug(f"Processing image_id '{image_id}' with time_int={time_int}")
            
            # Find matching CSV row for this well_id + time_int
            matching_rows = csv_df[
                (csv_df['well_id'].str.contains(well_id, na=False)) & 
                (csv_df['time_int'] == time_int)
            ]
            
            image_info = {
                'frame_index': i,
                'raw_image_data_info': {},
                'raw_stitch_image_path': None,
                'processed_image_path': None
            }
            
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                logger.debug(f"Found CSV data for image {image_id}, time_int={time_int}")
                # Convert pandas values to native Python types for JSON serialization
                def to_python_type(val):
                    if pd.isna(val):
                        return None
                    elif hasattr(val, 'item'):  # numpy types
                        return val.item()
                    else:
                        return val
                
                # Normalize channel column name from CSV (supports both 'BF Channel' and 'Channel')
                ch_val = row.get('BF Channel', None)
                if pd.isna(ch_val) if 'BF Channel' in row else True:
                    ch_val = row.get('Channel', None)

                image_info['raw_image_data_info'] = {
                    # Original CSV column names (preserves data lineage)
                    'Height (um)': to_python_type(row.get('Height (um)', None)),
                    'Height (px)': to_python_type(row.get('Height (px)', None)),
                    'Width (um)': to_python_type(row.get('Width (um)', None)),
                    'Width (px)': to_python_type(row.get('Width (px)', None)),
                    'BF Channel': to_python_type(ch_val),
                    'Objective': to_python_type(row.get('Objective', None)),
                    'Time (s)': to_python_type(row.get('Time (s)', None)),
                    'Time Rel (s)': to_python_type(row.get('Time Rel (s)', None)),

                    # Code-friendly aliases
                    'height_um': to_python_type(row.get('Height (um)', None)),
                    'height_px': to_python_type(row.get('Height (px)', None)),
                    'width_um': to_python_type(row.get('Width (um)', None)),
                    'width_px': to_python_type(row.get('Width (px)', None)),
                    'bf_channel': to_python_type(ch_val),
                    'objective': to_python_type(row.get('Objective', None)),
                    'raw_time_s': to_python_type(row.get('Time (s)', None)),
                    'relative_time_s': to_python_type(row.get('Time Rel (s)', None)),

                    # Additional metadata
                    'experiment_date': to_python_type(row.get('experiment_date', None)),
                    'time_string': to_python_type(row.get('time_string', None)),
                    'microscope': to_python_type(row.get('microscope', None)),
                    'nd2_series_num': to_python_type(row.get('nd2_series_num', None)),
                }
            else:
                logger.warning(f"No CSV data found for image {image_id}, time_int={time_int}")
            
            # Add processed image path if images_dir provided
            if images_dir:
                processed_path = Path(images_dir) / f"{image_id}.jpg"
                image_info['processed_image_path'] = str(processed_path)
            
            # Add raw stitch image path if source_root provided
            if source_root:
                # Build path: source_root/built_image_data/stitched_FF_images/experiment/well_tXXXX.jpg
                raw_stitch_path = Path(source_root) / 'built_image_data' / 'stitched_FF_images' / experiment_name / f"{well_id}_t{time_int:04d}.jpg"
                image_info['raw_stitch_image_path'] = str(raw_stitch_path)
            
            new_image_ids[image_id] = image_info
            
            # Process all images
        
        # 3. Replace image_ids list with dictionary
        video_data['image_ids'] = new_image_ids
        
        # 4. Update processed vs raw dimension clarity
        if 'image_size' in video_data:
            video_data['processed_image_size_px'] = video_data.pop('image_size')
        
        # logger.info(f"Schema enhancement complete. image_ids type: {type(video_data['image_ids'])}")
        return video_data

    @staticmethod
    def validate_entity_tracking_completeness(metadata, verbose=False):
        """
        Validate that metadata contains complete entity tracking information.
        
        Args:
            metadata: Metadata dictionary to validate
            verbose: Enable verbose output
            
        Returns:
            bool: True if entity tracking is complete, False otherwise
        """
        if not isinstance(metadata, dict):
            return False
            
        # Check for entity_tracker section
        if 'entity_tracker' not in metadata:
            if verbose:
                print("   ❌ Missing entity_tracker section")
            return False
            
        entity_tracker = metadata['entity_tracker']
        
        # Check for required fields in entity tracker
        required_fields = ['entities', 'summary', 'pipeline_step']
        for field in required_fields:
            if field not in entity_tracker:
                if verbose:
                    print(f"   ❌ Missing entity_tracker.{field}")
                return False
                
        # Check if entities section has the expected entity types
        entities = entity_tracker.get('entities', {})
        expected_types = ['experiments', 'videos', 'images']  # Don't require embryos/snips for Module 0
        
        for entity_type in expected_types:
            if entity_type not in entities:
                if verbose:
                    print(f"   ❌ Missing entity type: {entity_type}")
                return False
                
        # Validate that we have actual entities (not just empty lists)
        if len(entities.get('experiments', [])) == 0:
            if verbose:
                print("   ❌ No experiments found in entity tracker")
            return False
            
        if verbose:
            print("   ✅ Entity tracking is complete")
        return True
    
    @staticmethod
    def process_experiments(source_dir, output_dir, experiment_names=None, verbose=True, overwrite=False, force_raw_data=False):
        """
        Organize experiments and create videos/metadata with autosave functionality.
        
        Args:
            source_dir: Source directory containing experiments
            output_dir: Output directory for organized data
            experiment_names: Optional list of specific experiments to process
            verbose: Enable verbose output
            overwrite: Whether to overwrite existing processed experiments
        """
        raw_data_dir = Path(output_dir) / "raw_data_organized"
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"📂 Source directory: {source_dir}")
            print(f"📂 Output directory: {raw_data_dir}")
            print(f"🔄 Overwrite mode: {overwrite}")

        # Load existing per-experiment metadata to check what's already processed
        existing_metadata = DataOrganizer.load_existing_per_experiment_metadata(
            raw_data_dir, experiment_names, verbose
        )
        if verbose and existing_metadata.get('experiments'):
            existing_count = len(existing_metadata.get('experiments', {}))
            print(f"📖 Found existing metadata for {existing_count} experiments")

        # Find experiments to process
        if experiment_names:
            experiment_dirs = [Path(source_dir) / name for name in experiment_names if (Path(source_dir) / name).is_dir()]
            if verbose:
                pass
                # print(f"🔍 Processing specified experiments: {experiment_names}")
        else:
            experiment_dirs = DataOrganizer.find_experiment_directories(Path(source_dir))
            if verbose:
                pass
                # print(f"🔍 Found {len(experiment_dirs)} experiments in source directory")

        if not experiment_dirs:
            print("❌ No experiments found to process!")
            return

        # Filter experiments based on existing metadata and overwrite setting
        experiments_to_process = []
        experiments_skipped = []
        
        # Check if entity tracking is complete in existing metadata
        entity_tracking_complete = DataOrganizer.validate_entity_tracking_completeness(
            existing_metadata, verbose=False
        )
        
        for exp_dir in experiment_dirs:
            experiment_id = exp_dir.name
            
            # Check if already processed AND entity tracking is complete
            already_processed = experiment_id in existing_metadata.get('experiments', {})
            needs_processing = not already_processed or not entity_tracking_complete

            if already_processed and entity_tracking_complete and not overwrite and not force_raw_data:
                experiments_skipped.append(experiment_id)
                if verbose:
                    print(f"⏭️  Skipping already processed experiment: {experiment_id}")
            elif already_processed and force_raw_data:
                experiments_to_process.append(exp_dir)
                if verbose:
                    print(f"🔄 Force-regenerating raw data for: {experiment_id}")
            elif already_processed and not entity_tracking_complete:
                experiments_to_process.append(exp_dir)
                if verbose:
                    print(f"🔄 Re-processing experiment (missing entity tracking): {experiment_id}")
            elif not already_processed:
                experiments_to_process.append(exp_dir)
                if verbose:
                    print(f"🆕 Processing new experiment: {experiment_id}")
            else:
                experiments_to_process.append(exp_dir)
                if already_processed and overwrite:
                    if verbose:
                        print(f"🔄 Will overwrite experiment: {experiment_id}")
                        
        if verbose:
            print(f"\n📊 Processing Summary:")
            print(f"   🎯 To process: {len(experiments_to_process)} experiments")
            print(f"   ⏭️  Skipping: {len(experiments_skipped)} experiments")
            
        if not experiments_to_process:
            if entity_tracking_complete:
                print("✅ All experiments already processed! Use overwrite=True to reprocess.")
                return
            else:
                print("📋 All experiments processed but entity tracking incomplete - fixing...")
                
            # Ensure per-experiment metadata files have complete entity tracking
            if verbose:
                print("📋 Updating per-experiment metadata files with entity tracking...")

            # Re-scan to ensure we have current data
            current_scan = DataOrganizer.scan_organized_experiments(raw_data_dir, experiment_names=experiment_names, verbose=False)

            # Add entity tracker (consistent with normal processing)
            metadata = EntityIDTracker.add_entity_tracker(
                current_scan,
                pipeline_step="module_0_data_organization"
            )

            # Save per-experiment files with entity tracking
            processed_experiments = metadata.get('experiments', {})
            for exp_id in processed_experiments.keys():
                DataOrganizer.save_per_experiment_metadata(raw_data_dir, metadata, exp_id, verbose)

            if verbose:
                exp_count = len(processed_experiments)
                entities = EntityIDTracker.extract_entities(metadata)
                entity_counts = {k: len(v) for k, v in entities.items()}
                print(f"✅ Per-experiment metadata updated with entity tracking ({exp_count} experiments)")
                print(f"📊 Entities: {entity_counts}")
            return

        # Process experiments one by one with incremental saves
        for i, exp_dir in enumerate(experiments_to_process, 1):
            experiment_id = exp_dir.name
            print(f"\n🧪 Processing experiment {i}/{len(experiments_to_process)}: {experiment_id}")
            
            # if verbose:
            #     stitch_count = len(list(exp_dir.glob('*_stitch.*')))
            #     print(f"   Found {stitch_count} stitch files")
                
            # Process this experiment
            DataOrganizer.organize_experiment(exp_dir, raw_data_dir, experiment_id, False, overwrite, force_raw_data)  # Set verbose=False
            
            # Update and save metadata incrementally for robustness
            # if verbose:
            #     print(f"   💾 Updating metadata for {experiment_id}...")
                
            current_metadata = DataOrganizer.scan_organized_experiments(raw_data_dir, experiment_names=experiment_names, verbose=False)
            
            # Add entity tracker to autosave (consistent with final save)
            current_metadata = EntityIDTracker.add_entity_tracker(
                current_metadata, 
                pipeline_step="module_0_data_organization"
            )
            
            # Save metadata after each experiment (autosave)
            try:
                # Save per-experiment file
                DataOrganizer.save_per_experiment_metadata(raw_data_dir, current_metadata, experiment_id, verbose)

                if verbose:
                    print(f"   ✅ Per-experiment metadata saved for {experiment_id}")
            except Exception as e:
                print(f"   ⚠️  Failed to save per-experiment metadata: {e}")

        # Final metadata generation and validation
        print("\n📋 Generating final experiment metadata...")
        final_metadata = DataOrganizer.scan_organized_experiments(raw_data_dir, experiment_names=experiment_names, verbose=True)  # Keep verbose for our debug

        # Add entity tracker (MANDATORY for downstream modules)
        # if verbose:
        #     print("📋 Adding embedded entity tracker...")

        final_metadata = EntityIDTracker.add_entity_tracker(
            final_metadata, 
            pipeline_step="module_0_data_organization"
        )

        if verbose:
            entities = EntityIDTracker.extract_entities(final_metadata)
            entity_counts = {k: len(v) for k, v in entities.items()}
            print(f"✅ Entity tracker embedded: {entity_counts}")

        # Final save with JSON serialization fix
        final_metadata_clean = DataOrganizer.convert_to_json_serializable(final_metadata)

        # Save per-experiment files only
        processed_experiments = final_metadata.get('experiments', {})
        for exp_id in processed_experiments.keys():
            DataOrganizer.save_per_experiment_metadata(raw_data_dir, final_metadata, exp_id, verbose)

        print(f"✅ Complete! Per-experiment metadata files created for {len(processed_experiments)} experiments:")
        for exp_id in processed_experiments.keys():
            per_exp_path = raw_data_dir / exp_id / f"experiment_metadata_{exp_id}.json"
            print(f"   📄 {per_exp_path}")
        
        if verbose:
            exp_count = len(final_metadata.get('experiments', {}))
            video_count = sum(len(exp.get('videos', {})) for exp in final_metadata.get('experiments', {}).values())
            print(f"📊 Final Summary: {exp_count} experiments, {video_count} videos processed")
            print(f"🎯 Processed {len(experiments_to_process)} new/updated experiments")
            print(f"⏭️  Skipped {len(experiments_skipped)} existing experiments")

    @staticmethod
    def find_experiment_directories(base_dir):
        experiments = []
        for potential_dir in Path(base_dir).iterdir():
            if potential_dir.is_dir():
                stitch_files = list(potential_dir.glob('*_stitch.*'))
                if stitch_files:
                    experiments.append(potential_dir)
        return experiments

    @staticmethod
    def parse_stitch_filename(filename):
        well_match = re.search(r'([A-H]\d{2})', filename)
        if not well_match:
            return None
        well_id = well_match.group(1)
        frame_match = re.search(r't?(\d{3,4})', filename)
        if not frame_match:
            return None
        frame = frame_match.group(1)
        
        # Extract channel information
        channel_match = re.search(r'ch(\d{2})', filename)
        channel = int(channel_match.group(1)) if channel_match else 0
        
        return well_id, frame, channel

    @staticmethod
    def organize_experiment(experiment_dir, output_dir, experiment_id, verbose=True, overwrite=False, force_raw_data=False):
        stitch_files = list(Path(experiment_dir).glob('*_stitch.*'))
        wells = defaultdict(list)
        
        if verbose:
            print(f"   📁 Organizing {len(stitch_files)} stitch files...")
            
        for stitch_file in stitch_files:
            result = DataOrganizer.parse_stitch_filename(stitch_file.name)
            if result:
                well_id, frame, channel = result
                wells[well_id].append((stitch_file, frame, channel))
            elif verbose:
                print(f"   ⚠️  Could not parse filename: {stitch_file.name}")
        
        if verbose:
            print(f"   🔬 Found {len(wells)} wells: {list(wells.keys())}")
            
        for well_id, files in wells.items():
            video_id = f"{experiment_id}_{well_id}"
            if verbose:
                print(f"   🎬 Processing well {well_id} with {len(files)} frames...")
            DataOrganizer.process_well(files, Path(output_dir) / experiment_id, video_id, verbose, overwrite, force_raw_data)

    @staticmethod
    def process_well(image_files, exp_output_dir, video_id, verbose=True, overwrite=False, force_raw_data=False):
        images_dir = Path(exp_output_dir) / "images" / video_id
        vids_dir = Path(exp_output_dir) / "vids"
        images_dir.mkdir(parents=True, exist_ok=True)
        vids_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = vids_dir / f"{video_id}.mp4"

        # Skip video only if it exists AND force_raw_data is NOT set
        # Keep overwrite for backward compatibility but force_raw_data is the primary control
        if video_path.exists() and not force_raw_data and not overwrite:
            if verbose:
                print(f"     ⏭️  Video already exists: {video_path.name}")
            return
        
        if verbose:
            print(f"     📸 Converting {len(image_files)} images to JPEG...")
        
        # Parallel JPEG conversion with tqdm progress bar and smart skipping
        jpeg_paths = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            sorted_images = sorted(image_files, key=lambda x: x[1])  # Sort by frame number
            total_images = len(sorted_images)
            
            if TQDM_AVAILABLE and verbose:
                pbar = tqdm(total=total_images, desc=f"{video_id} JPEG conversion", leave=False)
            else:
                pbar = None
            
            # First pass: collect existing files and queue new conversions
            for stitch_path, frame, channel in sorted_images:
                # Build full image_id filename with channel: {video_id}_ch{channel}_tNNNN.jpg
                try:
                    image_id = build_image_id(video_id, int(frame), channel)
                except Exception:
                    # Fallback if frame is not numeric
                    image_id = f"{video_id}_ch{channel:02d}_t{str(frame).zfill(4)}"
                jpeg_filename = f"{image_id}.jpg"
                jpeg_path = images_dir / jpeg_filename

                if jpeg_path.exists() and not force_raw_data and not overwrite:
                    # File exists, skip conversion (overwrite is for metadata only)
                    jpeg_paths.append((frame, jpeg_path))
                    if pbar:
                        pbar.update(1)
                    continue
                
                # Queue for conversion
                future = executor.submit(DataOrganizer.convert_to_jpeg, stitch_path, jpeg_path)
                futures[future] = (frame, jpeg_path)
            
            # Second pass: collect conversion results
            for future in as_completed(futures):
                frame, jpeg_path = futures[future]
                try:
                    future.result()
                    jpeg_paths.append((frame, jpeg_path))
                except Exception as e:
                    print(f"Failed to convert frame {frame}: {e}")
                if pbar:
                    pbar.update(1)
            
            if pbar:
                pbar.close()
        
        # Sort by frame number and extract paths
        jpeg_paths.sort(key=lambda x: x[0])
        sorted_paths = [path for _, path in jpeg_paths]
        
        if verbose:
            # Simplified statistics - avoid complex calculations during processing
            converted_count = len(futures)  # Number of files we actually converted
            existing_count = len(sorted_paths) - converted_count  # Files that already existed
            print(f"     ✅ Processed {len(sorted_paths)} images ({converted_count} converted, {existing_count} existing)")
        
        # Create video (sequential)
        DataOrganizer.create_video_from_jpegs(sorted_paths, video_path, video_id, verbose)

    @staticmethod
    def convert_to_jpeg(source_path, target_path, quality=90):
        """
        Convert image to JPEG using pyvips when available for speed, fallback to OpenCV.
        This matches the performance optimizations from the original 01_prepare_videos.py
        """
        try:
            if PYVIPS_AVAILABLE:
                # Use pyvips for faster processing (matches original implementation)
                img = pyvips.Image.new_from_file(str(source_path), access='sequential')
                
                # Convert to RGB if needed (pyvips handles this automatically)
                if img.bands == 4:  # RGBA
                    img = img[:3]  # Take only RGB channels
                
                # Save as JPEG with specified quality
                img.write_to_file(str(target_path), Q=quality)
            else:
                # Fallback to OpenCV (original fallback implementation)
                image = cv2.imread(str(source_path))
                if image is not None:
                    if len(image.shape) == 3 and image.shape[2] == 4:
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                    cv2.imwrite(str(target_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        except Exception as e:
            print(f"Failed to convert {source_path}: {e}")

    @staticmethod
    def create_video_from_jpegs(jpeg_paths, video_path, video_id, verbose=True):
        """
        Create foundation video using VideoGenerator if available, otherwise use basic OpenCV.
        This replaces the old inline video creation with proper image_id overlay.
        """
        if not jpeg_paths:
            if verbose:
                print("     ❌ No JPEG files to create video from")
            return
        
        if VideoGenerator and VideoConfig:
            # Use advanced video generation if available
            try:
                # Initialize video generator with fast configuration
                video_generator = VideoGenerator(VideoConfig.fast_generation())
                
                # Create foundation video with proper image_id overlay (10% down from top-right)
                success = video_generator.create_foundation_video(
                    jpeg_paths=sorted(jpeg_paths),
                    video_path=video_path,
                    video_id=video_id,
                    verbose=verbose
                )
                
                if not success and verbose:
                    print(f"     ❌ Failed to create video: {video_path.name}")
                elif success and verbose:
                    print(f"     ✅ Foundation video created: {video_path.name}")
                    print(f"        📍 Image IDs positioned 10% down from top-right")
                    print(f"        🎯 Ready for future overlay enhancements")
                return
            except Exception as e:
                if verbose:
                    print(f"     ⚠️  VideoGenerator failed, falling back to basic OpenCV: {e}")
        
        # Fallback to basic OpenCV video creation
        try:
            if not jpeg_paths:
                return
                
            # Read first image to get dimensions
            first_img = cv2.imread(str(jpeg_paths[0]))
            if first_img is None:
                if verbose:
                    print(f"     ❌ Could not read first image: {jpeg_paths[0]}")
                return
                
            height, width = first_img.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 5.0
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                if verbose:
                    print(f"     ❌ Could not open video writer for: {video_path}")
                return
            
            # Write frames
            for jpeg_path in sorted(jpeg_paths):
                img = cv2.imread(str(jpeg_path))
                if img is not None:
                    video_writer.write(img)
            
            video_writer.release()
            
            if verbose:
                print(f"     ✅ Basic video created: {video_path.name}")
                
        except Exception as e:
            if verbose:
                print(f"     ❌ Failed to create video with OpenCV: {e}")
            return

    @staticmethod
    def scan_organized_experiments(raw_data_dir, experiment_names=None, verbose=True):
        metadata = {
            "file_info": {
                "creation_time": datetime.now().isoformat(),
                "script_version": "Module_0_Simplified"
            },
            "experiments": {}
        }
        
        all_experiment_dirs = [d for d in Path(raw_data_dir).iterdir() if d.is_dir()]
        
        # Filter to only requested experiments if specified
        if experiment_names:
            experiment_dirs = [d for d in all_experiment_dirs if d.name in experiment_names]
            if verbose:
                print(f"📂 Scanning {len(experiment_dirs)} experiment directories (filtered from {len(all_experiment_dirs)} total)...")
                print(f"🎯 Filtering to experiments: {experiment_names}")
                print(f"📁 Directories found: {[d.name for d in experiment_dirs]}")
        else:
            experiment_dirs = all_experiment_dirs
            if verbose:
                print(f"📂 Scanning {len(experiment_dirs)} experiment directories...")
        
        # Add progress bar for scanning experiments
        experiment_iter = tqdm(experiment_dirs, desc="Scanning experiments", disable=not verbose) if TQDM_AVAILABLE and verbose else experiment_dirs
        
        for exp_dir in experiment_iter:
            experiment_id = exp_dir.name
            if verbose and not TQDM_AVAILABLE:
                print(f"   📊 Scanning experiment: {experiment_id}")
            exp_metadata = DataOrganizer.scan_experiment_directory(exp_dir, experiment_id, verbose and not TQDM_AVAILABLE)
            if exp_metadata["videos"]:
                metadata["experiments"][experiment_id] = exp_metadata
            elif verbose and not TQDM_AVAILABLE:
                print(f"   ⚠️  No videos found for experiment: {experiment_id}")
        return metadata

    @staticmethod
    def scan_experiment_directory(exp_dir, experiment_id, verbose=True):
        exp_metadata = {
            "experiment_id": experiment_id,
            "videos": {}
        }
        vids_dir = Path(exp_dir) / "vids"
        images_dir = Path(exp_dir) / "images"
        
        if not vids_dir.exists():
            if verbose:
                print(f"     ⚠️  No vids directory found for {experiment_id}")
            return exp_metadata
            
        video_files = list(vids_dir.glob("*.mp4"))
        if verbose:
            print(f"     🎬 Found {len(video_files)} video files")
        
        # Add progress bar for videos within experiment
        video_iter = tqdm(video_files, desc=f"Scanning {experiment_id} videos", leave=False, disable=not TQDM_AVAILABLE or not verbose) if len(video_files) > 10 else video_files
        
        for video_file in video_iter:
            video_id = video_file.stem
            video_images_dir = images_dir / video_id
            if video_images_dir.exists():
                if verbose and not TQDM_AVAILABLE:
                    image_count = len(list(video_images_dir.glob("*.jpg")))
                    print(f"       📸 Video {video_id}: {image_count} images")
                video_metadata = DataOrganizer.scan_video_directory(video_id, video_file, video_images_dir)
                exp_metadata["videos"][video_id] = video_metadata
            elif verbose and not TQDM_AVAILABLE:
                print(f"       ⚠️  No images directory for video: {video_id}")
        return exp_metadata

    @staticmethod
    def scan_video_directory(video_id, video_path, images_dir):
        parts = video_id.split('_')
        well_id = parts[-1]
        experiment_id = '_'.join(parts[:-1])
        video_metadata = {
            "video_id": video_id,
            "well_id": well_id,
            "mp4_path": str(video_path),
            "processed_jpg_images_dir": str(images_dir),
            "image_ids": [],
            "total_frames": 0,
            "image_size": None  # Will be populated if needed by other modules
        }
        
        # Fast scan: just generate image IDs from filenames (no image loading!)
        jpeg_files = sorted(images_dir.glob("*.jpg"))
        image_ids = []
        
        for jpeg_file in jpeg_files:
            name = jpeg_file.stem
            # If filename already contains full image_id (new convention with channel), use it
            if name.startswith(f"{experiment_id}_{well_id}") and (re.search(r'_ch\d+_t\d{3,4}$', name) or re.search(r'_t\d{3,4}$', name)):
                image_id = name
            # If legacy numeric filename (NNNN), convert to image_id with default channel 0
            elif re.fullmatch(r'\d{1,4}', name):
                image_id = build_image_id(video_id, int(name), channel=0)
            else:
                # Fallback: attempt to construct from video_id and the stem
                if re.search(r'_ch\d+_t\d{1,4}$', name) or re.search(r'_t\d{1,4}$', name):
                    image_id = name
                else:
                    # Default to channel 0 for unknown format
                    image_id = f"{experiment_id}_{well_id}_ch00_t{name}"
            image_ids.append(image_id)
        
        video_metadata["image_ids"] = image_ids
        video_metadata["total_frames"] = len(image_ids)
        
        # DEBUG: CSV loading and optional enhancement performed at debug log level
        experiment_name, well_id_parsed, channel = DataOrganizer.parse_video_id_for_metadata(video_id)
        # logger.debug(f"Video {video_id} parsed as: exp_name={experiment_name}, well={well_id_parsed}, channel={channel}")
        
        csv_df = DataOrganizer.load_legacy_metadata_csv(experiment_name)
        # logger.debug(f"CSV loaded: {csv_df is not None}, rows: {len(csv_df) if csv_df is not None else 0}")
        
        if csv_df is not None:
            logger.debug(f"CSV columns: {list(csv_df.columns)}")
            
            # Test different well_id matching strategies
            if 'well_id' in csv_df.columns:
                well_rows = csv_df[csv_df['well_id'].str.contains(well_id_parsed, na=False)]
                logger.debug(f"Found {len(well_rows)} rows for well {well_id_parsed} (contains strategy)")
                if not well_rows.empty:
                    logger.debug(f"Sample well_id values: {well_rows['well_id'].head().tolist()}")
            elif 'well' in csv_df.columns:
                well_rows = csv_df[csv_df['well'] == well_id_parsed]
                logger.debug(f"Found {len(well_rows)} rows for well {well_id_parsed} (exact well strategy)")
            else:
                logger.debug("No well_id or well column found in CSV")
                logger.debug(f"Available columns: {list(csv_df.columns)}")
                if len(csv_df) > 0:
                    logger.debug(f"First row sample: {dict(csv_df.iloc[0])}")
            
            # Perform schema transformation at debug level
            logger.debug("Testing schema transformation...")
            video_metadata = DataOrganizer.enhance_video_metadata_with_csv(
                video_metadata, csv_df, experiment_name, well_id_parsed,
                images_dir=Path(video_metadata.get('processed_jpg_images_dir')), 
                source_root=Path('/net/trapnell/vol1/home/nlammers/projects/data/morphseq')
            )
            
            logger.debug(f"Enhanced data image_ids type: {type(video_metadata.get('image_ids', []))}")
            if isinstance(video_metadata.get('image_ids'), dict) and video_metadata['image_ids']:
                first_key = next(iter(video_metadata['image_ids']))
                first_data = video_metadata['image_ids'][first_key]
                logger.debug(f"First image enhanced: {first_key}")
                logger.debug(f"Raw data keys: {list(first_data.get('raw_image_data_info', {}).keys())}")
                raw_data = first_data.get('raw_image_data_info', {})
                if raw_data:
                    logger.debug(f"Height (um): {raw_data.get('Height (um)')}")
                    logger.debug(f"height_um alias: {raw_data.get('height_um')}")
            
            logger.debug(f"Well metadata - medium: {video_metadata.get('medium')}")
            logger.debug(f"Well metadata - genotype: {video_metadata.get('genotype')}")
            
            # Continue processing without early exit
        
        return video_metadata

    @staticmethod
    def get_image_path_from_id(image_id, images_dir):
        """Return the expected on-disk path for a given `image_id` under `images_dir`.

        With the new convention the filename is the full `image_id`.jpg and lives
        inside the `images_dir` for the corresponding video.
        """
        return Path(images_dir) / f"{image_id}.jpg"

    @staticmethod
    def load_existing_per_experiment_metadata(raw_data_dir, experiment_names=None, verbose=False):
        """Load existing per-experiment metadata files and combine into a unified structure."""
        combined_metadata = {
            "file_info": {
                "creation_time": datetime.now().isoformat(),
                "script_version": "Module_0_Simplified"
            },
            "experiments": {}
        }

        if experiment_names:
            # Check specific experiments
            experiments_to_check = experiment_names
        else:
            # Check all experiment directories that exist
            experiments_to_check = [d.name for d in raw_data_dir.iterdir()
                                   if d.is_dir() and (d / f"experiment_metadata_{d.name}.json").exists()]

        for exp_id in experiments_to_check:
            per_exp_path = raw_data_dir / exp_id / f"experiment_metadata_{exp_id}.json"
            if per_exp_path.exists():
                try:
                    with open(per_exp_path, 'r') as f:
                        exp_data = json.load(f)

                    # Extract experiment data (handle different structures)
                    if 'experiments' in exp_data and exp_id in exp_data['experiments']:
                        combined_metadata['experiments'][exp_id] = exp_data['experiments'][exp_id]
                    elif exp_id in exp_data:
                        combined_metadata['experiments'][exp_id] = exp_data[exp_id]

                    # Preserve other metadata from the first file found
                    if not combined_metadata.get('entity_tracker') and exp_data.get('entity_tracker'):
                        combined_metadata['entity_tracker'] = exp_data['entity_tracker']

                    if verbose:
                        print(f"   📖 Loaded existing metadata for {exp_id}")

                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Could not load metadata for {exp_id}: {e}")

        return combined_metadata

    @staticmethod
    def save_per_experiment_metadata(raw_data_dir, metadata, experiment_id, verbose=False):
        """Save metadata for a single experiment to its own JSON file."""
        exp_dir = raw_data_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        per_exp_metadata_path = exp_dir / f"experiment_metadata_{experiment_id}.json"

        # Extract just this experiment's data
        if 'experiments' in metadata and experiment_id in metadata['experiments']:
            experiment_data = {
                'experiments': {
                    experiment_id: metadata['experiments'][experiment_id]
                }
            }
            # Preserve top-level metadata like entity tracking
            for key in metadata:
                if key != 'experiments':
                    experiment_data[key] = metadata[key]

            try:
                clean_data = DataOrganizer.convert_to_json_serializable(experiment_data)
                with open(per_exp_metadata_path, 'w') as f:
                    json.dump(clean_data, f, indent=2)
                if verbose:
                    print(f"   📄 Per-experiment metadata saved: {per_exp_metadata_path}")
                return per_exp_metadata_path
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Failed to save per-experiment metadata for {experiment_id}: {e}")
                return None
        else:
            if verbose:
                print(f"   ⚠️  No data found for experiment {experiment_id}")
            return None

    @staticmethod
    def get_images_for_detection(metadata, experiment_ids=None):
        images = []
        target_experiments = experiment_ids or metadata["experiments"].keys()
        for exp_id in target_experiments:
            if exp_id not in metadata["experiments"]:
                continue
            for video_id, video_data in metadata["experiments"][exp_id]["videos"].items():
                images_dir = Path(video_data["processed_jpg_images_dir"])
                for image_id in video_data["image_ids"]:
                    image_path = DataOrganizer.get_image_path_from_id(image_id, images_dir)
                    if image_path.exists():
                        images.append({
                            'image_id': image_id,
                            'image_path': str(image_path),
                            'video_id': video_id,
                            'well_id': video_data['well_id'],
                            'experiment_id': exp_id,
                            'frame_number': int(image_id.split('_t')[-1])
                        })
        return images

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test DataOrganizer script.")
    parser.add_argument("--source_dir", required=True, help="Source directory with experiment data.")
    parser.add_argument("--output_dir", required=True, help="Output directory for test run.")
    parser.add_argument("--experiment", required=True, help="Experiment name to process.")
    args = parser.parse_args()

    DataOrganizer.process_experiments(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        experiment_names=[args.experiment],
        verbose=True,
        overwrite=True
    )



# python /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/data_organization/data_organizer_refactor_test.py   \
#     --source_dir /net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data/stitched_FF_images   \
#     --output_dir /net/trapnell/vol1/home/mdcolon/proj/morphseq/test_data/refactor_test_output   \
#     --experiment "20250612_30hpf_ctrl_atf6"
