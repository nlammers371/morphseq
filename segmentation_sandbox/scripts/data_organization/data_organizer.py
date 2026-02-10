"""
@GPT4.1 - IMPLEMENT DataOrganizer class here

DETAILED REQUIREMENTS from module_0_2_simplified_dataorganization.md:

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
    def process_experiments(source_dir, output_dir, experiment_names=None, verbose=True, overwrite=False):
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
        
        metadata_path = raw_data_dir / "experiment_metadata.json"
        
        if verbose:
            print(f"📂 Source directory: {source_dir}")
            print(f"📂 Output directory: {raw_data_dir}")
            print(f"📋 Metadata file: {metadata_path}")
            print(f"🔄 Overwrite mode: {overwrite}")
            if not PYVIPS_AVAILABLE:
                print("⚠️  PyVips not available - using OpenCV (slower). Install with: pip install pyvips")

        # Load existing metadata to check what's already processed
        existing_metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    existing_metadata = json.load(f)
                if verbose:
                    existing_count = len(existing_metadata.get('experiments', {}))
                    print(f"📖 Found existing metadata with {existing_count} experiments")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Could not load existing metadata: {e}")
                existing_metadata = {}

        # Find experiments to process
        if experiment_names:
            experiment_dirs = [Path(source_dir) / name for name in experiment_names if (Path(source_dir) / name).is_dir()]
            if verbose:
                print(f"🔍 Processing specified experiments: {experiment_names}")
        else:
            experiment_dirs = DataOrganizer.find_experiment_directories(Path(source_dir))
            if verbose:
                print(f"🔍 Found {len(experiment_dirs)} experiments in source directory")

        if not experiment_dirs:
            print("❌ No experiments found to process!")
            
            # Initialize empty metadata file if it doesn't exist
            if not metadata_path.exists():
                if verbose:
                    print("📋 Initializing empty experiment_metadata.json...")
                empty_metadata = {
                    "file_info": {
                        "creation_time": datetime.now().isoformat(),
                        "script_version": "Module_0_Simplified"
                    },
                    "experiments": {}
                }
                try:
                    with open(metadata_path, 'w') as f:
                        json.dump(empty_metadata, f, indent=2)
                    if verbose:
                        print(f"✅ Empty metadata file created: {metadata_path}")
                except Exception as e:
                    print(f"⚠️  Failed to create empty metadata file: {e}")
            else:
                if verbose:
                    print(f"📋 Metadata file already exists: {metadata_path}")
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
            
            if already_processed and entity_tracking_complete and not overwrite:
                experiments_skipped.append(experiment_id)
                if verbose:
                    print(f"⏭️  Skipping already processed experiment: {experiment_id}")
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
                
            # Ensure metadata file has complete entity tracking
            if not metadata_path.exists():
                if verbose:
                    print("📋 Creating metadata file from existing processed experiments...")
                # Scan existing organized data to create metadata
                metadata = DataOrganizer.scan_organized_experiments(raw_data_dir, verbose=False)
            else:
                if verbose:
                    print("📋 Updating existing metadata file with entity tracking...")
                # Use existing metadata and update it
                metadata = existing_metadata
                # Re-scan to ensure we have current data
                current_scan = DataOrganizer.scan_organized_experiments(raw_data_dir, verbose=False)
                metadata['experiments'] = current_scan.get('experiments', {})
                
            # Add entity tracker (consistent with normal processing)
            metadata = EntityIDTracker.add_entity_tracker(
                metadata, 
                pipeline_step="module_0_data_organization"
            )
            
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                if verbose:
                    exp_count = len(metadata.get('experiments', {}))
                    entities = EntityIDTracker.extract_entities(metadata)
                    entity_counts = {k: len(v) for k, v in entities.items()}
                    print(f"✅ Metadata updated with entity tracking ({exp_count} experiments)")
                    print(f"📊 Entities: {entity_counts}")
            except Exception as e:
                print(f"⚠️  Failed to save metadata: {e}")
            return

        # Process experiments one by one with incremental saves
        for i, exp_dir in enumerate(experiments_to_process, 1):
            experiment_id = exp_dir.name
            print(f"\n🧪 Processing experiment {i}/{len(experiments_to_process)}: {experiment_id}")
            
            if verbose:
                stitch_count = len(list(exp_dir.glob('*_stitch.*')))
                print(f"   Found {stitch_count} stitch files")
                
            # Process this experiment
            DataOrganizer.organize_experiment(exp_dir, raw_data_dir, experiment_id, verbose, overwrite)
            
            # Update and save metadata incrementally for robustness
            if verbose:
                print(f"   💾 Updating metadata for {experiment_id}...")
                
            current_metadata = DataOrganizer.scan_organized_experiments(raw_data_dir, verbose=False)
            
            # Add entity tracker to autosave (consistent with final save)
            current_metadata = EntityIDTracker.add_entity_tracker(
                current_metadata, 
                pipeline_step="module_0_data_organization"
            )
            
            # Save metadata after each experiment (autosave)
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(current_metadata, f, indent=2)
                if verbose:
                    exp_count = len(current_metadata.get('experiments', {}))
                    print(f"   ✅ Metadata saved ({exp_count} total experiments)")
            except Exception as e:
                print(f"   ⚠️  Failed to save metadata: {e}")

        # Final metadata generation and validation
        print("\n📋 Generating final experiment metadata...")
        final_metadata = DataOrganizer.scan_organized_experiments(raw_data_dir, verbose)

        # Add entity tracker (MANDATORY for downstream modules)
        if verbose:
            print("📋 Adding embedded entity tracker...")

        final_metadata = EntityIDTracker.add_entity_tracker(
            final_metadata, 
            pipeline_step="module_0_data_organization"
        )

        if verbose:
            entities = EntityIDTracker.extract_entities(final_metadata)
            entity_counts = {k: len(v) for k, v in entities.items()}
            print(f"✅ Entity tracker embedded: {entity_counts}")

        # Final save
        with open(metadata_path, 'w') as f:
            json.dump(final_metadata, f, indent=2)
            
        print(f"✅ Complete! Metadata saved to: {metadata_path}")
        
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
    def organize_experiment(experiment_dir, output_dir, experiment_id, verbose=True, overwrite=False):
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
            DataOrganizer.process_well(files, Path(output_dir) / experiment_id, video_id, verbose, overwrite)

    @staticmethod
    def process_well(image_files, exp_output_dir, video_id, verbose=True, overwrite=False):
        images_dir = Path(exp_output_dir) / "images" / video_id
        vids_dir = Path(exp_output_dir) / "vids"
        images_dir.mkdir(parents=True, exist_ok=True)
        vids_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = vids_dir / f"{video_id}.mp4"
        
        if video_path.exists() and not overwrite:
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
                
                if jpeg_path.exists() and not overwrite:
                    # File exists, add to collection and update progress
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
    def scan_organized_experiments(raw_data_dir, verbose=True):
        metadata = {
            "file_info": {
                "creation_time": datetime.now().isoformat(),
                "script_version": "Module_0_Simplified"
            },
            "experiments": {}
        }
        
        experiment_dirs = [d for d in Path(raw_data_dir).iterdir() if d.is_dir() and d.name != "experiment_metadata.json"]
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
        
        return video_metadata

    @staticmethod
    def get_image_path_from_id(image_id, images_dir):
        """Return the expected on-disk path for a given `image_id` under `images_dir`.

        With the new convention the filename is the full `image_id`.jpg and lives
        inside the `images_dir` for the corresponding video.
        """
        return Path(images_dir) / f"{image_id}.jpg"

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
