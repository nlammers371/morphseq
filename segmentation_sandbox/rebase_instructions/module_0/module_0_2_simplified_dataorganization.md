# Module 0.2: Complete Simplified Data Organization (1 Day)

## 🎯 Purpose
**SIMPLIFIED**: Follow the original 01_prepare_videos.py approach - just organize files properly and create lightweight metadata by scanning the organized structure. Uses Module 0.1 parsing utilities.

## 📊 Key File Organization

### **Input Structure** (Raw stitch files)
```
directory_with_experiments/
├── 20240411/
│   ├── A01_t0000_ch00_stitch.png
│   ├── A01_t0001_ch00_stitch.png
│   └── B02_0123_stitch.tif        # May not have 't' prefix
└── 20250624_chem02_28C_T00_1356/  # Complex experiment ID
    ├── H01_t0000_ch00_stitch.png
    └── H01_t0001_ch00_stitch.png
```

### **Output Structure** (Organized files)
```
output_parent_dir/
└── raw_data_organized/
    ├── experiment_metadata.json
    ├── 20240411/
    │   ├── vids/
    │   │   └── 20240411_A01.mp4
    │   └── images/
    │       └── 20240411_A01/
    │           ├── 0000.jpg         # Simple frame number!
    │           └── 0001.jpg
    └── 20250624_chem02_28C_T00_1356/
        ├── vids/
        │   └── 20250624_chem02_28C_T00_1356_H01.mp4
        └── images/
            └── 20250624_chem02_28C_T00_1356_H01/
                ├── 0000.jpg
                └── 0001.jpg
```

### **Metadata JSON Structure** (Lightweight tracking)
```json
{
  "file_info": {
    "creation_time": "2024-01-01T12:00:00",
    "script_version": "Module_0_Simplified"
  },
  "experiments": {
    "20240411": {
      "experiment_id": "20240411", 
      "videos": {
        "20240411_A01": {
          "video_id": "20240411_A01",
          "well_id": "A01",
          "mp4_path": "/path/to/20240411/vids/20240411_A01.mp4",
          "processed_jpg_images_dir": "/path/to/20240411/images/20240411_A01",
          "image_ids": [
            "20240411_A01_t0000",    // WITH 't' prefix for JSON tracking!
            "20240411_A01_t0001"     // Differentiates from snip_id later
          ],
          "total_frames": 2
        }
      }
    }
  }
}
```

## 🔧 Complete Implementation

### **Simple Discovery & Processing**
```python
# Import parsing utilities from Module 0.1
from .parsing_utils import (
    parse_video_id, build_image_id, get_image_path_from_id,
    extract_frame_number, normalize_frame_number
)

def process_experiments(source_dir: Path, output_dir: Path, 
                       experiment_names: Optional[List[str]] = None):
    """
    PSEUDOCODE: Complete processing using Module 0.1 utilities
    
    LOGIC:
    1. Find experiment directories  
    2. For each experiment: organize images and create videos
    3. Scan organized structure to create metadata (using 0.1 parsing)
    4. Save metadata JSON
    """
    raw_data_dir = output_dir / "raw_data_organized"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Find experiments to process
    if experiment_names:
        experiment_dirs = [source_dir / name for name in experiment_names
                         if (source_dir / name).is_dir()]
    else:
        experiment_dirs = find_experiment_directories(source_dir)
    
    # Process each experiment
    for exp_dir in experiment_dirs:
        experiment_id = exp_dir.name
        print(f"Processing experiment: {experiment_id}")
        
        organize_experiment(exp_dir, raw_data_dir, experiment_id)
    
    # Create metadata by scanning organized structure
    metadata_path = raw_data_dir / "experiment_metadata.json"
    metadata = scan_organized_experiments(raw_data_dir)
    save_experiment_metadata(metadata, metadata_path)
    
    print(f"Complete! Metadata saved to: {metadata_path}")

# Core Functions

def find_experiment_directories(base_dir: Path) -> List[Path]:
    """Find directories containing stitch files"""
    experiments = []
    for potential_dir in base_dir.iterdir():
        if potential_dir.is_dir():
            stitch_files = list(potential_dir.glob('*_stitch.*'))
            if stitch_files:
                experiments.append(potential_dir)
    return experiments

def parse_stitch_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Extract well_id and frame from stitch filename
    
    Examples:
    'A01_t0000_ch00_stitch.png' → ('A01', '0000')
    'B02_0123_stitch.tif' → ('B02', '0123')
    """
    # Find well pattern [A-H][0-9]{2}
    well_match = re.search(r'([A-H]\d{2})', filename)
    if not well_match:
        return None
    well_id = well_match.group(1)
    
    # Find 3-4 digits (with or without 't' prefix)
    frame_match = re.search(r't?(\d{3,4})', filename)
    if not frame_match:
        return None
    frame = frame_match.group(1)
    
    return well_id, frame

def organize_experiment(experiment_dir: Path, output_dir: Path, experiment_id: str):
    """
    Organize one experiment following original 01_prepare_videos.py structure
    """
    stitch_files = list(experiment_dir.glob('*_stitch.*'))
    
    # Group by well
    wells = defaultdict(list)
    for stitch_file in stitch_files:
        result = parse_stitch_filename(stitch_file.name)
        if result:
            well_id, frame = result
            wells[well_id].append((stitch_file, frame))
    
    # Process each well
    for well_id, files in wells.items():
        video_id = f"{experiment_id}_{well_id}"
        process_well(files, output_dir / experiment_id, video_id)

def process_well(image_files: List[Tuple[Path, str]], exp_output_dir: Path, video_id: str):
    """Process one well - convert images and create video"""
    # Create directories
    images_dir = exp_output_dir / "images" / video_id
    vids_dir = exp_output_dir / "vids"
    images_dir.mkdir(parents=True, exist_ok=True)
    vids_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert images
    jpeg_paths = []
    for stitch_path, frame in sorted(image_files, key=lambda x: x[1]):
        # CRITICAL: Save as simple frame number (NO 't' prefix)
        jpeg_filename = f"{frame.zfill(4)}.jpg"
        jpeg_path = images_dir / jpeg_filename
        
        if not jpeg_path.exists():  # Skip if already processed
            convert_to_jpeg(stitch_path, jpeg_path)
        jpeg_paths.append(jpeg_path)
    
    # Create video
    video_path = vids_dir / f"{video_id}.mp4"
    if not video_path.exists():
        create_video_from_jpegs(jpeg_paths, video_path, video_id)

def convert_to_jpeg(source_path: Path, target_path: Path, quality: int = 90):
    """Simple image conversion (try pyvips, fallback to OpenCV)"""
    try:
        if PYVIPS_AVAILABLE:
            img = pyvips.Image.new_from_file(str(source_path))
            if img.bands == 4:  # RGBA to RGB
                img = img[:3]
            img.write_to_file(str(target_path), Q=quality)
        else:
            # OpenCV fallback
            image = cv2.imread(str(source_path))
            if image is not None:
                if len(image.shape) == 3 and image.shape[2] == 4:  # BGRA to BGR
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                cv2.imwrite(str(target_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    except Exception as e:
        print(f"Failed to convert {source_path}: {e}")

def create_video_from_jpegs(jpeg_paths: List[Path], video_path: Path, video_id: str):
    """Create MP4 video from JPEG sequence with frame overlays"""
    if not jpeg_paths:
        return
    
    # Get dimensions from first image
    first_frame = cv2.imread(str(jpeg_paths[0]))
    if first_frame is None:
        return
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, 5, (width, height))
    
    for jpeg_path in jpeg_paths:
        frame = cv2.imread(str(jpeg_path))
        if frame is None:
            continue
            
        # Add frame number overlay
        frame_num = jpeg_path.stem  # e.g., "0042"
        cv2.putText(frame, frame_num, (width - 100, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        video_writer.write(frame)
    
    video_writer.release()

# Metadata Creation by File Scanning

def scan_organized_experiments(raw_data_dir: Path) -> Dict:
    """Scan organized file structure and create metadata"""
    metadata = {
        "file_info": {
            "creation_time": datetime.now().isoformat(),
            "script_version": "Module_0_Simplified"
        },
        "experiments": {}
    }
    
    # Find experiment directories
    for exp_dir in raw_data_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name != "experiment_metadata.json":
            experiment_id = exp_dir.name
            exp_metadata = scan_experiment_directory(exp_dir, experiment_id)
            if exp_metadata["videos"]:  # Only add if has videos
                metadata["experiments"][experiment_id] = exp_metadata
    
    return metadata

def scan_experiment_directory(exp_dir: Path, experiment_id: str) -> Dict:
    """Scan one experiment directory"""
    exp_metadata = {
        "experiment_id": experiment_id,
        "videos": {}
    }
    
    vids_dir = exp_dir / "vids"
    images_dir = exp_dir / "images"
    
    # Find video files
    if vids_dir.exists():
        for video_file in vids_dir.glob("*.mp4"):
            video_id = video_file.stem  # e.g., "20240411_A01"
            
            # Find corresponding image directory
            video_images_dir = images_dir / video_id
            
            if video_images_dir.exists():
                video_metadata = scan_video_directory(video_id, video_file, video_images_dir)
                exp_metadata["videos"][video_id] = video_metadata
    
    return exp_metadata

def scan_video_directory(video_id: str, video_path: Path, images_dir: Path) -> Dict:
    """Scan video directory and build metadata"""
    # Parse video_id to get components
    # video_id could be "20240411_A01" or "20250624_chem02_28C_T00_1356_H01"
    parts = video_id.split('_')
    well_id = parts[-1]  # Last part is always well (e.g., "A01", "H01")
    experiment_id = '_'.join(parts[:-1])  # Everything before well
    
    video_metadata = {
        "video_id": video_id,
        "well_id": well_id,
        "mp4_path": str(video_path),
        "processed_jpg_images_dir": str(images_dir),
        "image_ids": [],
        "total_frames": 0
    }
    
    # Scan JPEG files
    jpeg_files = sorted(images_dir.glob("*.jpg"))
    
    image_ids = []
    for jpeg_file in jpeg_files:
        frame = jpeg_file.stem  # e.g., "0000"
        # CRITICAL: Create image_id WITH 't' prefix for JSON tracking
        image_id = f"{experiment_id}_{well_id}_t{frame}"
        image_ids.append(image_id)
    
    video_metadata["image_ids"] = image_ids
    video_metadata["total_frames"] = len(image_ids)
    
    return video_metadata

def save_experiment_metadata(metadata: Dict, output_path: Path):
    """Save metadata with simple backup"""
    if output_path.exists():
        # Simple timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = output_path.with_suffix(f'.backup_{timestamp}.json')
        shutil.copy2(output_path, backup_path)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

# Utility Functions for Downstream Use

def get_image_path_from_id(image_id: str, images_dir: Path) -> Path:
    """
    Convert image_id to actual file path
    
    INPUT: "20240411_A01_t0000"  
    OUTPUT: Path("images/20240411_A01/0000.jpg")
    """
    # Extract frame number (remove 't' prefix)
    frame = image_id.split('_t')[-1]  # "0000"
    return images_dir / f"{frame}.jpg"

def load_experiment_metadata(metadata_path: Path) -> Dict:
    """Load experiment metadata JSON"""
    with open(metadata_path) as f:
        return json.load(f)

def get_images_for_detection(metadata: Dict, 
                           experiment_ids: Optional[List[str]] = None) -> List[Dict]:
    """
    Get image information for GDINO detection
    
    Returns list of dicts with image_id, image_path, video_id, etc.
    """
    images = []
    
    target_experiments = experiment_ids or metadata["experiments"].keys()
    
    for exp_id in target_experiments:
        if exp_id not in metadata["experiments"]:
            continue
            
        for video_id, video_data in metadata["experiments"][exp_id]["videos"].items():
            images_dir = Path(video_data["processed_jpg_images_dir"])
            
            for image_id in video_data["image_ids"]:
                image_path = get_image_path_from_id(image_id, images_dir)
                
                if image_path.exists():
                    images.append({
                        'image_id': image_id,
                        'image_path': str(image_path),
                        'video_id': video_id,
                        'well_id': video_data['well_id'],
                        'experiment_id': exp_id,
                        'frame_number': int(image_id.split('_t')[-1])  # Extract frame
                    })
    
    return images
```

## 🎯 Usage Example

```python
# Process experiments (like original 01_prepare_videos.py)
from pathlib import Path

source_dir = Path("/data/stitched_images")
output_dir = Path("/data/output")

# Process specific experiments
process_experiments(
    source_dir=source_dir,
    output_dir=output_dir, 
    experiment_names=["20240411", "20250624_chem02_28C_T00_1356"]
)

# Load metadata for downstream use
metadata_path = output_dir / "raw_data_organized" / "experiment_metadata.json"
metadata = load_experiment_metadata(metadata_path)

# Get images for GDINO detection (Module 3.1 will use this)
images_for_detection = get_images_for_detection(
    metadata, 
    experiment_ids=["20240411"]
)

print(f"Found {len(images_for_detection)} images ready for detection")

# Example image info:
# {
#   'image_id': '20240411_A01_t0000',           # JSON tracking ID (with 't')
#   'image_path': '/path/to/20240411/images/20240411_A01/0000.jpg',  # Actual file
#   'video_id': '20240411_A01',
#   'well_id': 'A01',
#   'experiment_id': '20240411',
#   'frame_number': 0
# }
```

## 📋 Success Criteria

- [ ] **Simple file organization** matching original script structure
- [ ] **Correct image naming** (disk: `0000.jpg`, JSON: `t0000`)  
- [ ] **Handles complex experiment IDs** by parsing backwards
- [ ] **Lightweight metadata** created by file scanning
- [ ] **Ready for GDINO** (provides image paths and metadata)
- [ ] **Fast and simple** (no complex tracking, just file operations)

**This is much cleaner!** Follow the original approach - organize files simply, create basic metadata by scanning.