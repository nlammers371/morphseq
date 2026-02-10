# Module 1.1: ExperimentMetadata Implementation (1 Day)

## 🎯 Purpose
Update ExperimentMetadata class to work with Module 0.1 parsing utilities and new image ID format (with 't' prefix). Provide directory scanning and entity tracking for downstream GDINO usage.

## 📊 Key Updates from Existing Class

### **1. Import Module 0.1 Utilities**
```python
from .parsing_utils import (
    parse_video_id, build_image_id, get_image_path_from_id,
    extract_frame_number, normalize_frame_number, parse_entity_id
)
```

### **2. Update Image ID Format Handling**
```python
# OLD scan_video_directory():
image_id = f"{experiment_id}_{well_id}_{frame}"  # No 't' prefix

# NEW scan_video_directory():  
image_id = build_image_id(f"{experiment_id}_{well_id}", frame)  # With 't' prefix
```

### **3. Enhanced Directory Scanning**
```python
def scan_organized_experiments(self, raw_data_dir: Path) -> Dict:
    """
    PSEUDOCODE: Scan organized directory and compare with tracking
    
    LOGIC:
    - Scan directory structure using Module 0.1 parsing
    - Compare found experiments/videos/images with entity_tracking
    - Identify new items vs already tracked items
    - Update internal tracking and return discovery results
    """
    
def discover_new_content(self, raw_data_dir: Path) -> Dict:
    """
    PSEUDOCODE: Find new experiments/videos/images since last scan
    
    LOGIC:
    - Get current entity_tracking lists
    - Scan directory structure 
    - Return dict of new items found
    """
```

### **4. Add Entity Tracking Structure**
**Complete EntityIDTracker integration for ExperimentMetadata:**

```python
from .entity_id_tracker import EntityIDTracker

class ExperimentMetadata:
    def __init__(self, metadata_path):
        # ... existing init ...
        if self.metadata_path.exists():
            self.metadata = self.load_json()
            self._validate_and_update_tracking()
        else:
            self.metadata = self._create_empty_metadata()
    
    def _create_empty_metadata(self):
        return {
            "file_info": {"creation_time": datetime.now().isoformat()},
            "experiments": {},
            "entity_tracking": {"experiments": [], "videos": [], "images": [], "embryos": [], "snips": []}
        }
    
    def _validate_and_update_tracking(self):
        """Validate hierarchy and update tracking on load."""
        entities = EntityIDTracker.extract_entities(self.metadata)
        EntityIDTracker.validate_hierarchy(entities, raise_on_violations=True)
        self.metadata["entity_tracking"] = {k: list(v) for k, v in entities.items()}
    
    def save(self):
        """Save with validation."""
        entities = EntityIDTracker.extract_entities(self.metadata)
        EntityIDTracker.validate_hierarchy(entities, raise_on_violations=True)
        self.metadata["entity_tracking"] = {k: list(v) for k, v in entities.items()}
        # ... existing save logic
    
    def get_entity_summary(self):
        """Get entity counts."""
        entities = EntityIDTracker.extract_entities(self.metadata)
        return EntityIDTracker.get_counts(entities)
```

def update_entity_tracking(self, experiment_id: str, video_id: str, image_ids: List[str]):
    """
    PSEUDOCODE: Update tracking lists with new entities
    
    LOGIC:
    - Add to experiment_ids if new
    - Add to video_ids if new  
    - Add to image_ids if new (with 't' prefix)
    - Maintain sorted order for consistency
    """
```

### **5. Integration Method for GSINMO and SAM2**
```python
def get_images_for_detection(self, 
                           experiment_ids: Optional[List[str]] = None,
                           video_ids: Optional[List[str]] = None) -> List[Dict]:
    """
    CRITICAL: Get image info for GDINO detection system
    
    PSEUDOCODE:
    - Filter by experiment_ids or video_ids if provided
    - For each video: get image_ids and convert to file paths
    - Use Module 0.1 utilities to convert image_id to disk path
    - Return list of dicts with image_id, image_path, video_id, etc.
    
    Returns format expected by GDINO:
    [
        {
            'image_id': '20240411_A01_t0000',  # JSON tracking format
            'image_path': '/path/to/20240411/images/20240411_A01/0000.jpg',  # Disk path
            'video_id': '20240411_A01',
            'experiment_id': '20240411',
            'well_id': 'A01',
            'frame_number': 0
        }
    ]
    """
    images_for_detection = []
    
    target_experiments = experiment_ids or self.metadata["entity_tracking"]["experiment_ids"]
    
    for exp_id in target_experiments:
        for video_id, video_data in self.metadata["experiments"][exp_id]["videos"].items():
            if video_ids and video_id not in video_ids:
                continue
                
            images_dir = Path(video_data["processed_jpg_images_dir"])
            
            for image_id in video_data["image_ids"]:
                # Convert image_id to actual file path using Module 0.1
                image_path = self.convert_image_id_to_path(image_id, images_dir)
                
                if image_path.exists():
                    # Parse image_id using Module 0.1
                    components = parse_entity_id(image_id)[1]
                    
                    images_for_detection.append({
                        'image_id': image_id,
                        'image_path': str(image_path),
                        'video_id': video_id,
                        'experiment_id': components['experiment_id'],
                        'well_id': components['well_id'], 
                        'frame_number': int(components['frame_number'])
                    })
    
    return images_for_detection

def convert_image_id_to_path(self, image_id: str, images_dir: Path) -> Path:
    """
    Convert image_id to file path using Module 0.1
    
    PSEUDOCODE:
    - Extract frame number from image_id (remove 't' prefix)
    - Build file path: images_dir / {frame}.jpg
    """
    frame = extract_frame_number(image_id)  # Handles 't' prefix removal
    frame_str = normalize_frame_number(frame)
    return images_dir / f"{frame_str}.jpg"
```

### **6. Incremental Processing Methods**
```python
def sync_with_directory(self, raw_data_dir: Path) -> Dict:
    """
    PSEUDOCODE: Sync metadata with current directory state
    
    LOGIC:
    - Scan directory structure
    - Compare with entity tracking
    - Add new experiments/videos/images
    - Return sync results
    """
    
def add_experiment_from_directory(self, experiment_dir: Path) -> bool:
    """
    PSEUDOCODE: Add experiment by scanning its directory
    
    LOGIC:
    - Use Module 0.2 scan functions to analyze directory
    - Add to metadata with proper image_id format (with 't')
    - Update entity tracking
    """
```

## 📋 Implementation Steps

### **Step 1: Update Dependencies**
- Import Module 0.1 parsing utilities
- Update existing methods to use parsing functions

### **Step 2: Fix Image ID Format**
- Update `scan_video_directory()` to use `build_image_id()` 
- Ensure all image_ids stored with 't' prefix

### **Step 3: Add Entity Tracking**
- Add entity_tracking structure to metadata
- Implement tracking update methods

### **Step 4: Implement GDINO Integration**
- Add `get_images_for_detection()` method
- Add path conversion utilities

### **Step 5: Add Directory Sync**
- Implement directory scanning with change detection
- Add incremental processing support

## 🎯 Success Criteria

- [ ] **Uses Module 0.1 parsing** throughout
- [ ] **Image IDs have 't' prefix** in JSON
- [ ] **Entity tracking structure** implemented  
- [ ] **GDINO integration method** ready
- [ ] **Directory sync capability** working
- [ ] **Backward compatibility** maintained

## 🔄 Integration Points

**Uses from Module 0.1**: All parsing utilities  
**Uses from Module 0.2**: Directory scanning logic
**Feeds into Module 3.1**: `get_images_for_detection()` for GDINO