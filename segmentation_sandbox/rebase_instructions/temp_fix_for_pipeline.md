# MorphSeq Pipeline Improvements: GroundingDINO & ExperimentMetadata

## Overview
This document outlines critical improvements needed for the GroundingDINO annotations system and ExperimentMetadata class to ensure robust, maintainable functionality in the MorphSeq pipeline.

---

## 1. GroundingDINO Annotations Class Improvements

### 1.1 Critical Fixes

#### Fix Data Structure Inconsistency ⚠️ **PRIORITY 1**
**Problem**: Class uses `self.data` in initialization but `self.annotations` throughout methods, causing AttributeError.

**Solution**:
```python
# In __init__ method, change:
self.data = self._load_or_initialize()  # ❌ Wrong

# To:
self.annotations = self._load_or_initialize()  # ✅ Correct
```

**Impact**: All method calls must use `self.annotations` consistently:
- `self.annotations.get("images", {})`
- `self.annotations.get("file_info", {})`
- `self.annotations.get("high_quality_annotations", {})`

#### Simplify Image ID Handling ⚠️ **PRIORITY 2**
**Problem**: Complex entity ID parsing in `gdino_inference_with_visualization` causes ID mismatches. Files on disk are named differently (e.g., `t0000.jpg`) than their full image IDs (e.g., `20231206_F11_t0000`).

**Solution**: Use ExperimentMetadata as the source of truth for ID mappings
```python
# Current complex approach:
parsed_ids = parse_entity_id(str(image_path))  # ❌ Overly complex and brittle

# Better approach - let metadata handle the mapping:
# The metadata already knows that image_id "20231206_F11_t0000" 
# corresponds to file "/path/to/20231206/images/20231206_F11/t0000.jpg"

# In GroundingDINO, when you have an image_path, query metadata to get its ID
# This requires adding a reverse lookup function to ExperimentMetadata
```

**Benefits**:
- Single source of truth for ID-to-path mappings
- Handles any file naming convention
- No brittle path parsing or assumptions about directory structure
- Consistent ID matching between annotation storage and retrieval

**Note**: This approach requires the `get_image_id_path()` function in ExperimentMetadata, and potentially its inverse `get_path_to_image_id()` for reverse lookups.

#### Remove PyTorch 2.6+ Compatibility Patches
**Problem**: Unnecessary complexity from PyTorch version handling.

**Solution**: Remove all instances of the torch.load patching:
```python
# Remove this entire pattern:
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load
```

**Replace with**: Direct model loading as in previous versions.

### 1.2 Architectural Improvements

#### Reorganize Class Methods by Functionality
Group methods into logical sections for better maintainability:

```python
class GroundedDinoAnnotations:
    # === Core Data Management ===
    def __init__(self, ...): pass
    def _load_or_initialize(self): pass
    def save(self): pass
    def _mark_unsaved(self): pass
    
    # === Annotation CRUD Operations ===
    def add_annotation(self, ...): pass
    def get_annotations_for_image(self, ...): pass
    def get_annotated_image_ids(self, ...): pass
    
    # === Batch Processing ===
    def process_missing_annotations(self, ...): pass
    def get_missing_annotations(self, ...): pass
    
    # === Quality Filtering ===
    def generate_high_quality_annotations(self, ...): pass
    def get_or_generate_high_quality_annotations(self, ...): pass
    
    # === Metadata Integration ===
    def set_metadata_path(self, ...): pass
    def get_all_metadata_image_ids(self, ...): pass
    
    # === Utilities & Reporting ===
    def get_summary(self): pass
    def print_summary(self): pass
```

#### Improve EntityIDTracker Integration
**Keep EntityIDTracker** but make it context-aware:

```python
def __init__(self, annotations_path, track_entities=True):
    self.track_entities = track_entities
    # ... rest of initialization
    
def save(self, force_track_entities=None):
    """Save with entity tracking (required for pipeline, optional for debugging)."""
    # Allow override for debugging, but default to True for pipeline
    should_track = force_track_entities if force_track_entities is not None else self.track_entities
    
    if should_track:
        self.annotations = EntityIDTracker.update_entity_tracker(
            self.annotations, 
            pipeline_step="grounded_dino_annotations"
        )
    
    # Atomic write with backup
    temp_path = Path(str(self.annotations_path) + '.tmp')
    with open(temp_path, 'w') as f:
        json.dump(self.annotations, f, indent=2)
    temp_path.rename(self.annotations_path)
```

**Usage Context**:
- **In production pipeline**: `track_entities=True` (default) - REQUIRED for data lineage
- **During debugging/data manipulation**: Can temporarily set `track_entities=False` or use `save(force_track_entities=False)`
- **Key principle**: EntityIDTracker is essential for the pipeline's data integrity, but can be bypassed during development/debugging to isolate issues

### 1.3 Error Handling & Debugging

#### Add Comprehensive Logging
```python
def get_annotated_image_ids(self, prompt, verbose=False):
    """Get annotated images with better debugging."""
    annotated = set()
    images = self.annotations.get("images", {})
    
    if verbose:
        print(f"Checking {len(images)} images for prompt: '{prompt}'")
    
    for image_id, annotations in images.items():
        for ann in annotations:
            if ann.get("prompt") == prompt:  # Exact match
                annotated.add(image_id)
                if verbose:
                    print(f"  ✓ Found annotation for {image_id}")
                break
    
    if verbose:
        print(f"Total annotated images: {len(annotated)}")
    
    return annotated
```

#### Validate Annotation Detection
Add validation to ensure prompt matching works correctly:
```python
def validate_prompt_matching(self, prompt):
    """Debug helper to show what prompts exist in annotations."""
    all_prompts = set()
    for image_id, annotations in self.annotations.get("images", {}).items():
        for ann in annotations:
            all_prompts.add(ann.get("prompt", ""))
    
    print(f"Searching for prompt: '{prompt}'")
    print(f"Available prompts: {all_prompts}")
    print(f"Exact match exists: {prompt in all_prompts}")
```

---

## 2. ExperimentMetadata Class Improvements

### 2.1 Add Image Path Resolution

#### New Method: get_image_id_path
```python
def get_image_id_path(self, image_id: str) -> Optional[Path]:
    """
    Get the full file path for a given image_id.
    
    Args:
        image_id: Image ID like "20231206_F11_t0000"
        
    Returns:
        Path object to the image file, or None if not found
        
    Example:
        >>> meta = ExperimentMetadata("metadata.json")
        >>> path = meta.get_image_id_path("20231206_F11_t0000")
        >>> print(path)
        /net/trapnell/vol1/.../20231206/images/20231206_F11/20231206_F11_t0000.jpg
    """
    from pathlib import Path
    from .parsing_utils import parse_entity_id
    
    try:
        # Parse image_id to extract components
        parsed = parse_entity_id(image_id)
        experiment_id = parsed.get("experiment_id")
        video_id = parsed.get("video_id")
    except ValueError:
        return None
    
    # Navigate metadata structure
    experiments = self.metadata.get("experiments", {})
    if experiment_id not in experiments:
        return None
        
    videos = experiments[experiment_id].get("videos", {})
    if video_id not in videos:
        return None
        
    images_dir = videos[video_id].get("processed_jpg_images_dir")
    if not images_dir:
        return None
    
    # Construct path (assuming {image_id}.jpg naming)
    return Path(images_dir) / f"{image_id}.jpg"
```

### 2.2 Future Enhancements (Post-MVP)

#### Path Caching for Performance
```python
def _build_image_path_cache(self):
    """Build cache for repeated lookups (implement later if needed)."""
    self._image_path_cache = {}
    # Implementation deferred until performance need is proven
```

#### Batch Path Resolution
```python
def get_image_paths_batch(self, image_ids: List[str]) -> Dict[str, Optional[Path]]:
    """Get paths for multiple images efficiently (future enhancement)."""
    return {image_id: self.get_image_id_path(image_id) for image_id in image_ids}
```

---

## 3. Integration Best Practices

### 3.1 Maintain Backward Compatibility
- Keep existing API signatures when possible
- Use entity parsing only when explicitly needed
- Make all advanced features opt-in

### 3.2 Progressive Enhancement Strategy
1. **Phase 1**: Fix critical bugs (data structure, ID handling)
2. **Phase 2**: Add simple improvements (get_image_id_path)
3. **Phase 3**: Optimize if needed (caching, batch operations)

### 3.3 Testing Checklist
- [ ] Verify `self.annotations` used consistently
- [ ] Test annotation retrieval with exact prompt matching
- [ ] Confirm image paths resolve correctly
- [ ] Validate EntityIDTracker doesn't break core functionality
- [ ] Check batch processing with auto-save works

---

## 4. Implementation Priority Order

1. **Immediate (Breaking Fixes)**:
   - Fix data structure inconsistency
   - Simplify image ID handling
   - Add get_image_id_path to ExperimentMetadata

2. **Short-term (Stability)**:
   - Remove PyTorch compatibility patches
   - Improve error messages and logging
   - Add debugging helpers

3. **Long-term (Optimization)**:
   - Implement caching if needed
   - Add batch operations
   - Performance profiling and tuning

---

## 5. Code Quality Guidelines

### Principles to Follow
- **KISS**: Keep It Simple, Stupid - prefer simple solutions
- **YAGNI**: You Aren't Gonna Need It - don't add features until needed
- **DRY**: Don't Repeat Yourself - but balance with clarity

### Documentation Standards
- Every public method needs a docstring
- Include usage examples for complex methods
- Document expected data structures
- Add type hints for clarity

### Error Handling
- Return None/empty for missing data (don't raise exceptions)
- Log warnings for potential issues
- Provide helpful error messages
- Add verbose flags for debugging

---

## Summary

The key to improving these classes is to focus on fixing the critical bugs first, then gradually enhancing functionality. The proposed changes maintain the useful features (like EntityIDTracker and metadata integration) while simplifying the implementation and improving reliability.

Remember: A working simple solution is better than a broken complex one. Start simple, validate it works, then enhance as needed.