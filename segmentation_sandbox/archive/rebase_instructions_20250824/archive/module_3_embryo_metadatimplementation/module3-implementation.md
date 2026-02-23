# Module 3: Biological Annotations Implementation Guide

## Overview
Implement EmbryoMetadata system for biological annotations, building on SAM2 segmentation results.

## File Structure
```
segmentation_sandbox/
└── annotations/
    ├── unified_managers.py
    ├── annotation_batch.py
    ├── embryo_metadata.py
    └── embryo_metadata_tutorial.ipynb
```

## Task 1: Copy `unified_managers.py`

**Source**: Copy from `rebase_instructions/module_3/unified_embryo_annotation_managers.py`

**Modifications needed**:

1. **Fix imports**:
```python
# Add at top
from utils.parsing_utils import extract_frame_number, parse_entity_id
from metadata.schema_manager import SchemaManager
```

2. **Fix frame number extraction** (in _check_dead_conflicts, around line 180):
```python
# OLD: Custom parsing
frame_num = int(snip_id.split("_s")[1])

# NEW: Use parsing utils
frame_num = extract_frame_number(snip_id)
```

## Task 2: Copy `annotation_batch.py`

**Source**: Copy from `rebase_instructions/module_3/annotation_batch.py`

**Modifications needed**:

1. **Add imports**:
```python
from utils.parsing_utils import parse_entity_id, get_entity_type
```

2. **Fix validate_id_format** (around line 85):
```python
def validate_id_format(self, entity_id: str, entity_type: str) -> bool:
    """Use parsing utils for validation."""
    detected_type = get_entity_type(entity_id)
    return detected_type == entity_type
```

## Task 3: Create `embryo_metadata.py`

**Source**: Adapt from `rebase_instructions/module_3/streamlined_embryo_metadata.py`

**Key modifications**:

1. **Update imports**:
```python
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

# Module imports
from utils.base_file_handler import BaseFileHandler
from utils.entity_id_tracker import EntityIDTracker
from utils.parsing_utils import (
    get_entity_type, 
    extract_embryo_id,
    extract_frame_number
)
from metadata.schema_manager import SchemaManager
from annotations.unified_managers import UnifiedEmbryoManager
from annotations.annotation_batch import AnnotationBatch
```

2. **Fix inheritance and init**:
```python
class EmbryoMetadata(BaseFileHandler, UnifiedEmbryoManager):
    def __init__(self, 
                 sam_annotation_path: Union[str, Path],
                 embryo_metadata_path: Optional[Union[str, Path]] = None,
                 gen_if_no_file: bool = False,
                 verbose: bool = True,
                 schema_path: Optional[str] = None):
        
        # Initialize file handler
        if embryo_metadata_path is None:
            sam_path = Path(sam_annotation_path)
            embryo_metadata_path = sam_path.with_name(
                sam_path.stem + "_embryo_metadata.json"
            )
        
        super().__init__(embryo_metadata_path, verbose=verbose)
        
        # Schema manager
        self.schema_manager = SchemaManager(schema_path) if schema_path else SchemaManager()
        
        # Load SAM annotations
        self.sam_annotation_path = Path(sam_annotation_path)
        self.sam_annotations = self.load_json(self.sam_annotation_path)
        
        # Rest of init...
```

3. **Add entity validation to save**:
```python
def save(self, backup: bool = True):
    """Save with entity validation using embedded tracker approach."""
    # EntityIDTracker is a PURE CONTAINER - use static methods for embedded tracking
    try:
        # Update embedded entity tracker in the embryo metadata
        self.data = EntityIDTracker.update_entity_tracker(
            self.data,
            pipeline_step="module_3_embryo_metadata"
        )
        
        # Validate entity consistency
        entities = EntityIDTracker.extract_entities(self.data)
        
        # Basic validation - ensure snips have embryos
        for snip_id in entities["snips"]:
            embryo_id = self.get_embryo_id_from_snip(snip_id)
            if not embryo_id:
                raise ValueError(f"Orphaned snip: {snip_id}")
        
        if self.verbose:
            entity_counts = EntityIDTracker.get_counts(entities)
            print(f"📋 Entity tracker updated: {entity_counts}")
            
    except Exception as e:
        if self.verbose:
            print(f"⚠️ Entity validation warning: {e}")
    
    # Update file info
    self.data["file_info"]["last_updated"] = self.get_timestamp()
    
    # Save
    self.save_json(self.data, create_backup=backup)
    
    if self.verbose:
        embryo_count = len(self.data["embryos"])
        snip_count = sum(len(e["snips"]) for e in self.data["embryos"].values())
        print(f"💾 Saved: {embryo_count} embryos, {snip_count} snips")
```

4. **Fix frame resolution** (in _resolve_frames):
```python
def _resolve_frames(self, embryo_id: str, frames: Union[str, List[str]]) -> List[str]:
    """Resolve frame specification to snip IDs."""
    available_snips = self.get_available_snips(embryo_id)
    
    if frames == "all":
        return available_snips
    elif isinstance(frames, list):
        return [s for s in frames if s in available_snips]
    elif isinstance(frames, str):
        # Frame ranges
        if ":" in frames:
            # Parse frame range
            parts = frames.split(":")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if len(parts) > 1 and parts[1] else 9999
            step = int(parts[2]) if len(parts) > 2 else 1
            
            # Filter snips by frame number
            result = []
            for snip_id in available_snips:
                frame_num = extract_frame_number(snip_id)
                if start <= frame_num < end and (frame_num - start) % step == 0:
                    result.append(snip_id)
            return result
    
    return []
```

## Task 4: Create Tutorial Notebook

**Source**: Convert `rebase_instructions/module_3/embryo_metadata_morphseq-tutorial-py.py` to Jupyter

**Key sections to include**:
1. Setup and initialization
2. Loading SAM annotations
3. Adding phenotypes/genotypes
4. Batch operations
5. DEAD phenotype rules
6. Querying data
7. Export options

**Test notebook sections**:
```python
# Cell 1: Initialize
from annotations.embryo_metadata import EmbryoMetadata
em = EmbryoMetadata("grounded_sam_annotations.json", gen_if_no_file=True)
print(f"Loaded {em.embryo_count} embryos")

# Cell 2: Add annotations
embryo_id = list(em.data["embryos"].keys())[0]
em.add_genotype(embryo_id, "tmem67", "sa1423", "homozygous", "user")
em.add_phenotype(embryo_id + "_s0100", "EDEMA", "user", confidence=0.9)

# Cell 3: Batch operations
from annotations.annotation_batch import AnnotationBatch
batch = AnnotationBatch("batch_user", "Test batch")
batch.add_phenotype(embryo_id, "CONVERGENCE_EXTENSION", frames="0000:0200")
results = em.apply_batch(batch)
print(f"Applied {results['applied']} annotations")

# Cell 4: Save
em.save()
```

## Final Integration Test

```python
# Complete pipeline integration
from metadata.experiment_metadata import ExperimentMetadata
from detection_segmentation import GroundedSamAnnotations
from annotations.embryo_metadata import EmbryoMetadata

# 1. Load experiment metadata
exp_meta = ExperimentMetadata("experiment_metadata.json")

# 2. Assuming SAM2 completed
sam_path = "grounded_sam_annotations.json"

# 3. Create embryo metadata
em = EmbryoMetadata(sam_path, gen_if_no_file=True)

# 4. Add biological annotations
embryos = list(em.data["embryos"].keys())[:3]
for i, embryo_id in enumerate(embryos):
    # Alternate genotypes
    gene = "WT" if i % 2 == 0 else "tmem67"
    em.add_genotype(embryo_id, gene, author="pipeline_test")

# 5. Check entity counts
print(em.get_entity_counts())

# 6. Validate and save
em.save()

print("✓ Biological annotation system ready")
```

## Accomplishment Checklist

- [ ] UnifiedManagers uses parsing_utils
- [ ] AnnotationBatch validates IDs correctly
- [ ] EmbryoMetadata loads SAM annotations
- [ ] Entity validation on save
- [ ] Frame resolution uses extract_frame_number
- [ ] DEAD phenotype exclusivity works
- [ ] Batch operations apply successfully
- [ ] Tutorial notebook runs all examples
- [ ] Integration test completes