# Module 2: Metadata System Implementation Guide

## Overview

This module builds directly on Module 1's `BaseAnnotationParser` and leverages the existing data models. The implementation seamlessly extends the base functionality while keeping the API intuitive and straightforward.

**Key Design Principle**: Build on what exists - don't reinvent. The `BaseAnnotationParser` already provides ID parsing, file I/O, change tracking, and entity navigation. We just add domain-specific logic.

## Architecture Overview

```
Module 1 (Already Implemented):
├── BaseAnnotationParser     # Core functionality for all parsers
├── Data Models             # Phenotype, Genotype, Flag, Treatment classes
└── ID Parsing Utils        # parse_snip_id, parse_embryo_id, etc.

Module 2 (This Implementation):
├── ExperimentMetadata      # Extends BaseAnnotationParser
├── EmbryoMetadata         # Extends BaseAnnotationParser + Mixins
└── Manager Mixins         # Domain-specific operations
```
much of the implementation of this class is inspired by morphseq/segmentation_sandbox/scripts/utils/embryo_metada_dev_instruction
folder, when looking for inspiration and function definitions to help rebase the code feel free to utilize this. 

## Phase 1: ExperimentMetadata with Integrated QC

### Step 1: Create `utils/metadata/experiment/experiment_metadata.py`

```python
"""
ExperimentMetadata - manages experiment structure with integrated image QC.
Inherits all base functionality from BaseAnnotationParser.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
from ...core import BaseAnnotationParser, parse_entity_id, get_parent_ids

class ExperimentMetadata(BaseAnnotationParser):
    """
    Manages experiment → video → image hierarchy with integrated image QC.
    
    Inherits from BaseAnnotationParser:
    - Automatic backup on save
    - Change tracking
    - Entity navigation (get_entity, ensure_entity_exists)
    - Batch operations
    - GSAM ID management
    """
    
    def __init__(self, filepath: Union[str, Path], **kwargs):
        # Define default image QC definitions
        self.image_qc_definitions = {
            "image_level": {
                "BLUR": "Image is blurry",
                "DARK": "Image is too dark", 
                "OVEREXPOSED": "Image is overexposed",
                "CORRUPT": "Cannot read image file"
            },
            "video_level": {
                "FOCUS_DRIFT": "Focus problems during acquisition",
                "STAGE_DRIFT": "XY stage position drift"
            }
        }
        super().__init__(filepath, **kwargs)
    
    def _load_or_initialize(self) -> Dict:
        """Load or create new metadata structure."""
        if self.filepath.exists():
            return self.load_json()  # From BaseAnnotationParser
        
        return {
            "file_info": {
                "version": "2.0",
                "creation_time": self.get_timestamp(),  # From BaseAnnotationParser
                "gsam_annotation_id": self.ensure_gsam_id()  # From BaseAnnotationParser
            },
            "experiments": {},
            "image_qc_definitions": self.image_qc_definitions
        }
    
    def _validate_schema(self, data: Dict) -> None:
        """Validate structure - required by BaseAnnotationParser."""
        required = ["file_info", "experiments"]
        for key in required:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
    
    # -------------------------------------------------------------------------
    # Experiment/Video/Image Management (using BaseAnnotationParser methods)
    # -------------------------------------------------------------------------
    
    def add_video(self, video_id: str, metadata: Optional[Dict] = None) -> bool:
        """
        Add video to experiment structure.
        
        Uses BaseAnnotationParser.ensure_entity_exists internally.
        """
        default_data = {
            "video_id": video_id,
            "created": self.get_timestamp(),
            "images": {},
            "qc_flags": [],
            **(metadata or {})
        }
        
        # ensure_entity_exists handles all the hierarchy creation
        self.ensure_entity_exists(video_id, "video", default_data)
        
        if self.verbose:
            print(f"✅ Added video: {video_id}")
        
        return True
    
    def add_images_batch(self, image_ids: List[str]) -> int:
        """
        Add multiple images efficiently using batch operations.
        
        Leverages BaseAnnotationParser.process_entities_batch.
        """
        # Group by video for efficiency (from base_utils)
        from ...core import group_entities_by_parent
        grouped = group_entities_by_parent(image_ids, "image")
        
        added = 0
        for video_id, images in grouped.items():
            # Ensure video exists
            self.ensure_entity_exists(video_id, "video", {"images": {}})
            
            # Add images to video
            updates = {img_id: {"added": self.get_timestamp()} for img_id in images}
            added += self.update_entities_batch(updates, "image")
        
        return added
    
    # -------------------------------------------------------------------------
    # Image QC Integration (Step 02 and before)
    # -------------------------------------------------------------------------
    
    def add_image_qc_flag(self, entity_id: str, flag: str, author: str,
                         details: str = "", severity: str = "warning") -> bool:
        """
        Add image integrity QC flag.
        
        Uses the Flag data model from Module 1.
        """
        # Import Flag model from Module 1
        from ...core.base_models import Flag
        
        # Determine level from entity_id
        level, components = parse_entity_id(entity_id)
        
        # Validate flag
        level_key = f"{level}_level"
        if flag not in self.data.get("image_qc_definitions", {}).get(level_key, {}):
            raise ValueError(f"Invalid {level} QC flag: {flag}")
        
        # Create Flag instance
        flag_obj = Flag(
            value=flag,
            author=author,
            notes=details,
            severity=severity,
            flag_type="image_integrity"
        )
        
        # Get entity and add flag
        entity = self.ensure_entity_exists(entity_id, level, {"qc_flags": []})
        entity["qc_flags"].append(flag_obj.to_dict())
        
        self.mark_changed()
        return True
    
    def get_images_by_qc_status(self, flag: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Find images by QC status.
        
        Returns:
            Dict mapping flag -> list of image IDs
        """
        results = {}
        
        # Use get_summary_stats from BaseAnnotationParser to find all images
        for exp_data in self.data.get("experiments", {}).values():
            for video_data in exp_data.get("videos", {}).values():
                for image_id, image_data in video_data.get("images", {}).items():
                    for qc_flag in image_data.get("qc_flags", []):
                        flag_name = qc_flag["value"]
                        if flag is None or flag_name == flag:
                            if flag_name not in results:
                                results[flag_name] = []
                            results[flag_name].append(image_id)
        
        return results
```

## Phase 2: EmbryoMetadata with Mixin Architecture

### Step 2: Create `utils/metadata/embryo/embryo_metadata.py`

```python
"""
EmbryoMetadata - the main class that combines BaseAnnotationParser with manager mixins.
This is the refactored version from the original implementation.
"""

# Import from embryo_metadata_refactored.py (the 436-line version)
# This already perfectly demonstrates the mixin architecture
# Key sections:

from ...core import BaseAnnotationParser, parse_entity_id, get_parent_ids
from .data_managers.permitted_values_manager import PermittedValuesManager
from .data_managers.embryo_phenotype_manager import EmbryoPhenotypeManager
from .data_managers.embryo_genotype_manager import EmbryoGenotypeManager
from .data_managers.embryo_flag_manager import EmbryoFlagManager
from .data_managers.embryo_treatment_manager import EmbryoTreatmentManager

class EmbryoMetadata(BaseAnnotationParser,
                    EmbryoPhenotypeManager,
                    EmbryoGenotypeManager,
                    EmbryoFlagManager,
                    EmbryoTreatmentManager):
    """
    Main class combining base functionality with domain-specific managers.
    
    From BaseAnnotationParser:
    - File I/O with atomic saves and backups
    - Change tracking and auto-save
    - Entity navigation and batch operations
    - GSAM ID management
    
    From Mixins:
    - Phenotype temporal tracking (EmbryoPhenotypeManager)
    - Genotype single-value enforcement (EmbryoGenotypeManager)
    - Multi-level flag system (EmbryoFlagManager)
    - Treatment hierarchy (EmbryoTreatmentManager)
    """
    
    # Implementation from embryo_metadata_refactored.py lines 71-168
    # Key initialization that shows integration:
    
    def __init__(self, sam_annotation_path, embryo_metadata_path=None, **kwargs):
        # Initialize schema manager for validation
        self.schema_manager = PermittedValuesManager()
        self.permitted_values = self.schema_manager.schema
        
        # Load SAM annotations for structure
        self.sam_annotation_path = validate_path(sam_annotation_path)
        self.sam_annotations = self.load_json(self.sam_annotation_path)
        
        # Initialize base class
        super().__init__(embryo_metadata_path or self._auto_generate_path(), **kwargs)
        
        # The base class handles all file I/O, the mixins add domain logic
```

### Step 3: Manager Mixins Using Data Models

#### `embryo_phenotype_manager.py` (Key Methods)

```python
"""Uses Phenotype data model from Module 1."""

from ...core.base_models import Phenotype

class EmbryoPhenotypeManager:
    """
    Phenotype operations leveraging BaseAnnotationParser navigation.
    
    Key insight: We don't reimplement entity finding - we use
    get_entity() and ensure_entity_exists() from the base class.
    """
    
    def add_phenotype(self, snip_id: str, phenotype: str, author: str, **kwargs):
        # Create Phenotype instance using Module 1's data model
        phenotype_obj = Phenotype(
            value=phenotype,
            author=author,
            confidence=kwargs.get('confidence', 1.0),
            notes=kwargs.get('notes', '')
        )
        
        # Validate against schema
        phenotype_obj.validate(self.schema_manager.get_all_values('phenotypes'))
        
        # Use BaseAnnotationParser's entity navigation
        # get_embryo_id_from_snip uses parse_snip_id from Module 1
        embryo_id = self.get_embryo_id_from_snip(snip_id)
        
        # ensure_entity_exists creates the structure if needed
        embryo = self.ensure_entity_exists(embryo_id, "embryo", 
                                         {"snips": {}, "metadata": {}})
        
        # Add phenotype
        if snip_id not in embryo["snips"]:
            embryo["snips"][snip_id] = {}
        
        embryo["snips"][snip_id]["phenotype"] = phenotype_obj.to_dict()
        
        # Handle terminal phenotypes
        if self.schema_manager.is_terminal_phenotype(phenotype):
            self._propagate_terminal_phenotype(embryo_id, snip_id, phenotype, author)
        
        self.mark_changed()  # From BaseAnnotationParser
```

#### `embryo_genotype_manager.py` (Key Methods)

```python
"""Uses Genotype data model from Module 1."""

from ...core.base_models import Genotype

class EmbryoGenotypeManager:
    
    def add_genotype(self, embryo_id: str, gene_name: str, allele: str, **kwargs):
        # Create Genotype instance
        genotype_obj = Genotype(
            value=gene_name,  # gene name as value
            author=kwargs.get('author', 'system'),
            allele=allele,
            zygosity=kwargs.get('zygosity', 'unknown'),
            confidence=kwargs.get('confidence', 1.0),
            method=kwargs.get('method'),
            notes=kwargs.get('notes', '')
        )
        
        # Use base class to get embryo
        embryo = self.ensure_entity_exists(embryo_id, "embryo", {"genotypes": {}})
        
        # Check overwrite protection
        if gene_name in embryo["genotypes"] and not kwargs.get('overwrite', False):
            raise ValueError(f"Genotype for {gene_name} exists. Use overwrite=True.")
        
        embryo["genotypes"][gene_name] = genotype_obj.to_dict()
        self.mark_changed()
```

#### `embryo_flag_manager.py` (Key Methods)

```python
"""Multi-level flags using Flag model."""

from ...core.base_models import Flag

class EmbryoFlagManager:
    
    def add_flag(self, entity_id: str, flag: str, level: str, **kwargs):
        # Create Flag instance
        flag_obj = Flag(
            value=flag,
            author=kwargs.get('author', 'system'),
            flag_type=f"embryo_{level}",
            severity=kwargs.get('severity', 'warning'),
            auto_generated=kwargs.get('auto_generated', False),
            notes=kwargs.get('notes', '')
        )
        
        # Validate flag
        if not self.schema_manager.is_valid_flag(flag, level):
            raise ValueError(f"Invalid {level} flag: {flag}")
        
        # Store at appropriate level
        if level in ["experiment", "video", "image"]:
            # Use BaseAnnotationParser's entity management
            entity = self.ensure_entity_exists(entity_id, level, {"flags": []})
            entity["flags"].append(flag_obj.to_dict())
        
        elif level == "snip":
            # Snips are nested under embryos
            embryo_id = self.get_embryo_id_from_snip(entity_id)
            embryo = self.get_entity(embryo_id, "embryo")
            if entity_id in embryo.get("snips", {}):
                embryo["snips"][entity_id].setdefault("flags", [])
                embryo["snips"][entity_id]["flags"].append(flag_obj.to_dict())
        
        self.mark_changed()
```

#### `embryo_treatment_manager.py` (Key Methods)

```python
"""Treatment management using Treatment model."""

from ...core.base_models import Treatment, TreatmentValue

class EmbryoTreatmentManager:
    
    def add_treatment(self, entity_id: str, treatment: str, author: str, **kwargs):
        # Parse treatment value
        treatment_val = TreatmentValue(
            treatment_type=treatment,
            concentration=kwargs.get('concentration'),
            duration=kwargs.get('duration'),
            temperature=kwargs.get('temperature')
        )
        
        # Create Treatment instance
        treatment_obj = Treatment(
            value=treatment_val,
            author=author,
            details=kwargs.get('details', ''),
            notes=kwargs.get('notes', '')
        )
        
        # Validate
        if not self.schema_manager.is_valid_treatment(treatment):
            raise ValueError(f"Invalid treatment: {treatment}")
        
        # Determine level and store
        level, _ = parse_entity_id(entity_id)
        
        if level == "experiment":
            # Experiment-level treatment
            self.data.setdefault("treatments", {})
            self.data["treatments"][entity_id] = treatment_obj.to_dict()
        
        elif level == "embryo":
            # Embryo-level treatment (overrides experiment)
            embryo = self.ensure_entity_exists(entity_id, "embryo", {})
            embryo["treatment"] = treatment_obj.to_dict()
        
        self.mark_changed()
```

## Phase 3: Batch Processing Integration

### Step 4: Batch Operations
# -------------------------------------------------------------------------
# Lightweight Annotation Batch Utility
# -------------------------------------------------------------------------

class AnnotationBatch:
    """
    Tiny helper for building lists of annotation assignments (phenotype, genotype,
    treatment) before handing them to the high-performance BatchProcessor.
    """

    def __init__(self, batch_type: str):
        self.batch_type = batch_type          # 'phenotype' | 'genotype' | 'treatment'
        self._entries: list[dict] = []

    # --------------------------- public API --------------------------------
    def add(self, **kwargs):
        """
        Append an entry.

        Keys depend on batch_type and match the dictionaries expected by
        BatchOperations:
          - phenotype: embryo_id, phenotype, frames, author, confidence, notes
          - genotype : embryo_id, genotype, gene, author, notes, overwrite
          - treatment: embryo_id, treatment, details, author, notes, …
        """
        self._entries.append(kwargs)

    def to_list(self) -> list[dict]:
        """Return a shallow-copy list ready for BatchOperations."""
        return list(self._entries)

    # --------------------- niceties for inspection -------------------------
    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:               # noqa: D401
        return f"<AnnotationBatch type={self.batch_type} n={len(self)}>"


# Type-specific convenience wrappers with explicit .add() signatures --------
class PhenotypeBatch(AnnotationBatch):
    def __init__(self):
        super().__init__("phenotype")

    def add(
        self,
        embryo_id: str,
        phenotype: str,
        frames: str = "all",
        author: str = "system",
        confidence: float = 1.0,
        notes: str = "",
    ):
        super().add(
            embryo_id=embryo_id,
            phenotype=phenotype,
            frames=frames,
            author=author,
            confidence=confidence,
            notes=notes,
        )


class GenotypeBatch(AnnotationBatch):
    def __init__(self):
        super().__init__("genotype")

    def add(
        self,
        embryo_id: str,
        genotype: str,
        gene: str = "WT",
        author: str = "system",
        notes: str = "",
        overwrite: bool = False,
    ):
        super().add(
            embryo_id=embryo_id,
            genotype=genotype,
            gene=gene,
            author=author,
            notes=notes,
            overwrite=overwrite,
        )


class TreatmentBatch(AnnotationBatch):
    def __init__(self):
        super().__init__("treatment")

    def add(
        self,
        embryo_id: str,
        treatment: str,
        author: str = "system",
        details: str = "",
        notes: str = "",
    ):
        super().add(
            embryo_id=embryo_id,
            treatment=treatment,
            author=author,
            details=details,
            notes=notes,
        )


The batch processing is already implemented in:
- `embryo_metadata_batch.py` - Contains RangeParser, TemporalRangeParser, BatchProcessor
- Key feature: Leverages BaseAnnotationParser's batch methods

```python
# Example from embryo_metadata_batch.py showing integration


### Step 5: AnnotationBatch Utility

Introduce a small helper that lets users accumulate phenotype / genotype / treatment
assignments in a tidy way before handing them to the existing batch engine.

```python
from utils.metadata.batch.annotation_batch import PhenotypeBatch

pbatch = PhenotypeBatch()
pbatch.add(
    embryo_id="20240411_A01_e01",
    phenotype="DEAD",
    frames="[45::]",
    confidence=1.0,
    author="annotator"
)

# When ready, hand it to the engine
em.batch_add_phenotypes(pbatch, author="annotator")

## Phase 4: Integration Layer

### Step 5: SAM/GSAM Linking

From `embryo_metadata_integration.py`:

```python
class GsamIdManager:
    """Manages GSAM IDs for bidirectional linking."""
    
    @staticmethod
    def link_embryo_metadata_to_sam(metadata: EmbryoMetadata, sam_path: Path):
        # Get or create GSAM ID
        gsam_id = metadata.ensure_gsam_id()  # From BaseAnnotationParser
        
        # Add to SAM file
        sam_data = safe_json_load(sam_path)  # From Module 1
        sam_data["file_info"]["gsam_annotation_id"] = gsam_id
        safe_json_save(sam_data, sam_path)  # From Module 1
        
        # Store link in metadata
        metadata.data["file_info"]["linked_sam_annotation"] = str(sam_path)
        metadata.mark_changed()  # From BaseAnnotationParser
        
        return gsam_id
```

## Usage Examples

```python
# ExperimentMetadata - Simple and intuitive
exp_meta = ExperimentMetadata("experiments.json")
exp_meta.add_video("20240411_A01")
exp_meta.add_images_batch(["20240411_A01_0001", "20240411_A01_0002"])
exp_meta.add_image_qc_flag("20240411_A01_0001", "BLUR", "qc_system")
exp_meta.save()  # Automatic backup!

# EmbryoMetadata - Feature-rich but still intuitive
em = EmbryoMetadata(
    sam_annotation_path="sam_annotations.json",
    gen_if_no_file=True  # Auto-create from SAM structure
)

# i think itd be better to have phenotype_batches    
# 
#
python
'''
from ... import phenotpe_batch as p_batch 

pbatch =p_batc.init()
pbatch.add(embryo_id, phenotype, frames, notes, author )
pbatch (shoes its contents)
{
        "embryo_id": "20240411_A01_e01",
        "phenotype": "DEAD",
        "frames": "[45::]",  # From frame 45 onward
        "confidence": 1.0
    },
        {
        "embryo_id": "20240411_A01_e01",
        "phenotype": "DEAD",
        "frames": "[43:]",  # From frame 45 onward
        "confidence": 1.0
    }


# Batch operations with temporal ranges
em.batch_add_phenotypes(pbatch, author) 


'''


# Simple phenotype addition
em.add_phenotype("20240411_A01_e01_s0042",[5::] ,"EDEMA", "researcher") #implementation of range parse class. 

# Genotype with all the options
em.add_genotype(
    "20240411_A01_e01", 
    gene_name="lmx1b",
    allele="mutant",
    zygosity="homozygous",
    notes="PCR confirmed"
)

# Batch operations with temporal ranges
em.batch_add_phenotypes([
    {
        "embryo_id": "20240411_A01_e01",
        "phenotype": "DEAD",
        "frames": "[45::]",  # From frame 45 onward
        "confidence": 1.0
    }
], author="batch_annotator")

# Everything auto-saves at intervals!
```

## Key Advantages of This Architecture

1. **No Duplication**: We leverage BaseAnnotationParser for all common functionality
2. **Clean Separation**: Each mixin handles one concern (phenotypes, genotypes, etc.)
3. **Validation Built-in**: Data models from Module 1 handle validation
4. **Intuitive API**: Methods do what you expect with sensible defaults
5. **Extensible**: Easy to add new managers or extend existing ones

## Implementation Checklist

- [ ] Create ExperimentMetadata extending BaseAnnotationParser
- [ ] Import manager mixins from the refactored implementation
- [ ] Ensure PermittedValuesManager works with centralized schema
- [ ] Test inheritance chain works correctly
- [ ] Verify batch operations use base class methods
- [ ] verify that api is intuitive 
- [ ] verify process of in memory manipulation, then when saving creating a backup, then when done saving delete the backup 
- [ ] Test SAM/GSAM bidirectional linking
- [ ] Validate all data models are used consistently
- [ ] Test auto-save and backup functionality