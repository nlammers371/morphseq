# Module 2: Metadata System

## Overview
Refactor ExperimentMetadata to inherit from BaseAnnotationParser and integrate QC flags directly. Create EmbryoMetadata with phenotype/genotype tracking. Both classes will use the unified entity ID parser for seamless navigation.

## Dependencies
- Module 1: Core Foundation (must be completed first)
- Existing code to refactor: experiment_metadata_utils.py, experiment_data_qc_utils.py
- No external dependencies beyond Python stdlib

## Files to Create/Modify

```
utils/
└── metadata/
    ├── __init__.py
    ├── experiment/
    │   ├── __init__.py
    │   ├── experiment_metadata.py
    │   ├── experiment_qc.py
    │   └── experiment_utils.py
    └── embryo/
        ├── __init__.py
        ├── embryo_metadata.py
        ├── embryo_managers.py
        └── embryo_batch.py
```

## Implementation Steps

### Step 1: Create `utils/metadata/experiment/experiment_metadata.py`

```python
"""Enhanced experiment metadata with integrated QC support."""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import warnings

from ...core import (
    BaseAnnotationParser, parse_entity_id, get_parent_ids,
    get_timestamp, validate_path, QCFlagModel
)

class ExperimentMetadata(BaseAnnotationParser):
    """
    Manages experiment structure and organization with integrated QC.
    
    Tracks hierarchy: experiments → videos → images
    Includes QC flags at all levels (experiment, video, image)
    
    Structure:
    {
        "file_info": {...},
        "experiments": {
            "20240411": {
                "experiment_id": "20240411",
                "videos": {
                    "20240411_A01": {
                        "video_id": "20240411_A01",
                        "well_id": "A01",
                        "images": ["20240411_A01_0000", ...],
                        "qc_flags": []
                    }
                },
                "qc_flags": []
            }
        },
        "qc_definitions": {
            "experiment_level": {...},
            "video_level": {...},
            "image_level": {...}
        }
    }
    """
    
    def __init__(self, filepath: Union[str, Path], 
                 auto_save_interval: Optional[int] = 10,
                 verbose: bool = True):
        """Initialize experiment metadata manager."""
        # Define default QC flag categories
        self.default_qc_definitions = {
            "experiment_level": {
                "POOR_IMAGING_CONDITIONS": "Suboptimal imaging setup",
                "INCOMPLETE": "Experiment was not completed",
                "PROTOCOL_DEVIATION": "Deviation from standard protocol"
            },
            "video_level": {
                "DRY_WELL": "Well dried out during imaging",
                "FOCUS_DRIFT": "Focus problems during acquisition",
                "STAGE_DRIFT": "XY stage position drift",
                "MISSING_FRAMES": "Frames missing from sequence"
            },
            "image_level": {
                "BLUR": "Image is blurry (low Laplacian variance)",
                "DARK": "Image is too dark",
                "OVEREXPOSED": "Image is overexposed", 
                "CORRUPT": "Cannot read image file",
                "EMPTY": "No embryo visible"
            }
        }
        
        super().__init__(filepath, auto_save_interval, verbose)
        
    def _load_or_initialize(self) -> Dict:
        """Load existing metadata or create new structure."""
        if self.filepath.exists():
            data = self.load_json()
            # Migrate old format if needed
            if "experiments" in data and "qc_definitions" not in data:
                data = self._migrate_to_integrated_qc(data)
            return data
            
        return self._create_empty_metadata()
    
    def _create_empty_metadata(self) -> Dict:
        """Create new metadata structure with QC integration."""
        return {
            "file_info": {
                "version": "2.0",  # Version 2 includes integrated QC
                "creation_time": self.get_timestamp(),
                "last_updated": self.get_timestamp(),
                "script_version": "unified_pipeline"
            },
            "experiments": {},
            "qc_definitions": self.default_qc_definitions.copy(),
            "statistics": {
                "total_experiments": 0,
                "total_videos": 0,
                "total_images": 0,
                "last_updated": self.get_timestamp()
            }
        }
    
    def _validate_schema(self, data: Dict) -> None:
        """Validate metadata structure."""
        required_keys = ["file_info", "experiments", "qc_definitions"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
    
    def _migrate_to_integrated_qc(self, old_data: Dict) -> Dict:
        """Migrate from old format to integrated QC format."""
        if self.verbose:
            print("🔄 Migrating to integrated QC format...")
            
        # Add QC definitions if missing
        if "qc_definitions" not in old_data:
            old_data["qc_definitions"] = self.default_qc_definitions.copy()
        
        # Add QC flags arrays to all entities if missing
        for exp_data in old_data.get("experiments", {}).values():
            if "qc_flags" not in exp_data:
                exp_data["qc_flags"] = []
                
            for video_data in exp_data.get("videos", {}).values():
                if "qc_flags" not in video_data:
                    video_data["qc_flags"] = []
                    
                # Convert old image_ids list to new structure if needed
                if "images" in video_data and isinstance(video_data["images"], list):
                    image_dict = {}
                    for img_id in video_data["images"]:
                        image_dict[img_id] = {"qc_flags": []}
                    video_data["images"] = image_dict
        
        old_data["file_info"]["version"] = "2.0"
        return old_data
    
    # -------------------------------------------------------------------------
    # Experiment Management
    # -------------------------------------------------------------------------
    
    def add_experiment(self, experiment_id: str) -> bool:
        """Add a new experiment."""
        if experiment_id in self.data["experiments"]:
            return False
            
        self.data["experiments"][experiment_id] = {
            "experiment_id": experiment_id,
            "created": self.get_timestamp(),
            "last_updated": self.get_timestamp(),
            "videos": {},
            "qc_flags": []
        }
        
        self._update_statistics()
        self.mark_changed()
        
        if self.verbose:
            print(f"✅ Added experiment: {experiment_id}")
        
        return True
    
    def add_video(self, video_id: str, video_info: Optional[Dict] = None) -> bool:
        """Add a video to an experiment."""
        level, components = self.parse_entity_id(video_id, "video")
        exp_id = components["experiment_id"]
        
        # Ensure experiment exists
        if exp_id not in self.data["experiments"]:
            self.add_experiment(exp_id)
        
        exp_data = self.data["experiments"][exp_id]
        
        if video_id in exp_data["videos"]:
            return False
        
        video_data = {
            "video_id": video_id,
            "well_id": components["well_id"],
            "created": self.get_timestamp(),
            "images": {},
            "qc_flags": []
        }
        
        if video_info:
            video_data.update(video_info)
            
        exp_data["videos"][video_id] = video_data
        exp_data["last_updated"] = self.get_timestamp()
        
        self._update_statistics()
        self.mark_changed()
        
        if self.verbose:
            print(f"✅ Added video: {video_id}")
            
        return True
    
    def add_images(self, image_ids: List[str]) -> int:
        """Add multiple images, auto-detecting their videos."""
        added_count = 0
        
        for image_id in image_ids:
            level, components = self.parse_entity_id(image_id, "image")
            exp_id = components["experiment_id"]
            video_id = components["video_id"]
            
            # Ensure video exists
            if video_id not in self.data["experiments"].get(exp_id, {}).get("videos", {}):
                self.add_video(video_id)
            
            video_data = self.data["experiments"][exp_id]["videos"][video_id]
            
            if image_id not in video_data["images"]:
                video_data["images"][image_id] = {
                    "qc_flags": [],
                    "frame_number": components["frame_number"]
                }
                added_count += 1
        
        if added_count > 0:
            self._update_statistics()
            self.mark_changed()
            
        return added_count
    
    # -------------------------------------------------------------------------
    # QC Flag Management
    # -------------------------------------------------------------------------
    
    def add_qc_flag(self, entity_id: str, flag: str, author: str, 
                    notes: str = "", severity: str = "warning") -> bool:
        """Add QC flag to any entity (auto-detects level)."""
        level, components = self.parse_entity_id(entity_id)
        
        # Validate flag
        qc_level = f"{level}_level"
        valid_flags = self.data["qc_definitions"].get(qc_level, {})
        
        if flag not in valid_flags:
            if self.verbose:
                print(f"⚠️  Unknown flag '{flag}' for {level} level")
            return False
        
        flag_entry = {
            "flag": flag,
            "author": author,
            "timestamp": self.get_timestamp(),
            "notes": notes,
            "severity": severity
        }
        
        # Add flag based on level
        if level == "experiment":
            exp_data = self.data["experiments"].get(entity_id)
            if exp_data:
                exp_data["qc_flags"].append(flag_entry)
                self.mark_changed()
                return True
                
        elif level == "video":
            exp_id = components["experiment_id"]
            video_data = self.data["experiments"].get(exp_id, {}).get("videos", {}).get(entity_id)
            if video_data:
                video_data["qc_flags"].append(flag_entry)
                self.mark_changed()
                return True
                
        elif level == "image":
            video_id = components["video_id"]
            exp_id = components["experiment_id"]
            image_data = (self.data["experiments"].get(exp_id, {})
                         .get("videos", {}).get(video_id, {})
                         .get("images", {}).get(entity_id))
            if image_data:
                image_data["qc_flags"].append(flag_entry)
                self.mark_changed()
                return True
        
        return False
    
    def get_qc_flags(self, entity_id: str) -> List[Dict]:
        """Get all QC flags for an entity."""
        level, components = self.parse_entity_id(entity_id)
        
        if level == "experiment":
            return self.data["experiments"].get(entity_id, {}).get("qc_flags", [])
        elif level == "video":
            exp_id = components["experiment_id"]
            return (self.data["experiments"].get(exp_id, {})
                   .get("videos", {}).get(entity_id, {})
                   .get("qc_flags", []))
        elif level == "image":
            video_id = components["video_id"]
            exp_id = components["experiment_id"]
            return (self.data["experiments"].get(exp_id, {})
                   .get("videos", {}).get(video_id, {})
                   .get("images", {}).get(entity_id, {})
                   .get("qc_flags", []))
        
        return []
    
    def _update_statistics(self):
        """Update summary statistics."""
        total_exp = len(self.data["experiments"])
        total_vid = sum(len(exp.get("videos", {})) for exp in self.data["experiments"].values())
        total_img = sum(
            len(vid.get("images", {}))
            for exp in self.data["experiments"].values()
            for vid in exp.get("videos", {}).values()
        )
        
        self.data["statistics"] = {
            "total_experiments": total_exp,
            "total_videos": total_vid,
            "total_images": total_img,
            "last_updated": self.get_timestamp()
        }
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """Get entity data by ID."""
        level, components = self.parse_entity_id(entity_id)
        
        if level == "experiment":
            return self.data["experiments"].get(entity_id)
        elif level == "video":
            exp_id = components["experiment_id"]
            return self.data["experiments"].get(exp_id, {}).get("videos", {}).get(entity_id)
        elif level == "image":
            exp_id = components["experiment_id"]
            video_id = components["video_id"]
            return (self.data["experiments"].get(exp_id, {})
                   .get("videos", {}).get(video_id, {})
                   .get("images", {}).get(entity_id))
        
        return None
```

### Step 2: Create `utils/metadata/embryo/embryo_metadata.py`

```python
"""
EmbryoMetadata for tracking phenotypes, genotypes, treatments, and flags.
Implements all requirements from embryometada_class_specs.txt
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import defaultdict
import random

from ...core import (
    BaseAnnotationParser, parse_entity_id, get_parent_ids,
    get_timestamp, validate_path
)

class EmbryoMetadata(BaseAnnotationParser):
    """
    Tracks embryo-specific metadata including:
    - Phenotypes (temporal, at snip level)
    - Genotypes (embryo level)
    - Treatments (experiment/embryo level)
    - Flags (multi-level: snip, video, image, experiment)
    - Source tracking (lineage from experiments)
    - Configuration (model configs, GSAM IDs)
    
    Structure follows requirements from embryometada_class_specs.txt
    """
    
    def __init__(self, sam_annotation_path: Union[str, Path],
                 embryo_metadata_path: Optional[Union[str, Path]] = None,
                 gen_if_no_file: bool = False,
                 auto_validate: bool = True,
                 verbose: bool = True):
        """Initialize EmbryoMetadata with SAM annotation linkage."""
        self.sam_annotation_path = validate_path(sam_annotation_path, must_exist=True)
        
        # Auto-generate metadata path if not provided
        if embryo_metadata_path is None:
            embryo_metadata_path = self.sam_annotation_path.with_name(
                self.sam_annotation_path.stem + "_embryo_metadata.json"
            )
        
        self.gen_if_no_file = gen_if_no_file
        self.auto_validate = auto_validate
        
        # Initialize permitted values for validation
        self.permitted_values = self._get_default_permitted_values()
        
        # Load SAM annotations for source data
        self.sam_annotations = self.load_json(self.sam_annotation_path)
        
        super().__init__(embryo_metadata_path, verbose=verbose)
        
        # Perform initial validation if requested
        if self.auto_validate:
            self._validate_consistency()
    
    def _get_default_permitted_values(self) -> Dict:
        """Define default permitted values for all fields."""
        return {
            "phenotypes": {
                "NONE": "No phenotype observed (default)",
                "DEAD": "Embryo is dead (cannot coexist with other phenotypes)",
                "EDEMA": "Pericardial or yolk edema",
                "BODY_AXIS": "Body axis defects",
                "CONVERGENCE_EXTENSION": "CE defects",
                "HEART_DEFECT": "Heart morphology or function defects",
                "BRAIN_DEFECT": "Brain morphology defects",
                "EYE_DEFECT": "Eye development defects",
                "TAIL_DEFECT": "Tail morphology defects",
                "PIGMENTATION": "Pigmentation defects",
                "MOVEMENT_DEFECT": "Abnormal or absent movement"
            },
            "genotypes": {
                "WT": "Wild type",
                "HET": "Heterozygous",
                "HOM": "Homozygous",
                "UNKNOWN": "Genotype unknown"
            },
            "treatments": {
                "CONTROL": "Control/untreated",
                "DMSO": "DMSO vehicle control",
                "HEAT_SHOCK": "Heat shock treatment",
                "COLD_SHOCK": "Cold shock treatment",
                "DRUG_TREATED": "Drug treatment (specify in notes)",
                "MORPHOLINO": "Morpholino injection",
                "CRISPR": "CRISPR injection",
                "MRNA_INJECTION": "mRNA injection"
            },
            "flags": {
                "snip_level": {
                    "MOTION_BLUR": "Snip has motion blur",
                    "MASK_ON_EDGE": "Embryo mask touches image edge",
                    "HIGHLY_VAR_MASK": "Mask area varies >10% from average",
                    "DETECTION_MISSING": "Embryo not detected in this frame"
                },
                "video_level": {
                    "NONZERO_SEED_FRAME": "Seed frame is not first frame",
                    "NO_EMBRYO_DETECTED": "No embryo detected in video",
                    "TRACKING_FAILURE": "Tracking lost during video"
                },
                "image_level": {
                    "MULTIPLE_EMBRYOS": "Multiple embryos in single image",
                    "PARTIAL_EMBRYO": "Only part of embryo visible"
                },
                "experiment_level": {
                    "TREATMENT_VARIATION": "Treatment conditions varied",
                    "INCOMPLETE_TRACKING": "Not all embryos tracked"
                }
            },
            "severity_levels": ["info", "warning", "error", "critical"]
        }
    
    def _load_or_initialize(self) -> Dict:
        """Load existing metadata or initialize from SAM annotations."""
        if self.filepath.exists():
            data = self.load_json()
            self._validate_schema(data)
            return data
        elif self.gen_if_no_file:
            return self._initialize_from_sam()
        else:
            raise FileNotFoundError(
                f"Embryo metadata not found at {self.filepath} and gen_if_no_file=False"
            )
    
    def _initialize_from_sam(self) -> Dict:
        """Initialize embryo metadata structure from SAM annotations."""
        gsam_id = self.sam_annotations.get("gsam_annotation_id", self._generate_gsam_id())
        
        metadata = {
            "file_info": {
                "version": "1.0",
                "creation_time": self.get_timestamp(),
                "last_updated": self.get_timestamp(),
                "source_sam_annotation": str(self.sam_annotation_path),
                "gsam_annotation_id": gsam_id
            },
            "permitted_values": self.permitted_values,
            "embryos": {},
            "treatments": {},  # Experiment-level treatments
            "flags": {
                "experiment": {},
                "video": {},
                "image": {},
                "snip": {}
            },
            "config": {}
        }
        
        # Extract embryo structure from SAM annotations
        if "experiments" in self.sam_annotations:
            for exp_id, exp_data in self.sam_annotations["experiments"].items():
                # Initialize experiment-level treatment
                metadata["treatments"][exp_id] = {
                    "treatment": "CONTROL",
                    "author": "system_init",
                    "timestamp": self.get_timestamp(),
                    "notes": ""
                }
                
                for video_id, video_data in exp_data.get("videos", {}).items():
                    for embryo_id in video_data.get("embryo_ids", []):
                        if embryo_id not in metadata["embryos"]:
                            metadata["embryos"][embryo_id] = self._create_embryo_entry(
                                embryo_id, exp_id, video_id
                            )
                        
                        # Add snips for this embryo
                        for image_id, image_data in video_data.get("images", {}).items():
                            embryos_in_image = image_data.get("embryos", {})
                            if embryo_id in embryos_in_image:
                                snip_id = embryos_in_image[embryo_id].get("snip_id")
                                if snip_id:
                                    metadata["embryos"][embryo_id]["snips"][snip_id] = {
                                        "phenotype": {
                                            "value": "NONE",
                                            "author": "system_init",
                                            "timestamp": self.get_timestamp(),
                                            "confidence": 1.0
                                        },
                                        "flags": [],
                                        "frame_number": parse_entity_id(snip_id)[1]["frame_number"]
                                    }
        
        # Inherit model configurations
        if "config" in self.sam_annotations:
            metadata["config"] = self.sam_annotations["config"].copy()
        
        return metadata
    
    def _create_embryo_entry(self, embryo_id: str, exp_id: str, video_id: str) -> Dict:
        """Create a new embryo entry with proper structure."""
        return {
            "embryo_id": embryo_id,
            "genotype": {
                "value": None,
                "gene": None,
                "author": None,
                "timestamp": None,
                "notes": ""
            },
            "treatment": {
                "value": None,  # Can override experiment-level treatment
                "author": None,
                "timestamp": None,
                "notes": ""
            },
            "phenotypes": {},  # Will be populated at snip level
            "flags": {},       # Will be populated at multiple levels
            "metadata": {
                "created": self.get_timestamp(),
                "last_updated": self.get_timestamp()
            },
            "source": {
                "experiment_id": exp_id,
                "video_id": video_id,
                "sam_annotation_source": str(self.sam_annotation_path)
            },
            "snips": {}
        }
    
    def _validate_schema(self, data: Dict) -> None:
        """Validate metadata structure against expected schema."""
        required_keys = ["file_info", "permitted_values", "embryos", "flags", "config"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate permitted values structure
        required_categories = ["phenotypes", "genotypes", "treatments", "flags"]
        for cat in required_categories:
            if cat not in data["permitted_values"]:
                raise ValueError(f"Missing permitted values category: {cat}")
    
    def _generate_gsam_id(self) -> int:
        """Generate a 4-digit GSAM ID."""
        return random.randint(1000, 9999)
    
    # -------------------------------------------------------------------------
    # Phenotype Management (Temporal at Snip Level)
    # -------------------------------------------------------------------------
    
    def add_phenotype(self, snip_id: str, phenotype: str, author: str,
                     confidence: float = 1.0, notes: str = "",
                     overwrite_dead: bool = False) -> bool:
        """
        Add phenotype to a snip (temporal tracking).
        
        Special handling for DEAD phenotype:
        - Cannot coexist with other phenotypes
        - Cannot be overwritten unless overwrite_dead=True
        """
        # Validate phenotype
        if phenotype not in self.permitted_values["phenotypes"]:
            raise ValueError(f"Invalid phenotype: {phenotype}")
        
        # Find embryo containing this snip
        embryo_id = self._get_embryo_id_from_snip(snip_id)
        if not embryo_id:
            return False
        
        embryo_data = self.data["embryos"][embryo_id]
        snip_data = embryo_data["snips"].get(snip_id)
        
        if not snip_data:
            return False
        
        # Check for DEAD phenotype restrictions
        current_phenotype = snip_data["phenotype"]["value"]
        
        if current_phenotype == "DEAD" and not overwrite_dead:
            if self.verbose:
                print(f"⚠️  Cannot overwrite DEAD phenotype without overwrite_dead=True")
            return False
        
        if phenotype == "DEAD" and current_phenotype not in ["NONE", "DEAD"]:
            if self.verbose:
                print(f"⚠️  DEAD phenotype cannot coexist with {current_phenotype}")
            return False
        
        # Update phenotype
        snip_data["phenotype"] = {
            "value": phenotype,
            "author": author,
            "timestamp": self.get_timestamp(),
            "confidence": confidence,
            "notes": notes
        }
        
        # Update embryo metadata
        embryo_data["metadata"]["last_updated"] = self.get_timestamp()
        
        # If DEAD, propagate to all subsequent snips
        if phenotype == "DEAD":
            self._propagate_dead_phenotype(embryo_id, snip_id, author)
        
        self.mark_changed()
        return True
    
    def _propagate_dead_phenotype(self, embryo_id: str, start_snip_id: str, author: str):
        """Propagate DEAD phenotype to all subsequent snips."""
        embryo_data = self.data["embryos"][embryo_id]
        snips = embryo_data["snips"]
        
        # Get frame number of death
        death_frame = int(snips[start_snip_id]["frame_number"])
        
        for snip_id, snip_data in snips.items():
            if int(snip_data["frame_number"]) > death_frame:
                snip_data["phenotype"] = {
                    "value": "DEAD",
                    "author": author,
                    "timestamp": self.get_timestamp(),
                    "confidence": 1.0,
                    "notes": f"Propagated from death at frame {death_frame}"
                }
    
    # -------------------------------------------------------------------------
    # Genotype Management (Embryo Level)
    # -------------------------------------------------------------------------
    
    def add_genotype(self, embryo_id: str, genotype: str, author: str,
                    gene: Optional[str] = None, notes: str = "",
                    overwrite_genotype: bool = False) -> bool:
        """Add or update genotype for an embryo."""
        if genotype not in self.permitted_values["genotypes"]:
            raise ValueError(f"Invalid genotype: {genotype}")
        
        embryo_data = self.data["embryos"].get(embryo_id)
        if not embryo_data:
            return False
        
        # Check if genotype exists
        if embryo_data["genotype"]["value"] and not overwrite_genotype:
            if self.verbose:
                print(f"⚠️  Genotype already set. Use overwrite_genotype=True to change.")
            return False
        
        embryo_data["genotype"] = {
            "value": genotype,
            "gene": gene,
            "author": author,
            "timestamp": self.get_timestamp(),
            "notes": notes
        }
        
        embryo_data["metadata"]["last_updated"] = self.get_timestamp()
        self.mark_changed()
        
        return True
    
    # -------------------------------------------------------------------------
    # Treatment Management (Experiment or Embryo Level)
    # -------------------------------------------------------------------------
    
    def add_treatment(self, entity_id: str, treatment: str, author: str,
                     notes: str = "", concentration: Optional[str] = None,
                     duration: Optional[str] = None) -> bool:
        """Add treatment at experiment or embryo level."""
        if treatment not in self.permitted_values["treatments"]:
            raise ValueError(f"Invalid treatment: {treatment}")
        
        level, components = self.parse_entity_id(entity_id)
        
        treatment_data = {
            "value": treatment,
            "author": author,
            "timestamp": self.get_timestamp(),
            "notes": notes,
            "concentration": concentration,
            "duration": duration
        }
        
        if level == "experiment":
            self.data["treatments"][entity_id] = treatment_data
            self.mark_changed()
            return True
            
        elif level == "embryo":
            embryo_data = self.data["embryos"].get(entity_id)
            if embryo_data:
                embryo_data["treatment"] = treatment_data
                embryo_data["metadata"]["last_updated"] = self.get_timestamp()
                self.mark_changed()
                return True
        
        return False
    
    # -------------------------------------------------------------------------
    # Flag Management (Multi-level)
    # -------------------------------------------------------------------------
    
    def add_flag(self, entity_id: str, flag: str, author: str,
                details: str = "", severity: str = "warning") -> bool:
        """Add flag at any level (auto-detects from entity_id)."""
        level, components = self.parse_entity_id(entity_id)
        
        # Map entity level to flag level
        flag_level_map = {
            "experiment": "experiment_level",
            "video": "video_level", 
            "image": "image_level",
            "snip": "snip_level"
        }
        
        flag_level = flag_level_map.get(level)
        if not flag_level:
            return False
        
        # Validate flag
        valid_flags = self.permitted_values["flags"].get(flag_level, {})
        if flag not in valid_flags:
            if self.verbose:
                print(f"⚠️  Invalid flag '{flag}' for {flag_level}")
            return False
        
        flag_entry = {
            "flag": flag,
            "author": author,
            "timestamp": self.get_timestamp(),
            "details": details,
            "severity": severity
        }
        
        # Store flag based on level
        if level in ["experiment", "video", "image"]:
            flag_list = self.data["flags"][level].setdefault(entity_id, [])
            flag_list.append(flag_entry)
            self.mark_changed()
            return True
            
        elif level == "snip":
            # Find embryo containing this snip
            embryo_id = self._get_embryo_id_from_snip(entity_id)
            if embryo_id:
                snip_data = self.data["embryos"][embryo_id]["snips"].get(entity_id)
                if snip_data:
                    snip_data["flags"].append(flag_entry)
                    self.mark_changed()
                    return True
        
        return False
    
    def _get_embryo_id_from_snip(self, snip_id: str) -> Optional[str]:
        """Find which embryo contains a given snip."""
        # Parse snip_id to extract embryo_id
        level, components = self.parse_entity_id(snip_id, "snip")
        return components.get("embryo_id")
    
    # -------------------------------------------------------------------------
    # Batch Operations Support
    # -------------------------------------------------------------------------
    
    def batch_add_phenotypes(self, assignments: List[Dict], author: str) -> Dict:
        """Batch assign phenotypes with temporal range support."""
        results = {"success": 0, "failed": 0, "skipped": 0}
        
        for assignment in assignments:
            embryo_id = assignment["embryo_id"]
            phenotype = assignment["phenotype"]
            frames = assignment.get("frames", "all")
            
            # Parse temporal range
            snip_ids = self._parse_temporal_range(embryo_id, frames)
            
            for snip_id in snip_ids:
                success = self.add_phenotype(
                    snip_id, phenotype, author,
                    confidence=assignment.get("confidence", 1.0),
                    notes=assignment.get("notes", "")
                )
                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1
        
        return results
    
    def _parse_temporal_range(self, embryo_id: str, range_spec: str) -> List[str]:
        """
        Parse temporal range specification.
        
        Examples:
            "all" -> all snips
            "[10:20]" -> snips from frame 10 to 20
            "death:" -> from death frame onward
            "[23::]" -> from frame 23 to end
        """
        embryo_data = self.data["embryos"].get(embryo_id, {})
        all_snips = sorted(embryo_data.get("snips", {}).items(), 
                          key=lambda x: int(x[1]["frame_number"]))
        
        if range_spec == "all":
            return [s[0] for s in all_snips]
        
        # Parse range notation
        if range_spec.startswith("[") and range_spec.endswith("]"):
            range_part = range_spec[1:-1]
            if "::" in range_part:
                # Open-ended range
                start = int(range_part.split("::")[0])
                return [s[0] for s in all_snips if int(s[1]["frame_number"]) >= start]
            elif ":" in range_part:
                # Closed range
                start, end = map(int, range_part.split(":"))
                return [s[0] for s in all_snips 
                       if start <= int(s[1]["frame_number"]) <= end]
        
        return []
    
    # -------------------------------------------------------------------------
    # Query and Reporting
    # -------------------------------------------------------------------------
    
    def get_missing_genotypes(self) -> Dict[str, List[str]]:
        """Find embryos missing genotype data, grouped by experiment."""
        missing = defaultdict(list)
        
        for embryo_id, embryo_data in self.data["embryos"].items():
            if not embryo_data["genotype"]["value"]:
                exp_id = embryo_data["source"]["experiment_id"]
                missing[exp_id].append(embryo_id)
        
        if self.verbose and missing:
            total_missing = sum(len(v) for v in missing.values())
            print(f"⚠️  {total_missing} embryos missing genotype data across {len(missing)} experiments")
        
        return dict(missing)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        summary = {
            "total_embryos": len(self.data["embryos"]),
            "total_snips": sum(len(e["snips"]) for e in self.data["embryos"].values()),
            "genotyped": sum(1 for e in self.data["embryos"].values() if e["genotype"]["value"]),
            "phenotyped": sum(
                1 for e in self.data["embryos"].values()
                if any(s["phenotype"]["value"] != "NONE" for s in e["snips"].values())
            ),
            "treatments": len(set(
                t["value"] for t in self.data["treatments"].values()
                if t.get("value")
            )),
            "total_flags": sum(
                len(flags) for flag_list in self.data["flags"].values() 
                for flags in flag_list.values()
            )
        }
        
        return summary
    
    def _validate_consistency(self):
        """Validate consistency between SAM annotations and metadata."""
        if self.verbose:
            print("🔍 Validating consistency with SAM annotations...")
        
        # Check GSAM ID match
        sam_gsam_id = self.sam_annotations.get("gsam_annotation_id")
        our_gsam_id = self.data["file_info"].get("gsam_annotation_id")
        
        if sam_gsam_id and our_gsam_id and sam_gsam_id != our_gsam_id:
            warnings.warn(f"GSAM ID mismatch: SAM={sam_gsam_id}, Metadata={our_gsam_id}")
```

### Step 3: Create `utils/metadata/__init__.py`

```python
"""Metadata management modules."""

from .experiment.experiment_metadata import ExperimentMetadata
from .embryo.embryo_metadata import EmbryoMetadata

__all__ = ['ExperimentMetadata', 'EmbryoMetadata']
```

## Testing Checklist

- [ ] Test ExperimentMetadata with integrated QC
- [ ] Test migration from old format
- [ ] Test EmbryoMetadata initialization from SAM
- [ ] Test phenotype addition with DEAD propagation
- [ ] Test genotype management with overwrite protection
- [ ] Test treatment at both experiment and embryo levels
- [ ] Test multi-level flag system
- [ ] Test batch operations with temporal ranges
- [ ] Test consistency validation
- [ ] Test GSAM ID linking between SAM and metadata

## Implementation Log

| Date | Developer | Task | Status |
|------|-----------|------|--------|
| TBD | TBD | Create experiment_metadata.py with QC | Pending |
| TBD | TBD | Create embryo_metadata.py with all fields | Pending |
| TBD | TBD | Create manager classes | Pending |
| TBD | TBD | Create batch processing utilities | Pending |
| TBD | TBD | Unit tests for all components | Pending |
| TBD | TBD | Integration tests with SAM data | Pending |

## Notes for Implementer

1. **CRITICAL**: EmbryoMetadata must include ALL fields from specs:
   - Phenotypes (temporal at snip level)
   - Genotypes (embryo level)
   - **Treatments** (experiment or embryo level)
   - Flags (multi-level)
   - Source tracking
   - Config inheritance

2. **Phenotype Rules**:
   - DEAD phenotype cannot coexist with others
   - DEAD propagates to all subsequent frames
   - Default phenotype is NONE
   - Temporal tracking at snip level

3. **Genotype Protection**:
   - Requires overwrite_genotype=True to change
   - Warning issued for missing genotypes
   - Tracked at embryo level

4. **Treatment Flexibility**:
   - Can be set at experiment level (applies to all)
   - Can be overridden at individual embryo level
   - Includes concentration and duration fields

5. **Flag System**:
   - Different valid flags for each level
   - Severity levels: info, warning, error, critical
   - Stored separately from embryo data for efficiency

6. **Temporal Range Syntax**:
   ```
   "all" -> all frames
   "[10:20]" -> frames 10-20
   "[23::]" -> frame 23 to end
   "death:" -> from death onward
   ```

## Usage Examples

```python
# ExperimentMetadata with integrated QC
exp_meta = ExperimentMetadata("experiment_metadata.json")
exp_meta.add_experiment("20240411")
exp_meta.add_video("20240411_A01", {"mp4_path": "/path/to/video.mp4"})
exp_meta.add_qc_flag("20240411_A01", "DRY_WELL", "analyst", "Well dried at frame 50")

# EmbryoMetadata with full feature set
embryo_meta = EmbryoMetadata(
    sam_annotation_path="sam_annotations.json",
    gen_if_no_file=True
)

# Add treatment at experiment level
embryo_meta.add_treatment("20240411", "HEAT_SHOCK", "researcher", 
                         notes="37°C for 30 minutes", duration="30min")

# Add genotype
embryo_meta.add_genotype("20240411_A01_e01", "HOM", "geneticist", 
                        gene="lmx1b", notes="Confirmed by PCR")

# Add temporal phenotype
embryo_meta.add_phenotype("20240411_A01_e01_s0042", "EDEMA", "observer",
                         confidence=0.95, notes="Mild pericardial edema")

# Batch operations with temporal ranges
assignments = [
    {
        "embryo_id": "20240411_A01_e01",
        "phenotype": "DEAD",
        "frames": "[45::]",  # Death from frame 45 onward
        "confidence": 1.0
    }
]
embryo_meta.batch_add_phenotypes(assignments, "batch_analyst")

# Multi-level flags
embryo_meta.add_flag("20240411_A01_e01_s0042", "MASK_ON_EDGE", "qc_system")
embryo_meta.add_flag("20240411_A01", "NONZERO_SEED_FRAME", "qc_system")
embryo_meta.add_flag("20240411", "INCOMPLETE_TRACKING", "supervisor")
```

## Key Design Decisions

1. **Unified Entity Parser**: All entity IDs parsed with single function
2. **Integrated QC**: QC flags part of ExperimentMetadata, not separate
3. **Flexible Treatments**: Can be applied at multiple levels
4. **Temporal Phenotypes**: Tracked per frame for detailed analysis
5. **GSAM ID Linking**: Ensures traceability between annotation files
        