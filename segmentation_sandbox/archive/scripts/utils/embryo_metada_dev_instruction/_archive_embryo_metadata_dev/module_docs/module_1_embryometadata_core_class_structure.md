# Module 1: EmbryoMetadata Core Class Structure

## Overview
This module defines the core EmbryoMetadata class structure, initialization logic, and basic data management operations.

## Class Definition

```python
class EmbryoMetadata:
    """
    Main class for managing embryo metadata including phenotypes, genotypes, and flags.
    
    This class provides:
    - Hierarchical data storage (experiment → video → image → snip)
    - Validation against permitted values
    - Change tracking and atomic saves
    - Integration with GroundedSamAnnotation data
    - Management of treatment annotations (chemical or temperature)
    """
```

## Core Data Structure

```python
# Internal data structure
{
    "file_info": {
        "version": "1.0",
        "creation_time": "ISO timestamp",
        "last_updated": "ISO timestamp",
        "source_sam_annotation": "path/to/sam_annotation.json",
        "gsam_annotation_id": 1234  # 4-digit identifier
    },
    
    "permitted_values": {
        "phenotypes": {
            "NONE": {"description": "No phenotype", "is_default": true},
            "EDEMA": {"description": "Fluid accumulation", "is_default": false},
            "BODY_AXIS": {"description": "Body axis defect", "is_default": false},
            "CONVERGENCE_EXTENSION": {"description": "CE defect", "is_default": false},
            "DEAD": {"description": "Embryo death", "is_default": false, "exclusive": true}
        },
        "genotypes": {
            # No defaults for genotypes
        },
        "treatments": {
            "CHEMICAL": {"description": "Chemical inhibitor", "is_default": false},
            "TEMPERATURE": {"description": "Temperature treatment", "is_default": false}
        },
        "flags": {
            "snip_level": {
                "MOTION_BLUR": "Motion blur detected",
                "HIGHLY_VAR_MASK": "Mask area variance >10%"
            },
            "image_level": {
                "DETECTION_FAILURE": "Embryo detection failed",
                "MASK_ON_EDGE": "Mask within 5px of edge"
            },
            "video_level": {
                "NONZERO_SEED_FRAME": "Seed frame is not first frame",
                "NO_EMBRYO": "No embryo detected in video"
            },
            "experiment_level": {
                "INCOMPLETE": "Experiment incomplete"
            }
        }
    },
    
    "embryos": {
        "embryo_id": {
            "genotype": {
                "value": "wildtype",
                "author": "researcher1",
                "timestamp": "ISO timestamp",
                "notes": "Confirmed by PCR"
            },
            "treatment": {
                "value": null,
                "author": null,
                "timestamp": "ISO timestamp",
                "notes": null
            },
            "source": {
                "experiment_id": "20240411",
                "video_id": "20240411_A01",
                "sam_annotation_source": "path/to/annotation.json"
            },
            "snips": {
                "snip_id": {
                    "phenotype": {
                        "value": "EDEMA",
                        "author": "researcher1", 
                        "timestamp": "ISO timestamp",
                        "confidence": 0.95  # Optional, for ML predictions
                    },
                    "flags": [{
                        "value": "MOTION_BLUR",
                        "author": "qc_system",
                        "timestamp": "ISO timestamp"
                    }]
                }
            }
        }
    },
    
    "flags": {
        "experiment": {
            "experiment_id": [{flag_objects}]
        },
        "video": {
            "video_id": [{flag_objects}]
        },
        "image": {
            "image_id": [{flag_objects}]
        }
    },
    
    "config": {
        "detection_model": {
            "config": "GroundingDINO_SwinT_OGC.py",
            "weights": "groundingdino_swint_ogc.pth"
        },
        "segmentation_model": {
            "config": "sam2.1_hiera_l.yaml",
            "weights": "sam2.1_hiera_large.pt"
        }
    }
}
```

## Initialization

### Constructor Pseudocode

```python
def __init__(self, sam_annotation_path, embryo_metadata_path=None, 
             gen_if_no_file=False, auto_validate=True, verbose=True):
    """
    Initialize EmbryoMetadata instance.
    
    Steps:
    1. Validate input paths
    2. Load SAM annotations
    3. Load or create embryo metadata
    4. Initialize source tracking
    5. Set up configuration inheritance
    6. Perform consistency checks
    """
    
    # Step 1: Path validation
    self.sam_annotation_path = validate_path(sam_annotation_path, must_exist=True)
    self.embryo_metadata_path = validate_path(embryo_metadata_path)
    
    # Step 2: Load source data
    self.sam_annotations = load_json(self.sam_annotation_path)
    if not self.sam_annotations:
        raise ValueError("Failed to load SAM annotations")
    
    # Step 3: Load or initialize embryo metadata
    if self.embryo_metadata_path.exists():
        self.data = load_json(self.embryo_metadata_path)
        self._validate_schema(self.data)
    elif gen_if_no_file:
        self.data = self._initialize_empty_metadata()
    else:
        raise FileNotFoundError("Embryo metadata not found and gen_if_no_file=False")
    
    # Step 4: Initialize tracking
    self._unsaved_changes = False
    self._change_log = []
    
    # Step 5: Inherit configurations
    self._inherit_configurations()
    
    # Step 6: Consistency checks
    if auto_validate:
        self._run_consistency_checks()
```

### Empty Metadata Initialization

```python
def _initialize_empty_metadata(self):
    """
    Create empty metadata structure with defaults.
    
    Steps:
    1. Create base structure
    2. Import embryo_ids from SAM annotations
    3. Set up permitted values
    4. Initialize empty data stores
    """
    
    metadata = {
        "file_info": {
            "version": "1.0",
            "creation_time": get_timestamp(),
            "last_updated": get_timestamp(),
            "source_sam_annotation": str(self.sam_annotation_path),
            "gsam_annotation_id": generate_gsam_id()  # 4-digit random
        },
        "permitted_values": get_default_permitted_values(),
        "embryos": {},
        "flags": {
            "experiment": {},
            "video": {},
            "image": {}
        },
        "config": {}
    }
    
    # Import embryo structure from SAM annotations
    for exp_id, exp_data in self.sam_annotations["experiments"].items():
        for video_id, video_data in exp_data["videos"].items():
            embryo_ids = video_data.get("embryo_ids", [])
            
            for embryo_id in embryo_ids:
                if embryo_id not in metadata["embryos"]:
                    metadata["embryos"][embryo_id] = {
                        "genotype": None,
                        "source": {
                            "experiment_id": exp_id,
                            "video_id": video_id,
                            "sam_annotation_source": str(self.sam_annotation_path)
                        },
                        "snips": {}
                    }
                
                # Add snips for this embryo
                for image_id, image_data in video_data["images"].items():
                    for emb_id, emb_data in image_data["embryos"].items():
                        if emb_id == embryo_id:
                            snip_id = emb_data["snip_id"]
                            metadata["embryos"][embryo_id]["snips"][snip_id] = {
                                "phenotype": {
                                    "value": "NONE",
                                    "author": "system",
                                    "timestamp": get_timestamp()
                                },
                                "flags": []
                            }
    
    return metadata
```

## Core Methods

### Save Method

```python
def save(self, backup=True, force=False):
    """
    Save metadata to file with atomic write.
    
    Steps:
    1. Check for unsaved changes
    2. Create backup if requested
    3. Update timestamps
    4. Write with atomic operation
    5. Clear change tracking
    """
    
    if not self._unsaved_changes and not force:
        if self.verbose:
            print("No changes to save")
        return
    
    # Create backup
    if backup and self.embryo_metadata_path.exists():
        backup_path = create_timestamped_backup(self.embryo_metadata_path)
        if self.verbose:
            print(f"Created backup: {backup_path}")
    
    # Update metadata
    self.data["file_info"]["last_updated"] = get_timestamp()
    
    # Atomic save
    temp_path = self.embryo_metadata_path.with_suffix('.tmp')
    
    try:
        # Write to temp file
        with open(temp_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        # Atomic rename
        temp_path.replace(self.embryo_metadata_path)
        
        # Clear tracking
        self._unsaved_changes = False
        self._change_log.clear()
        
        if self.verbose:
            print(f"Saved to {self.embryo_metadata_path}")
            
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise SaveError(f"Failed to save: {e}")
```

### Consistency Check Method

```python
def _run_consistency_checks(self):
    """
    Validate data consistency.
    
    Checks:
    1. All referenced IDs exist in source
    2. No orphaned data
    3. Valid permitted values
    4. Special rules (e.g., DEAD exclusivity)
    5. Required fields present
    """
    
    issues = []
    
    # Check 1: Validate embryo IDs against source
    source_embryo_ids = set(self.sam_annotations.get("embryo_ids", []))
    our_embryo_ids = set(self.data["embryos"].keys())
    
    missing_in_source = our_embryo_ids - source_embryo_ids
    if missing_in_source:
        issues.append(f"Embryo IDs not in source: {missing_in_source}")
    
    # Check 2: Validate snip IDs
    source_snip_ids = set(self.sam_annotations.get("snip_ids", []))
    our_snip_ids = set()
    
    for embryo_data in self.data["embryos"].values():
        our_snip_ids.update(embryo_data["snips"].keys())
    
    missing_snips = our_snip_ids - source_snip_ids
    if missing_snips:
        issues.append(f"Snip IDs not in source: {missing_snips}")
    
    # Check 3: Validate all values against permitted
    invalid_phenotypes = self._check_permitted_values("phenotypes")
    invalid_genotypes = self._check_permitted_values("genotypes")
    invalid_flags = self._check_permitted_values("flags")
    
    if invalid_phenotypes:
        issues.append(f"Invalid phenotypes: {invalid_phenotypes}")
    if invalid_genotypes:
        issues.append(f"Invalid genotypes: {invalid_genotypes}")
    if invalid_flags:
        issues.append(f"Invalid flags: {invalid_flags}")
    
    # Check 4: DEAD phenotype exclusivity
    for embryo_id, embryo_data in self.data["embryos"].items():
        dead_snips = []
        other_phenotypes = []
        
        for snip_id, snip_data in embryo_data["snips"].items():
            phenotype = snip_data["phenotype"]["value"]
            if phenotype == "DEAD":
                dead_snips.append(snip_id)
            elif phenotype != "NONE":
                other_phenotypes.append((snip_id, phenotype))
        
        # Check for conflicts
        if dead_snips and other_phenotypes:
            issues.append(
                f"Embryo {embryo_id} has DEAD phenotype but also: {other_phenotypes}"
            )
    
    # Check 5: Missing genotypes warning
    missing_genotypes = []
    for embryo_id, embryo_data in self.data["embryos"].items():
        if not embryo_data.get("genotype"):
            missing_genotypes.append(embryo_id)
    
    if missing_genotypes:
        if self.verbose:
            print(f"⚠️ Warning: {len(missing_genotypes)} embryos missing genotype")
            print(f"   Experiments affected: {self._get_affected_experiments(missing_genotypes)}")
    
    # Handle issues
    if issues:
        if self.verbose:
            print("❌ Consistency check failed:")
            for issue in issues:
                print(f"   - {issue}")
        raise ConsistencyError("Data consistency check failed")
    
    if self.verbose:
        print("✅ Consistency check passed")
```

### Configuration Inheritance

```python
def _inherit_configurations(self):
    """
    Inherit model configurations from SAM annotations.
    
    Steps:
    1. Extract detection model config
    2. Extract segmentation model config  
    3. Copy to our config section
    4. Generate gsam_annotation_id if needed
    """
    
    # Get configs from SAM annotations
    sam_config = self.sam_annotations.get("config", {})
    seed_info = self.sam_annotations.get("seed_annotations_info", {})
    sam2_info = self.sam_annotations.get("sam2_model_info", {})
    
    # Detection model (from seed annotations)
    if seed_info:
        self.data["config"]["detection_model"] = {
            "config": seed_info.get("model_config", "unknown"),
            "weights": seed_info.get("model_weights", "unknown")
        }
    
    # Segmentation model (SAM2)
    if sam2_info:
        self.data["config"]["segmentation_model"] = {
            "config": sam2_info.get("config_path", "unknown"),
            "weights": sam2_info.get("checkpoint_path", "unknown")
        }
    
    # Ensure gsam_annotation_id exists
    if "gsam_annotation_id" not in self.data["file_info"]:
        self.data["file_info"]["gsam_annotation_id"] = generate_gsam_id()
        self._unsaved_changes = True
```

## Utility Methods

### Get Methods

```python
def get_embryo(self, embryo_id):
    """Get embryo data with validation."""
    if embryo_id not in self.data["embryos"]:
        raise KeyError(f"Embryo {embryo_id} not found")
    return self.data["embryos"][embryo_id]

def get_snip(self, embryo_id, snip_id):
    """Get snip data with validation."""
    embryo = self.get_embryo(embryo_id)
    if snip_id not in embryo["snips"]:
        raise KeyError(f"Snip {snip_id} not found in embryo {embryo_id}")
    return embryo["snips"][snip_id]

def list_embryos(self, experiment_id=None, video_id=None):
    """List embryos with optional filtering."""
    embryos = []
    
    for embryo_id, embryo_data in self.data["embryos"].items():
        source = embryo_data.get("source", {})
        
        # Apply filters
        if experiment_id and source.get("experiment_id") != experiment_id:
            continue
        if video_id and source.get("video_id") != video_id:
            continue
            
        embryos.append(embryo_id)
    
    return sorted(embryos)
```

### Summary Methods

```python
def get_summary(self):
    """Generate summary statistics."""
    summary = {
        "total_embryos": len(self.data["embryos"]),
        "total_snips": sum(
            len(emb["snips"]) for emb in self.data["embryos"].values()
        ),
        "genotyped_embryos": sum(
            1 for emb in self.data["embryos"].values() 
            if emb.get("genotype")
        ),
        "phenotyped_snips": sum(
            1 for emb in self.data["embryos"].values()
            for snip in emb["snips"].values()
            if snip["phenotype"]["value"] != "NONE"
        ),
        "experiments": len(set(
            emb["source"]["experiment_id"] 
            for emb in self.data["embryos"].values()
        )),
        "has_unsaved_changes": self._unsaved_changes
    }
    
    # Add phenotype breakdown
    phenotype_counts = defaultdict(int)
    for emb in self.data["embryos"].values():
        for snip in emb["snips"].values():
            phenotype = snip["phenotype"]["value"]
            phenotype_counts[phenotype] += 1
    
    summary["phenotype_counts"] = dict(phenotype_counts)
    
    return summary

def print_summary(self):
    """Print formatted summary."""
    summary = self.get_summary()
    print("\n📊 EMBRYO METADATA SUMMARY")
    print("=" * 40)
    print(f"🧬 Total embryos: {summary['total_embryos']}")
    print(f"📸 Total snips: {summary['total_snips']}")
    print(f"🧪 Genotyped: {summary['genotyped_embryos']} ({summary['genotyped_embryos']/max(1, summary['total_embryos'])*100:.1f}%)")
    print(f"🔬 Phenotyped: {summary['phenotyped_snips']} snips")
    print(f"📁 Experiments: {summary['experiments']}")
    
    if summary['phenotype_counts']:
        print("\n📊 Phenotype Distribution:")
        for phenotype, count in sorted(summary['phenotype_counts'].items()):
            if phenotype != "NONE":
                print(f"   {phenotype}: {count}")
    
    status = "⚠️ unsaved changes" if summary['has_unsaved_changes'] else "✅ saved"
    print(f"\n💾 Status: {status}")
```

## Error Handling

```python
class EmbryoMetadataError(Exception):
    """Base exception for EmbryoMetadata."""
    pass

class ConsistencyError(EmbryoMetadataError):
    """Raised when data consistency checks fail."""
    pass

class ValidationError(EmbryoMetadataError):
    """Raised when validation fails."""
    pass

class SaveError(EmbryoMetadataError):
    """Raised when save operation fails."""
    pass
```

## Performance Optimizations

1. **Lazy Loading**
   - Load SAM annotations only when needed
   - Cache frequently accessed data
   - Use generators for large iterations

2. **Efficient Lookups**
   - Index structures for quick access
   - Reverse mappings for common queries
   - Memoization of expensive operations

3. **Memory Management**
   - Stream large files instead of loading entirely
   - Clear caches when not needed
   - Use slots for class attributes


## Next Steps

- Implement Module 2: Data Models and Validation
- Define specific validation rules
- Create helper functions for common operations