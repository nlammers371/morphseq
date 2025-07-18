# Module 7: Integration Layer

## Overview
This module handles integration with external data sources, particularly GroundedSamAnnotation, and manages configuration inheritance and data synchronization.

## GroundedSamAnnotation Integration

### SAM Annotation Loader

```python
class SamAnnotationIntegration:
    """Integration with GroundedSamAnnotation data."""
    
    @staticmethod
    def load_sam_annotations(sam_path: Path) -> Dict:
        """
        Load and validate SAM annotation file.
        
        Args:
            sam_path: Path to grounded_sam_ft_annotations.json
        
        Returns:
            Loaded SAM annotations
        
        Raises:
            ValueError: If file invalid or missing required fields
        """
        if not sam_path.exists():
            raise FileNotFoundError(f"SAM annotation file not found: {sam_path}")
        
        try:
            with open(sam_path, 'r') as f:
                sam_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in SAM annotation file: {e}")
        
        # Validate required fields
        required_fields = ["experiments", "embryo_ids", "snip_ids"]
        missing = [field for field in required_fields if field not in sam_data]
        
        if missing:
            raise ValueError(f"SAM annotation missing required fields: {missing}")
        
        # Validate structure
        if not isinstance(sam_data.get("experiments"), dict):
            raise ValueError("SAM annotation 'experiments' must be a dictionary")
        
        return sam_data
    
    @staticmethod
    def extract_embryo_structure(sam_data: Dict) -> Dict:
        """
        Extract embryo/snip structure from SAM annotations.
        
        Returns:
            Dict mapping embryo_id to metadata including snips
        """
        embryo_structure = {}
        
        for exp_id, exp_data in sam_data.get("experiments", {}).items():
            for video_id, video_data in exp_data.get("videos", {}).items():
                embryo_ids = video_data.get("embryo_ids", [])
                
                # Process each image to find embryo/snip mappings
                for image_id, image_data in video_data.get("images", {}).items():
                    for embryo_id, embryo_info in image_data.get("embryos", {}).items():
                        if embryo_id not in embryo_structure:
                            embryo_structure[embryo_id] = {
                                "experiment_id": exp_id,
                                "video_id": video_id,
                                "snips": {}
                            }
                        
                        # Add snip
                        snip_id = embryo_info.get("snip_id")
                        if snip_id:
                            embryo_structure[embryo_id]["snips"][snip_id] = {
                                "image_id": image_id,
                                "frame_index": image_data.get("frame_index", -1),
                                "is_seed_frame": image_data.get("is_seed_frame", False),
                                "bbox": embryo_info.get("bbox", []),
                                "area": embryo_info.get("area", 0),
                                "mask_confidence": embryo_info.get("mask_confidence", 0)
                            }
        
        return embryo_structure
    
    @staticmethod
    def sync_with_sam_updates(metadata: 'EmbryoMetadata', 
                             new_sam_path: Path,
                             preserve_annotations: bool = True) -> Dict:
        """
        Synchronize with updated SAM annotations.
        
        Args:
            metadata: EmbryoMetadata instance
            new_sam_path: Path to new SAM annotation file
            preserve_annotations: Keep existing phenotypes/genotypes
        
        Returns:
            Sync results
        """
        # Load new SAM data
        new_sam_data = SamAnnotationIntegration.load_sam_annotations(new_sam_path)
        new_structure = SamAnnotationIntegration.extract_embryo_structure(new_sam_data)
        
        results = {
            "added_embryos": [],
            "removed_embryos": [],
            "added_snips": [],
            "removed_snips": [],
            "preserved_annotations": []
        }
        
        # Find new embryos
        existing_embryos = set(metadata.data["embryos"].keys())
        new_embryos = set(new_structure.keys())
        
        # Added embryos
        for embryo_id in new_embryos - existing_embryos:
            embryo_info = new_structure[embryo_id]
            metadata.data["embryos"][embryo_id] = {
                "genotype": None,
                "source": {
                    "experiment_id": embryo_info["experiment_id"],
                    "video_id": embryo_info["video_id"],
                    "sam_annotation_source": str(new_sam_path)
                },
                "snips": {}
            }
            
            # Add snips
            for snip_id, snip_info in embryo_info["snips"].items():
                metadata.data["embryos"][embryo_id]["snips"][snip_id] = {
                    "phenotype": {
                        "value": "NONE",
                        "author": "system",
                        "timestamp": datetime.now().isoformat()
                    },
                    "flags": []
                }
                results["added_snips"].append(snip_id)
            
            results["added_embryos"].append(embryo_id)
        
        # Removed embryos
        for embryo_id in existing_embryos - new_embryos:
            if preserve_annotations:
                # Archive instead of delete
                archived = metadata.data["embryos"][embryo_id].copy()
                archived["archived_date"] = datetime.now().isoformat()
                archived["archive_reason"] = "Not in updated SAM annotations"
                
                if "archived_embryos" not in metadata.data:
                    metadata.data["archived_embryos"] = {}
                
                metadata.data["archived_embryos"][embryo_id] = archived
                results["preserved_annotations"].append(embryo_id)
            
            del metadata.data["embryos"][embryo_id]
            results["removed_embryos"].append(embryo_id)
        
        # Check snip changes for existing embryos
        for embryo_id in existing_embryos & new_embryos:
            existing_snips = set(metadata.data["embryos"][embryo_id]["snips"].keys())
            new_snips = set(new_structure[embryo_id]["snips"].keys())
            
            # Added snips
            for snip_id in new_snips - existing_snips:
                metadata.data["embryos"][embryo_id]["snips"][snip_id] = {
                    "phenotype": {
                        "value": "NONE",
                        "author": "system",
                        "timestamp": datetime.now().isoformat()
                    },
                    "flags": []
                }
                results["added_snips"].append(snip_id)
            
            # Removed snips
            for snip_id in existing_snips - new_snips:
                if preserve_annotations:
                    # Keep annotation data in archive
                    snip_data = metadata.data["embryos"][embryo_id]["snips"][snip_id]
                    if snip_data["phenotype"]["value"] != "NONE" or snip_data.get("flags"):
                        results["preserved_annotations"].append(f"{embryo_id}/{snip_id}")
                
                del metadata.data["embryos"][embryo_id]["snips"][snip_id]
                results["removed_snips"].append(snip_id)
        
        # Update source reference
        metadata.data["file_info"]["source_sam_annotation"] = str(new_sam_path)
        metadata._unsaved_changes = True
        
        if metadata.verbose:
            print(f"🔄 Sync complete:")
            print(f"   Added: {len(results['added_embryos'])} embryos, {len(results['added_snips'])} snips")
            print(f"   Removed: {len(results['removed_embryos'])} embryos, {len(results['removed_snips'])} snips")
            if results['preserved_annotations']:
                print(f"   Preserved: {len(results['preserved_annotations'])} annotations")
        
        return results
```

### Configuration Inheritance

```python
class ConfigurationManager:
    """Manage configuration inheritance from source models."""
    
    @staticmethod
    def inherit_model_configs(metadata: 'EmbryoMetadata', 
                            sam_data: Dict) -> None:
        """
        Inherit model configurations from SAM annotations.
        
        Args:
            metadata: EmbryoMetadata instance
            sam_data: Loaded SAM annotation data
        """
        config = metadata.data.get("config", {})
        
        # Detection model config (from GroundedDINO)
        seed_info = sam_data.get("seed_annotations_info", {})
        if seed_info:
            config["detection_model"] = {
                "config": seed_info.get("model_config", "unknown"),
                "weights": seed_info.get("model_weights", "unknown"),
                "architecture": seed_info.get("model_architecture", "GroundedDINO")
            }
        
        # Segmentation model config (from SAM2)
        sam2_info = sam_data.get("sam2_model_info", {})
        if sam2_info:
            config["segmentation_model"] = {
                "config": Path(sam2_info.get("config_path", "unknown")).name,
                "weights": Path(sam2_info.get("checkpoint_path", "unknown")).name,
                "architecture": sam2_info.get("model_architecture", "SAM2")
            }
        
        # Processing parameters
        config["processing_params"] = {
            "target_prompt": sam_data.get("target_prompt", "individual embryo"),
            "segmentation_format": sam_data.get("segmentation_format", "rle"),
            "sam_creation_time": sam_data.get("creation_time", "unknown"),
            "sam_last_updated": sam_data.get("last_updated", "unknown")
        }
        
        metadata.data["config"] = config
    
    @staticmethod
    def validate_config_compatibility(metadata: 'EmbryoMetadata',
                                    new_sam_data: Dict) -> List[str]:
        """
        Validate configuration compatibility with new SAM data.
        
        Returns:
            List of compatibility warnings
        """
        warnings = []
        
        # Check detection model changes
        old_detection = metadata.data.get("config", {}).get("detection_model", {})
        new_detection = new_sam_data.get("seed_annotations_info", {})
        
        if old_detection.get("weights") != new_detection.get("model_weights"):
            warnings.append(
                f"Detection model changed: "
                f"{old_detection.get('weights')} → {new_detection.get('model_weights')}"
            )
        
        # Check segmentation model changes
        old_seg = metadata.data.get("config", {}).get("segmentation_model", {})
        new_seg = new_sam_data.get("sam2_model_info", {})
        
        old_seg_weights = Path(old_seg.get("weights", "")).name
        new_seg_weights = Path(new_seg.get("checkpoint_path", "")).name
        
        if old_seg_weights != new_seg_weights:
            warnings.append(
                f"Segmentation model changed: "
                f"{old_seg_weights} → {new_seg_weights}"
            )
        
        # Check prompt changes
        old_prompt = metadata.data.get("config", {}).get("processing_params", {}).get("target_prompt")
        new_prompt = new_sam_data.get("target_prompt")
        
        if old_prompt and new_prompt and old_prompt != new_prompt:
            warnings.append(
                f"Target prompt changed: '{old_prompt}' → '{new_prompt}'"
            )
        
        return warnings
```

### GSAM Annotation ID Management

```python
class GsamIdManager:
    """Manage GSAM annotation IDs for tracking."""
    
    @staticmethod
    def generate_gsam_id() -> int:
        """Generate a unique 4-digit GSAM annotation ID."""
        import random
        return random.randint(1000, 9999)
    
    @staticmethod
    def add_gsam_id_to_sam_annotation(sam_path: Path, 
                                     gsam_id: int = None) -> int:
        """
        Add GSAM annotation ID to SAM annotation file.
        
        This modifies the GroundedSamAnnotation file to include
        a tracking ID that links it to the EmbryoMetadata.
        
        Args:
            sam_path: Path to SAM annotation file
            gsam_id: Specific ID to use (or generate new)
        
        Returns:
            The GSAM ID that was added
        """
        # Load SAM annotation
        with open(sam_path, 'r') as f:
            sam_data = json.load(f)
        
        # Generate ID if not provided
        if gsam_id is None:
            gsam_id = GsamIdManager.generate_gsam_id()
        
        # Add to file_info section
        if "file_info" not in sam_data:
            sam_data["file_info"] = {}
        
        sam_data["file_info"]["gsam_annotation_id"] = gsam_id
        sam_data["file_info"]["gsam_id_added"] = datetime.now().isoformat()
        
        # Save back
        with open(sam_path, 'w') as f:
            json.dump(sam_data, f, indent=2)
        
        return gsam_id
    
    @staticmethod
    def link_embryo_metadata_to_sam(metadata: 'EmbryoMetadata',
                                   sam_path: Path) -> None:
        """
        Create bidirectional link between metadata and SAM annotation.
        
        Args:
            metadata: EmbryoMetadata instance
            sam_path: Path to SAM annotation file
        """
        # Check if SAM file has GSAM ID
        with open(sam_path, 'r') as f:
            sam_data = json.load(f)
        
        gsam_id = sam_data.get("file_info", {}).get("gsam_annotation_id")
        
        if not gsam_id:
            # Add GSAM ID to SAM file
            gsam_id = GsamIdManager.add_gsam_id_to_sam_annotation(sam_path)
            print(f"✅ Added GSAM ID {gsam_id} to SAM annotation file")
        
        # Store in metadata
        metadata.data["file_info"]["gsam_annotation_id"] = gsam_id
        metadata.data["file_info"]["linked_sam_annotation"] = str(sam_path)
        metadata._unsaved_changes = True
        
        print(f"🔗 Linked EmbryoMetadata to SAM annotation with ID {gsam_id}")
```

## External Data Integration

### ML Model Integration

```python
class MLModelIntegration:
    """Integration with ML models for automated annotation."""
    
    @staticmethod
    def prepare_batch_for_ml(metadata: 'EmbryoMetadata',
                           snip_ids: List[str],
                           include_features: List[str] = None) -> Dict:
        """
        Prepare snip data for ML model input.
        
        Args:
            metadata: EmbryoMetadata instance
            snip_ids: List of snip IDs to prepare
            include_features: Features to include (default: all)
        
        Returns:
            Dict with prepared data for ML
        """
        if include_features is None:
            include_features = ["bbox", "area", "mask_confidence", 
                              "frame_index", "is_seed_frame"]
        
        batch_data = {
            "snip_ids": [],
            "features": [],
            "metadata": []
        }
        
        for snip_id in snip_ids:
            # Find embryo and snip
            embryo_id = metadata._get_embryo_id_from_snip(snip_id)
            if not embryo_id:
                continue
            
            # Get SAM features
            sam_features = metadata._get_sam_features_for_snip(snip_id)
            if not sam_features:
                continue
            
            # Extract requested features
            feature_vector = []
            for feature in include_features:
                value = sam_features.get(feature, 0)
                if isinstance(value, list):
                    feature_vector.extend(value)
                else:
                    feature_vector.append(value)
            
            batch_data["snip_ids"].append(snip_id)
            batch_data["features"].append(feature_vector)
            batch_data["metadata"].append({
                "embryo_id": embryo_id,
                "experiment_id": metadata.data["embryos"][embryo_id]["source"]["experiment_id"],
                "video_id": metadata.data["embryos"][embryo_id]["source"]["video_id"]
            })
        
        return batch_data
    
    @staticmethod
    def apply_ml_predictions(metadata: 'EmbryoMetadata',
                           predictions: Dict[str, Dict],
                           model_name: str,
                           confidence_threshold: float = 0.5) -> Dict:
        """
        Apply ML model predictions to metadata.
        
        Args:
            metadata: EmbryoMetadata instance
            predictions: Dict mapping snip_id to prediction info
                {
                    "snip_id": {
                        "phenotype": "EDEMA",
                        "confidence": 0.85,
                        "probabilities": {"NONE": 0.1, "EDEMA": 0.85, ...}
                    }
                }
            model_name: Name of the ML model
            confidence_threshold: Minimum confidence to apply
        
        Returns:
            Application results
        """
        results = {
            "applied": 0,
            "skipped_low_confidence": 0,
            "skipped_existing": 0,
            "failed": []
        }
        
        for snip_id, prediction in predictions.items():
            try:
                confidence = prediction.get("confidence", 0)
                
                # Check confidence threshold
                if confidence < confidence_threshold:
                    results["skipped_low_confidence"] += 1
                    continue
                
                # Check if already has phenotype
                embryo_id = metadata._get_embryo_id_from_snip(snip_id)
                if not embryo_id:
                    results["failed"].append((snip_id, "Snip not found"))
                    continue
                
                current_phenotype = metadata.data["embryos"][embryo_id]["snips"][snip_id]["phenotype"]["value"]
                if current_phenotype != "NONE":
                    results["skipped_existing"] += 1
                    continue
                
                # Apply prediction
                success = metadata.add_phenotype(
                    snip_id,
                    prediction["phenotype"],
                    author=f"ml_{model_name}",
                    notes=f"ML prediction with confidence {confidence:.3f}",
                    confidence=confidence
                )
                
                if success:
                    results["applied"] += 1
                    
            except Exception as e:
                results["failed"].append((snip_id, str(e)))
        
        if metadata.verbose:
            print(f"🤖 ML predictions applied:")
            print(f"   Applied: {results['applied']}")
            print(f"   Low confidence: {results['skipped_low_confidence']}")
            print(f"   Already annotated: {results['skipped_existing']}")
            print(f"   Failed: {len(results['failed'])}")
        
        return results
```

### Database Integration

```python
class DatabaseIntegration:
    """Integration with database systems for persistence."""
    
    @staticmethod
    def export_to_database(metadata: 'EmbryoMetadata',
                         connection_string: str,
                         schema_name: str = "morphseq") -> None:
        """
        Export metadata to relational database.
        
        Args:
            metadata: EmbryoMetadata instance
            connection_string: Database connection string
            schema_name: Database schema name
        """
        import sqlalchemy as sa
        from sqlalchemy import create_engine, MetaData, Table, Column
        
        engine = create_engine(connection_string)
        db_metadata = MetaData(schema=schema_name)
        
        # Define tables
        embryos_table = Table('embryos', db_metadata,
            Column('embryo_id', sa.String(50), primary_key=True),
            Column('experiment_id', sa.String(20)),
            Column('video_id', sa.String(30)),
            Column('genotype', sa.String(100)),
            Column('genotype_confirmed', sa.Boolean),
            Column('genotype_method', sa.String(50)),
            Column('created_at', sa.DateTime),
            Column('updated_at', sa.DateTime)
        )
        
        snips_table = Table('snips', db_metadata,
            Column('snip_id', sa.String(60), primary_key=True),
            Column('embryo_id', sa.String(50), sa.ForeignKey('embryos.embryo_id')),
            Column('frame_number', sa.Integer),
            Column('phenotype', sa.String(50)),
            Column('phenotype_confidence', sa.Float),
            Column('phenotype_author', sa.String(50)),
            Column('created_at', sa.DateTime),
            Column('updated_at', sa.DateTime)
        )
        
        flags_table = Table('flags', db_metadata,
            Column('flag_id', sa.Integer, primary_key=True, autoincrement=True),
            Column('entity_id', sa.String(60)),
            Column('entity_type', sa.String(20)),
            Column('flag_value', sa.String(50)),
            Column('severity', sa.String(20)),
            Column('author', sa.String(50)),
            Column('created_at', sa.DateTime)
        )
        
        # Create tables
        db_metadata.create_all(engine)
        
        # Prepare data for insertion
        embryo_records = []
        snip_records = []
        flag_records = []
        
        for embryo_id, embryo_data in metadata.data["embryos"].items():
            # Embryo record
            genotype_info = embryo_data.get("genotype", {})
            embryo_records.append({
                'embryo_id': embryo_id,
                'experiment_id': embryo_data["source"]["experiment_id"],
                'video_id': embryo_data["source"]["video_id"],
                'genotype': genotype_info.get("value") if genotype_info else None,
                'genotype_confirmed': genotype_info.get("confirmed", False) if genotype_info else False,
                'genotype_method': genotype_info.get("method") if genotype_info else None,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            })
            
            # Snip records
            for snip_id, snip_data in embryo_data["snips"].items():
                phenotype_info = snip_data["phenotype"]
                snip_records.append({
                    'snip_id': snip_id,
                    'embryo_id': embryo_id,
                    'frame_number': metadata._extract_frame_number(snip_id),
                    'phenotype': phenotype_info["value"],
                    'phenotype_confidence': phenotype_info.get("confidence"),
                    'phenotype_author': phenotype_info["author"],
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                })
                
                # Snip flags
                for flag in snip_data.get("flags", []):
                    flag_records.append({
                        'entity_id': snip_id,
                        'entity_type': 'snip',
                        'flag_value': flag["value"],
                        'severity': flag.get("severity", "warning"),
                        'author': flag["author"],
                        'created_at': datetime.now()
                    })
        
        # Bulk insert
        with engine.connect() as conn:
            if embryo_records:
                conn.execute(embryos_table.insert(), embryo_records)
            if snip_records:
                conn.execute(snips_table.insert(), snip_records)
            if flag_records:
                conn.execute(flags_table.insert(), flag_records)
            
            conn.commit()
        
        if metadata.verbose:
            print(f"📊 Exported to database:")
            print(f"   Embryos: {len(embryo_records)}")
            print(f"   Snips: {len(snip_records)}")
            print(f"   Flags: {len(flag_records)}")
```

## Helper Methods

```python
def _get_sam_features_for_snip(metadata: 'EmbryoMetadata', 
                              snip_id: str) -> Optional[Dict]:
    """
    Get SAM annotation features for a snip.
    
    This requires loading the linked SAM annotation file.
    """
    # Get linked SAM file
    sam_path = metadata.data["file_info"].get("source_sam_annotation")
    if not sam_path or not Path(sam_path).exists():
        return None
    
    # Load SAM data
    sam_data = SamAnnotationIntegration.load_sam_annotations(Path(sam_path))
    
    # Find the snip
    for exp_data in sam_data.get("experiments", {}).values():
        for video_data in exp_data.get("videos", {}).values():
            for image_data in video_data.get("images", {}).values():
                for embryo_id, embryo_info in image_data.get("embryos", {}).items():
                    if embryo_info.get("snip_id") == snip_id:
                        return {
                            "bbox": embryo_info.get("bbox", []),
                            "area": embryo_info.get("area", 0),
                            "mask_confidence": embryo_info.get("mask_confidence", 0),
                            "frame_index": image_data.get("frame_index", -1),
                            "is_seed_frame": image_data.get("is_seed_frame", False)
                        }
    
    return None

def _count_all_entities(metadata: 'EmbryoMetadata') -> int:
    """Count all entities across all levels."""
    count = 0
    
    # Experiments
    count += len(set(
        emb["source"]["experiment_id"] 
        for emb in metadata.data["embryos"].values()
    ))
    
    # Videos
    count += len(set(
        emb["source"]["video_id"] 
        for emb in metadata.data["embryos"].values()
    ))
    
    # Images (need to load SAM data)
    # This is an estimate based on snips
    count += sum(len(emb["snips"]) for emb in metadata.data["embryos"].values())
    
    # Embryos
    count += len(metadata.data["embryos"])
    
    # Snips
    count += sum(len(emb["snips"]) for emb in metadata.data["embryos"].values())
    
    return count

def _get_all_entities_at_level(metadata: 'EmbryoMetadata', 
                              level: str) -> List[str]:
    """Get all entity IDs at a specific level."""
    entities = set()
    
    if level == "experiment":
        for embryo_data in metadata.data["embryos"].values():
            entities.add(embryo_data["source"]["experiment_id"])
    
    elif level == "video":
        for embryo_data in metadata.data["embryos"].values():
            entities.add(embryo_data["source"]["video_id"])
    
    elif level == "embryo":
        entities = set(metadata.data["embryos"].keys())
    
    elif level == "snip":
        for embryo_data in metadata.data["embryos"].values():
            entities.update(embryo_data["snips"].keys())
    
    elif level == "image":
        # This requires parsing from snip IDs or loading SAM data
        for embryo_data in metadata.data["embryos"].values():
            for snip_id in embryo_data["snips"].keys():
                # Extract image_id from snip_id
                parts = snip_id.split('_')
                if len(parts) >= 3:
                    image_id = '_'.join(parts[:3])  # YYYYMMDD_WELL_FRAME
                    entities.add(image_id)
    
    return list(entities)
```