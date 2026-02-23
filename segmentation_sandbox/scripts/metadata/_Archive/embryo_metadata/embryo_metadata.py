"""
EmbryoMetadata Class - Biological Annotation Management

This class manages embryo-specific biological annotations including:
- Phenotype and genotype data
- Quality control flags  
- Treatment annotations
- Integration with segmentation data

TODO: Implement EmbryoMetadata class with autosave functionality
similar to ExperimentMetadata.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# Import shared utilities
from utils.base_file_handler import BaseFileHandler
from utils.parsing_utils import parse_entity_id
from utils.entity_id_tracker import EntityIDTracker


class EmbryoMetadata(BaseFileHandler):
    """
    Main class for managing embryo metadata including phenotypes, genotypes, and flags.
    
    This class provides:
    - Hierarchical biological data storage (experiment → video → image → embryo)
    - Validation against permitted values schema
    - Change tracking and atomic saves with autosave functionality
    - Integration with GroundedSamAnnotation data
    - Management of treatment annotations
    """
    
    def __init__(self, 
                 sam_annotation_path: Union[str, Path], 
                 embryo_metadata_path: Optional[Union[str, Path]] = None,
                 gen_if_no_file: bool = False, 
                 auto_save_interval: int = 10,
                 verbose: bool = True):
        """
        Initialize EmbryoMetadata instance.
        
        Args:
            sam_annotation_path: Path to GroundedSam annotation file
            embryo_metadata_path: Path to embryo metadata file (auto-generated if None)
            gen_if_no_file: Create new file if metadata doesn't exist
            auto_save_interval: Number of operations before auto-save (0 to disable)
            verbose: Enable verbose output
        """
        # TODO: Implement EmbryoMetadata initialization
        # This should follow the same autosave pattern as ExperimentMetadata
        
        if embryo_metadata_path is None:
            sam_path = Path(sam_annotation_path)
            embryo_metadata_path = sam_path.with_name(
                sam_path.stem + "_embryo_metadata.json"
            )
        
        super().__init__(embryo_metadata_path, auto_save_interval=auto_save_interval)
        
        self.sam_annotation_path = Path(sam_annotation_path)
        self.verbose = verbose
        
        print(f"🚧 EmbryoMetadata class needs implementation")
        print(f"   SAM annotations: {self.sam_annotation_path}")
        print(f"   Embryo metadata: {self.filepath}")
        print(f"   Auto-save interval: {auto_save_interval}")
        
    def add_phenotype(self, embryo_id: str, phenotype_data: Dict) -> None:
        """Add phenotype annotation for an embryo."""
        # Minimal, fast implementation with lazy init and autosave
        # Lazy initialize in-memory structure
        if not hasattr(self, "metadata") or not isinstance(getattr(self, "metadata", None), dict):
            # Try to load existing file, otherwise create minimal structure
            try:
                self.metadata = self.load_json()
            except Exception:
                self.metadata = {
                    "file_info": {
                        "creation_time": datetime.now().isoformat(),
                        "version": "1.0",
                        "created_by": "EmbryoMetadata"
                    },
                    "embryos": {},
                    "entity_tracking": {"embryos": []}
                }

        # Ensure counters exist for autosave
        if not hasattr(self, "_ops_since_save"):
            self._ops_since_save = 0
        if not hasattr(self, "_auto_save_interval"):
            # Best-effort default that mirrors the __init__ default
            self._auto_save_interval = 10

        # Insert phenotype entry
        embryos = self.metadata.setdefault("embryos", {})
        embryo_entry = embryos.setdefault(embryo_id, {
            "embryo_id": embryo_id,
            "phenotypes": [],
            "genotypes": [],
            "flags": [],
            "created_time": datetime.now().isoformat(),
        })

        # Normalize phenotype payload and append with timestamp
        payload = dict(phenotype_data) if isinstance(phenotype_data, dict) else {"value": phenotype_data}
        payload["timestamp"] = datetime.now().isoformat()
        embryo_entry.setdefault("phenotypes", []).append(payload)

        # Track entity id
        tracking = self.metadata.setdefault("entity_tracking", {}).setdefault("embryos", [])
        if embryo_id not in tracking:
            tracking.append(embryo_id)
            tracking.sort()

        # Update file_info last_updated
        self.metadata.setdefault("file_info", {})["last_updated"] = datetime.now().isoformat()

        # Operation log (best-effort)
        try:
            self.log_operation("add_phenotype", entity_id=embryo_id)
        except Exception:
            # BaseFileHandler may not be fully initialized; ignore silently
            pass

        # Autosave handling
        self._ops_since_save += 1
        interval = getattr(self, "_auto_save_interval", 0) or 0
        if interval > 0 and self._ops_since_save >= interval:
            try:
                self.save_json(self.metadata)
                self._ops_since_save = 0
            except Exception:
                # Do not raise to keep method fast/non-blocking
                pass
        
    def add_genotype(self, embryo_id: str, genotype_data: Dict) -> None:
        """Add genotype annotation for an embryo."""  
        # TODO: Implement genotype addition with autosave
        pass
        
    def add_flag(self, entity_id: str, flag_type: str, details: str, author: str) -> None:
        """Add quality control flag."""
        # TODO: Implement flag addition with autosave
        pass
