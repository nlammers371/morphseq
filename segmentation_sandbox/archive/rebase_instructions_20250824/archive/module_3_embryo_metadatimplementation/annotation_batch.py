"""
AnnotationBatch - Lightweight batch container for embryo annotations.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
from data_managers.unified_embryo_manager import UnifiedEmbryoManager


class AnnotationBatch(UnifiedEmbryoManager):
    """Batch container for embryo annotations with validation and persistence."""
    
    def __init__(self, author: str, description: str = "", metadata_ref=None):
        self.author = author
        self.description = description
        self.metadata_ref = metadata_ref  # Optional reference to EmbryoMetadat
        self.created = datetime.now().isoformat()
        self.data = {}  # embryo_id -> annotation structure
        self.stats = {"phenotypes": 0, "genotypes": 0, "flags": 0, "treatments": 0}
    
    # ========== ADD OPERATIONS ==========
    
    def add(self, embryo_id: str, annotation_type: str, value: str, **kwargs):
        """Generic add function routing to specific methods."""
        if annotation_type == "phenotype":
            self.add_phenotype(embryo_id, value, **kwargs)
        elif annotation_type == "genotype":
            self.add_genotype(embryo_id, value, **kwargs)
        elif annotation_type == "flag":
            self.add_flag(embryo_id, value, **kwargs)
        elif annotation_type == "treatment":
            self.add_treatment(embryo_id, value, **kwargs)
        else:
            raise ValueError(f"Unknown annotation type: {annotation_type}")
    
    def add_phenotype(self, embryo_id: str, phenotype: str, frames: Union[str, List[str]] = "all",
                    notes: str = None, confidence: float = None, 
                    force_dead: bool = False, overwrite_dead: bool = False):
        """Add phenotype operation to batch."""
        # Validate inputs using inherited methods
        valid_phenotypes = self._get_valid_phenotypes()
        if phenotype not in valid_phenotypes:
            available = list(valid_phenotypes.keys())
            raise ValueError(f"Invalid phenotype '{phenotype}'. Available: {available}")
        
        if not self.validate_id_format(embryo_id, "embryo"):
            raise ValueError(f"Invalid embryo ID format: {embryo_id}")
        
        # Initialize embryo structure
        self._ensure_embryo_structure(embryo_id)
        
        # Store operation
        operation = {
            "value": phenotype,
            "author": self.author,
            "frames": frames,
            "notes": notes,
            "confidence": confidence,
            "force_dead": force_dead,
            "overwrite_dead": overwrite_dead,
            "timestamp": self.get_timestamp()
        }
        
        if "phenotype_operations" not in self.data[embryo_id]:
            self.data[embryo_id]["phenotype_operations"] = []
        
        self.data[embryo_id]["phenotype_operations"].append(operation)
        self.stats["phenotypes"] += 1

    def mark_dead(self, embryo_id: str, start_frame: int = None):
        """Mark embryo as dead from specified frame onward."""
        if start_frame is not None:
            frame_spec = f"{start_frame}:"
        else:
            frame_spec = "all"
        
        # Use force_dead=True and overwrite_dead=True for this operation
        self.add_phenotype(
            embryo_id, 
            "DEAD", 
            frames=frame_spec, 
            force_dead=True,
            overwrite_dead=True
        )
    
    def add_genotype(self, embryo_id: str, gene_name: str, allele: str = None,
                    zygosity: str = "heterozygous", notes: str = None):
        """Add genotype operation to batch."""
        # Validate inputs
        valid_genotypes = self._get_valid_genotypes()
        if gene_name not in valid_genotypes:
            available = list(valid_genotypes.keys())
            raise ValueError(f"Invalid gene '{gene_name}'. Available: {available}")
        
        valid_zygosity = self._get_valid_zygosity_types()
        if zygosity not in valid_zygosity:
            raise ValueError(f"Invalid zygosity '{zygosity}'. Available: {valid_zygosity}")
        
        if not self.validate_id_format(embryo_id, "embryo"):
            raise ValueError(f"Invalid embryo ID format: {embryo_id}")
        
        # Initialize and store
        self._ensure_embryo_structure(embryo_id)
        
        self.data[embryo_id]["genotype"] = {
            "value": gene_name,
            "allele": allele,
            "zygosity": zygosity,
            "author": self.author,
            "notes": notes,
            "timestamp": self.get_timestamp()
        }
        self.stats["genotypes"] += 1
    
    def add_flag(self, embryo_id: str, flag_type: str, level: str = "auto",
                priority: str = "medium", description: str = "", notes: str = ""):
        """Add flag operation to batch."""
        # Auto-detect level
        if level == "auto":
            level = "snip" if self.validate_id_format(embryo_id, "snip") else "embryo"
        
        # Validate inputs
        valid_flags = self._get_valid_flags_for_level(level)
        if flag_type not in valid_flags:
            raise ValueError(f"Invalid flag '{flag_type}' for {level} level. Available: {valid_flags}")
        
        valid_priorities = self._get_valid_priority_levels()
        if priority not in valid_priorities:
            raise ValueError(f"Invalid priority '{priority}'. Available: {valid_priorities}")
        
        # Initialize and store
        self._ensure_embryo_structure(embryo_id)
        
        flag_data = {
            "flag_type": flag_type,
            "level": level,
            "priority": priority,
            "description": description,
            "notes": notes,
            "author": self.author,
            "timestamp": self.get_timestamp()
        }
        
        if "flags" not in self.data[embryo_id]:
            self.data[embryo_id]["flags"] = []
        
        self.data[embryo_id]["flags"].append(flag_data)
        self.stats["flags"] += 1
    
    def add_treatment(self, embryo_id: str, treatment_name: str, dosage: str = None,
                     timing: str = None, notes: str = None):
        """Add treatment operation to batch."""
        # Validate inputs
        valid_treatments = self._get_valid_treatments()
        if treatment_name not in valid_treatments:
            available = list(valid_treatments.keys())
            raise ValueError(f"Invalid treatment '{treatment_name}'. Available: {available}")
        
        if not self.validate_id_format(embryo_id, "embryo"):
            raise ValueError(f"Invalid embryo ID format: {embryo_id}")
        
        # Initialize and store
        self._ensure_embryo_structure(embryo_id)
        
        treatment_id = f"{treatment_name}_{len(self.data[embryo_id].get('treatments', {})) + 1}"
        
        if "treatments" not in self.data[embryo_id]:
            self.data[embryo_id]["treatments"] = {}
        
        self.data[embryo_id]["treatments"][treatment_id] = {
            "value": treatment_name,
            "dosage": dosage,
            "timing": timing,
            "notes": notes,
            "author": self.author,
            "timestamp": self.get_timestamp()
        }
        self.stats["treatments"] += 1
    
    # ========== EDIT/REMOVE OPERATIONS ==========
    
    def remove_annotation(self, embryo_id: str, annotation_type: str, **filters):
        """Remove specific annotations from batch."""
        if embryo_id not in self.data:
            return False
        
        if annotation_type == "phenotype":
            self.data[embryo_id].pop("phenotype_operations", None)
        elif annotation_type == "genotype":
            self.data[embryo_id].pop("genotype", None)
        elif annotation_type == "flags":
            self.data[embryo_id].pop("flags", None)
        elif annotation_type == "treatments":
            self.data[embryo_id].pop("treatments", None)
        
        return True
    
    def edit_annotation(self, embryo_id: str, annotation_type: str, **updates):
        """Modify existing annotation in batch."""
        if embryo_id not in self.data:
            raise ValueError(f"Embryo {embryo_id} not found in batch")
        
        if annotation_type == "genotype" and "genotype" in self.data[embryo_id]:
            self.data[embryo_id]["genotype"].update(updates)
        # Add other edit logic as needed
        
        return True
    
    def clear_embryo(self, embryo_id: str, annotation_types: Union[str, List[str]] = "all"):
        """Remove all annotations for embryo."""
        if embryo_id not in self.data:
            return False
        
        if annotation_types == "all":
            self.data.pop(embryo_id)
        else:
            if isinstance(annotation_types, str):
                annotation_types = [annotation_types]
            
            for ann_type in annotation_types:
                self.data[embryo_id].pop(ann_type, None)
        
        return True
    
    # ========== ACCESS AND DISPLAY ==========
    
    def __getitem__(self, embryo_id: str):
        """Direct access to embryo data."""
        return self.data.get(embryo_id, {})
    
    def __str__(self) -> str:
        """Organized summary display."""
        lines = [
            f"AnnotationBatch (Author: {self.author})",
            f"Created: {self.created}",
            f"Description: {self.description}",
            f"Stats: {self.stats}",
            f"Embryos: {len(self.data)}",
            ""
        ]
        
        for embryo_id, data in self.data.items():
            lines.append(f"📋 {embryo_id}:")
            
            if "genotype" in data:
                g = data["genotype"]
                lines.append(f"  🧬 Genotype: {g['value']} ({g.get('zygosity', 'unknown')})")
            
            if "phenotype_operations" in data:
                lines.append(f"  🔬 Phenotypes: {len(data['phenotype_operations'])} operations")
                for op in data["phenotype_operations"]:
                    lines.append(f"    - {op['value']} (frames: {op['frames']})")

            if "treatments" in data:
                lines.append(f"  💊 Treatments: {len(data['treatments'])}")
                for t_id, t_data in data["treatments"].items():
                    lines.append(f"    - {t_data['value']}")
            
            if "flags" in data:
                lines.append(f"  🚩 Flags: {len(data['flags'])}")
                for flag in data["flags"]:
                    lines.append(f"    - {flag['flag_type']} ({flag['priority']})")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def preview(self) -> str:
        """Detailed summary by embryo."""
        return str(self)
    
    def get_stats(self) -> Dict:
        """Count summary."""
        return self.stats.copy()
    
    def get_embryo_list(self) -> List[str]:
        """All embryos in batch."""
        return list(self.data.keys())
    
    # ========== PERSISTENCE ==========
    
    def save_json(self, path: Union[str, Path]):
        """Save batch to JSON file."""
        batch_export = {
            "author": self.author,
            "description": self.description,
            "created": self.created,
            "stats": self.stats,
            "data": self.data
        }
        
        with open(path, 'w') as f:
            json.dump(batch_export, f, indent=2)
    
    @classmethod
    def load_json(cls, path: Union[str, Path]):
        """Load batch from JSON file."""
        with open(path, 'r') as f:
            batch_data = json.load(f)
        
        batch = cls(
            author=batch_data["author"],
            description=batch_data.get("description", "")
        )
        batch.created = batch_data["created"]
        batch.stats = batch_data["stats"]
        batch.data = batch_data["data"]
        
        return batch
    
    def to_dict(self) -> Dict:
        """Export to dictionary."""
        return {
            "author": self.author,
            "description": self.description,
            "created": self.created,
            "stats": self.stats,
            "data": self.data
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create from dictionary."""
        batch = cls(data["author"], data.get("description", ""))
        batch.created = data["created"]
        batch.stats = data["stats"]
        batch.data = data["data"]
        return batch
    
    # ========== UTILITY METHODS ==========
    
    def _ensure_embryo_structure(self, embryo_id: str):
        """Initialize embryo structure in batch."""
        if embryo_id not in self.data:
            self.data[embryo_id] = {}
   
    def query(self) -> EmbryoQuery:
    """Create new query builder."""
    return EmbryoQuery(self)


class EmbryoQuery:
   """Flexible query builder for embryo metadata."""
   
   def __init__(self, metadata: EmbryoMetadata):
       self.metadata = metadata
       self.filters = []
       self._result_cache = None
   
   def where(self, **kwargs):
       """Add filter conditions."""
       self.filters.append(kwargs)
       return self
   
   def phenotype(self, value: str, operator: str = "equals"):
       """Filter by phenotype."""
       self.filters.append({"phenotype": value, "op": operator})
       return self
   
   def genotype(self, value: str):
       """Filter by genotype."""
       self.filters.append({"genotype": value})
       return self
   
   def treatment(self, value: str):
       """Filter by treatment."""
       self.filters.append({"treatment": value})
       return self
   
   def has_flag(self, flag_type: str, level: str = "any"):
       """Filter by flag presence."""
       self.filters.append({"flag": flag_type, "level": level})
       return self
   
   def frame_range(self, start: int = None, end: int = None):
       """Filter by frame range."""
       self.filters.append({"frame_start": start, "frame_end": end})
       return self
   
   def execute(self) -> List[str]:
       """Execute query and return embryo IDs."""
       results = set(self.metadata.data["embryos"].keys())
       
       for filter_dict in self.filters:
           if "phenotype" in filter_dict:
               embryos_with_pheno = set()
               for snip in self.metadata.list_snips_by_phenotype(filter_dict["phenotype"]):
                   embryos_with_pheno.add(self.metadata.get_embryo_id_from_snip(snip))
               results &= embryos_with_pheno
           
           if "genotype" in filter_dict:
               embryos_with_geno = {e for e in results 
                                   if self.metadata.get_genotype(e) and 
                                   self.metadata.get_genotype(e)["value"] == filter_dict["genotype"]}
               results &= embryos_with_geno
           
           if "treatment" in filter_dict:
               embryos_with_treat = {e for e in results
                                    if filter_dict["treatment"] in 
                                    [t["value"] for t in self.metadata.get_treatments(e).values()]}
               results &= embryos_with_treat
       
       self._result_cache = list(results)
       return self._result_cache
   
   def count(self) -> int:
       """Count matching embryos."""
       if self._result_cache is None:
           self.execute()
       return len(self._result_cache)
   
   def first(self) -> Optional[str]:
       """Get first matching embryo."""
       if self._result_cache is None:
           self.execute()
       return self._result_cache[0] if self._result_cache else None

    # Add to EmbryoQuery class:

def get_snips(self) -> List[str]:
    """Get all snips for matching embryos."""
    if self._result_cache is None:
        self.execute()
    
    snips = []
    for embryo_id in self._result_cache:
        snips.extend(self.metadata.get_available_snips(embryo_id))
    return snips

def get_data(self) -> Dict[str, Dict]:
    """Get full data for matching embryos."""
    if self._result_cache is None:
        self.execute()
    
    return {e_id: self.metadata.data["embryos"][e_id] 
            for e_id in self._result_cache}

def to_batch(self, author: str) -> AnnotationBatch:
    """Convert query results to annotation batch."""
    batch = AnnotationBatch(author, f"Query batch: {len(self.filters)} filters")
    batch.metadata_ref = self.metadata
    
    # Pre-populate with embryo IDs for further annotation
    for embryo_id in self.execute():
        batch._ensure_embryo_structure(embryo_id)
    
    return batch

# Integration with EmbryoMetadata:
class EmbryoMetadata:
    def query(self) -> EmbryoQuery:
        """Create new query builder."""
        return EmbryoQuery(self)