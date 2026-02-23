"""
Embryo Phenotype Manager
Handles all phenotype-related operations for EmbryoMetadata.
Designed as a mixin class to keep embryo_metadata.py manageable.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))


from typing import Dict, List, Optional
from embryo_metadata_models import Phenotype, Validator, ValidationError


class EmbryoPhenotypeManager:
    """
    Mixin class for phenotype management operations.
    
    This class should be inherited by EmbryoMetadata to provide
    phenotype-specific functionality while keeping the main class
    focused and manageable.
    
    Required attributes from parent class:
    - self.data: Dict containing embryo metadata
    - self.schema_manager: PermittedValuesManager instance
    - self.verbose: bool for logging
    - self.get_timestamp(): method for timestamps
    - self.log_operation(): method for operation logging
    - self.validate_id_format(): method for ID validation
    - self.get_embryo_id_from_snip(): method for embryo lookup
    - self.get_snip_data(): method for snip data retrieval
    """
    
    def add_phenotype(self, snip_id: str, phenotype: str, author: str,
                     notes: str = None, confidence: float = None,
                     force_dead: bool = False, verbose: bool = None) -> bool:
        """
        Add a phenotype to a snip.
        
        Args:
            snip_id: Valid snip ID (e.g., "20240411_A01_e01_0001")
            phenotype: Phenotype value (must be in permitted values)
            author: Author of the annotation
            notes: Optional notes
            confidence: Confidence score (0.0-1.0) for ML predictions
            force_dead: Allow adding DEAD even with existing phenotypes
            
        Returns:
            bool: True if added successfully
            
        Raises:
            ValueError: If validation fails
        """
        # Validate snip ID format
        if not self.validate_id_format(snip_id, "snip"):
            raise ValueError(f"Invalid snip ID format: {snip_id}")
        
        # Get embryo ID from snip
        embryo_id = self.get_embryo_id_from_snip(snip_id)
        if not embryo_id:
            raise ValueError(f"Cannot find embryo for snip: {snip_id}")
        
        # Validate phenotype against schema
        phenotypes = self.schema_manager.get_phenotypes()
        if not self.schema_manager.is_valid_phenotype(phenotype):
            available = list(phenotypes.keys())
            raise ValueError(f"Invalid phenotype '{phenotype}'. Available: {available}")
        
        # Get existing phenotypes for validation
        snip_data = self.get_snip_data(snip_id)
        existing_phenotypes = []
        if snip_data and "phenotype" in snip_data:
            existing_phenotypes.append(snip_data["phenotype"]["value"])
        
        # Validate using our validator
        try:
            Validator.validate_phenotype(phenotype, phenotypes, existing_phenotypes)
        except ValidationError as e:
            if not force_dead:
                raise ValueError(str(e))
        
        # Initialize embryo and snip data if needed
        if embryo_id not in self.data["embryos"]:
            self.data["embryos"][embryo_id] = {
                "genotypes": {},
                "treatments": {},
                "phenotypes": {},
                "flags": {},
                "metadata": {
                    "created": self.get_timestamp(),
                    "last_updated": self.get_timestamp()
                },
                "source": {"sam_annotation_source": str(self.sam_annotation_path)},
                "snips": {}
            }
        
        if snip_id not in self.data["embryos"][embryo_id]["snips"]:
            self.data["embryos"][embryo_id]["snips"][snip_id] = {"flags": []}
        
        # Create phenotype using our model
        phenotype_obj = Phenotype(
            value=phenotype,
            author=author,
            notes=notes,
            confidence=confidence
        )
        
        # Add to snip data
        self.data["embryos"][embryo_id]["snips"][snip_id]["phenotype"] = phenotype_obj.to_dict()
        
        # Log operation
        self.log_operation("add_phenotype", snip_id,
                         phenotype=phenotype, author=author, confidence=confidence)
        
        if self.verbose:
            print(f"✅ Added phenotype '{phenotype}' to {snip_id}")
        
        return True
    
    def edit_phenotype(self, embryo_id: str, phenotype_name: str,
                      severity: str = None, confidence: float = None,
                      notes: str = None) -> bool:
        """
        Edit an existing phenotype.
        
        Args:
            embryo_id: Valid embryo ID
            phenotype_name: Name of existing phenotype
            severity: New severity (optional)
            confidence: New confidence (optional)
            notes: New notes (optional)
            
        Returns:
            bool: True if edited successfully
            
        Raises:
            ValueError: If phenotype doesn't exist or validation fails
        """
        # Check if embryo exists
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        # Check if phenotype exists
        if phenotype_name not in self.data["embryos"][embryo_id]["phenotypes"]:
            available = list(self.data["embryos"][embryo_id]["phenotypes"].keys())
            raise ValueError(f"Phenotype '{phenotype_name}' not found for {embryo_id}. "
                           f"Available: {available}")
        
        # Get current phenotype
        current = self.data["embryos"][embryo_id]["phenotypes"][phenotype_name]
        
        # Update fields if provided
        if severity is not None:
            if not self.schema_manager.validate_value("severity_levels", severity):
                available = self.schema_manager.get_values("severity_levels")
                raise ValueError(f"Invalid severity '{severity}'. Available: {available}")
            current["severity"] = severity
        
        if confidence is not None:
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
            current["confidence"] = confidence
        
        if notes is not None:
            current["notes"] = notes
        
        # Update timestamp
        current["last_updated"] = self.get_timestamp()
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        # Log operation
        changes = {}
        if severity is not None:
            changes["severity"] = severity
        if confidence is not None:
            changes["confidence"] = confidence
        if notes is not None:
            changes["notes"] = notes
        
        self.log_operation("edit_phenotype", embryo_id, 
                         phenotype=phenotype_name, changes=changes)
        
        if self.verbose:
            print(f"✅ Updated phenotype '{phenotype_name}' for {embryo_id}")
        
        return True
    
    def remove_phenotype(self, embryo_id: str, phenotype_name: str) -> bool:
        """
        Remove a phenotype from an embryo.
        
        Args:
            embryo_id: Valid embryo ID
            phenotype_name: Name of phenotype to remove
            
        Returns:
            bool: True if removed successfully
            
        Raises:
            ValueError: If embryo or phenotype doesn't exist
        """
        # Check if embryo exists
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        # Check if phenotype exists
        if phenotype_name not in self.data["embryos"][embryo_id]["phenotypes"]:
            available = list(self.data["embryos"][embryo_id]["phenotypes"].keys())
            raise ValueError(f"Phenotype '{phenotype_name}' not found for {embryo_id}. "
                           f"Available: {available}")
        
        # Remove phenotype
        removed = self.data["embryos"][embryo_id]["phenotypes"].pop(phenotype_name)
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        # Log operation
        self.log_operation("remove_phenotype", embryo_id, 
                         phenotype=phenotype_name, removed_data=removed)
        
        if self.verbose:
            print(f"🗑️ Removed phenotype '{phenotype_name}' from {embryo_id}")
        
        return True
    
    def get_phenotypes(self, embryo_id: str) -> dict:
        """
        Get all phenotypes for an embryo.
        
        Args:
            embryo_id: Valid embryo ID
            
        Returns:
            dict: Phenotypes data
        """
        if embryo_id not in self.data["embryos"]:
            return {}
        
        return self.data["embryos"][embryo_id]["phenotypes"].copy()
    
    def list_phenotypes_by_name(self, phenotype_name: str) -> list:
        """
        Find all embryos with a specific phenotype.
        
        Args:
            phenotype_name: Name of phenotype to search for
            
        Returns:
            list: List of embryo IDs with this phenotype
        """
        embryos_with_phenotype = []
        
        for embryo_id, embryo_data in self.data["embryos"].items():
            if phenotype_name in embryo_data["phenotypes"]:
                embryos_with_phenotype.append(embryo_id)
        
        return embryos_with_phenotype
    
    def get_phenotype_statistics(self) -> Dict:
        """
        Get comprehensive phenotype statistics.
        
        Returns:
            Dict containing phenotype counts, completion rates, etc.
        """
        stats = {
            "total_snips": 0,
            "phenotyped_snips": 0,
            "phenotype_counts": {},
            "completion_rate": 0.0,
            "most_common": [],
            "by_author": {}
        }
        
        for embryo_data in self.data["embryos"].values():
            for snip_data in embryo_data.get("snips", {}).values():
                stats["total_snips"] += 1
                
                phenotype = snip_data.get("phenotype", {})
                if phenotype and phenotype.get("value") != "NONE":
                    stats["phenotyped_snips"] += 1
                    pheno_val = phenotype.get("value", "UNKNOWN")
                    
                    # Count phenotypes
                    stats["phenotype_counts"][pheno_val] = stats["phenotype_counts"].get(pheno_val, 0) + 1
                    
                    # Count by author
                    author = phenotype.get("author", "unknown")
                    if author not in stats["by_author"]:
                        stats["by_author"][author] = {}
                    stats["by_author"][author][pheno_val] = stats["by_author"][author].get(pheno_val, 0) + 1
        
        # Calculate completion rate
        if stats["total_snips"] > 0:
            stats["completion_rate"] = stats["phenotyped_snips"] / stats["total_snips"]
        
        # Get most common phenotypes
        if stats["phenotype_counts"]:
            stats["most_common"] = sorted(stats["phenotype_counts"].items(), 
                                        key=lambda x: x[1], reverse=True)[:10]
        
        return stats
    
    def get_phenotype(self, snip_id: str) -> Optional[Dict]:
        """
        Get phenotype data for a snip.
        
        Args:
            snip_id: Snip identifier
            
        Returns:
            Phenotype data or None if not found
        """
        snip_data = self.get_snip_data(snip_id)
        if snip_data:
            return snip_data.get("phenotype")
        return None
