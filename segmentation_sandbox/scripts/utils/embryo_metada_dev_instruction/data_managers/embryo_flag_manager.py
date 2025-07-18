"""
Embryo Flag Manager
Handles all flag-related operations for EmbryoMetadata.
Designed as a mixin class to keep embryo_metadata.py manageable.
"""

from typing import Dict, List, Optional
from embryo_metadata_models import Flag


class EmbryoFlagManager:
    """
    Mixin class for flag management operations.
    
    Required attributes from parent class:
    - self.data: Dict containing embryo metadata
    - self.schema_manager: PermittedValuesManager instance
    - self.verbose: bool for logging
    - self.get_timestamp(): method for timestamps
    - self.log_operation(): method for operation logging
    - self.validate_id_format(): method for ID validation
    """
    
    def add_flag(self, embryo_id: str, flag_type: str, description: str = "",
                priority: str = "medium", confidence: float = 1.0,
                notes: str = "", overwrite: bool = False) -> bool:
        """Add a flag to an embryo."""
        # Validate embryo ID
        if not self.validate_id_format(embryo_id, "embryo"):
            raise ValueError(f"Invalid embryo ID format: {embryo_id}")
        
        # Validate flag type
        if not self.schema_manager.validate_value("flag_types", flag_type):
            available = self.schema_manager.get_values("flag_types")
            raise ValueError(f"Invalid flag type '{flag_type}'. Available: {available}")
        
        # Validate priority
        if not self.schema_manager.validate_value("priority_levels", priority):
            available = self.schema_manager.get_values("priority_levels")
            raise ValueError(f"Invalid priority '{priority}'. Available: {available}")
        
        # Validate confidence
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
        
        # Initialize embryo data if needed
        if embryo_id not in self.data["embryos"]:
            self.data["embryos"][embryo_id] = {
                "phenotypes": {}, "genotypes": {}, "flags": {}, "treatments": {},
                "metadata": {"created": self.get_timestamp(), "last_updated": self.get_timestamp()}
            }
        
        # Check for existing flag
        if flag_type in self.data["embryos"][embryo_id]["flags"] and not overwrite:
            existing = self.data["embryos"][embryo_id]["flags"][flag_type]
            raise ValueError(f"Flag '{flag_type}' already exists for {embryo_id}. Use overwrite=True to replace.")
        
        # Create flag using our model
        flag = Flag(
            value=flag_type,
            author=self.config.get("default_author", "unknown"),
            flag_type=flag_type,
            priority=priority,
            confidence=confidence,
            notes=f"{description}" + (f" | {notes}" if notes else "")
        )
        
        # Add to data
        self.data["embryos"][embryo_id]["flags"][flag_type] = flag.to_dict()
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        # Log operation
        self.log_operation("add_flag", embryo_id, flag_type=flag_type, priority=priority, confidence=confidence)
        
        if self.verbose:
            print(f"🚩 Added flag '{flag_type}' to {embryo_id}")
        
        return True
    
    def edit_flag(self, embryo_id: str, flag_type: str, description: str = None, 
                 priority: str = None, confidence: float = None, notes: str = None) -> bool:
        """Edit an existing flag."""
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        if flag_type not in self.data["embryos"][embryo_id]["flags"]:
            available = list(self.data["embryos"][embryo_id]["flags"].keys())
            raise ValueError(f"Flag '{flag_type}' not found for {embryo_id}. Available: {available}")
        
        current = self.data["embryos"][embryo_id]["flags"][flag_type]
        
        if description is not None:
            current["description"] = description
        if priority is not None:
            if not self.schema_manager.validate_value("priority_levels", priority):
                available = self.schema_manager.get_values("priority_levels")
                raise ValueError(f"Invalid priority '{priority}'. Available: {available}")
            current["priority"] = priority
        if confidence is not None:
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
            current["confidence"] = confidence
        if notes is not None:
            current["notes"] = notes
        
        current["last_updated"] = self.get_timestamp()
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        self.log_operation("edit_flag", embryo_id, flag_type=flag_type, changes={k:v for k,v in locals().items() if v is not None and k not in ['self', 'embryo_id', 'flag_type', 'current']})
        
        if self.verbose:
            print(f"🚩 Updated flag '{flag_type}' for {embryo_id}")
        
        return True
    
    def remove_flag(self, embryo_id: str, flag_type: str) -> bool:
        """Remove a flag from an embryo."""
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        if flag_type not in self.data["embryos"][embryo_id]["flags"]:
            available = list(self.data["embryos"][embryo_id]["flags"].keys())
            raise ValueError(f"Flag '{flag_type}' not found for {embryo_id}. Available: {available}")
        
        removed = self.data["embryos"][embryo_id]["flags"].pop(flag_type)
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        self.log_operation("remove_flag", embryo_id, flag_type=flag_type, removed_data=removed)
        
        if self.verbose:
            print(f"🗑️ Removed flag '{flag_type}' from {embryo_id}")
        
        return True
    
    def get_flags(self, embryo_id: str) -> dict:
        """Get all flags for an embryo."""
        if embryo_id not in self.data["embryos"]:
            return {}
        return self.data["embryos"][embryo_id]["flags"].copy()
    
    def list_flags_by_type(self, flag_type: str) -> list:
        """Find all embryos with a specific flag type."""
        embryos_with_flag = []
        for embryo_id, embryo_data in self.data["embryos"].items():
            if flag_type in embryo_data["flags"]:
                embryos_with_flag.append(embryo_id)
        return embryos_with_flag
    
    def get_high_priority_flags(self) -> dict:
        """Get all high priority and critical flags across all embryos."""
        high_priority_flags = {}
        for embryo_id, embryo_data in self.data["embryos"].items():
            for flag_type, flag_data in embryo_data["flags"].items():
                if flag_data["priority"] in ["high", "critical"]:
                    if embryo_id not in high_priority_flags:
                        high_priority_flags[embryo_id] = {}
                    high_priority_flags[embryo_id][flag_type] = flag_data
        return high_priority_flags
    
    def get_flag(self, entity_id: str, flag_type: str) -> Optional[Dict]:
        """
        Get a specific flag for an entity.
        
        Args:
            entity_id: Entity identifier
            flag_type: Type of flag to retrieve
            
        Returns:
            Flag data or None if not found
        """
        # Check snip-level flags first
        for embryo_data in self.data["embryos"].values():
            if entity_id in embryo_data.get("snips", {}):
                snip_flags = embryo_data["snips"][entity_id].get("flags", [])
                for flag in snip_flags:
                    if flag.get("flag") == flag_type:
                        return flag
        
        # Check embryo-level flags
        if entity_id in self.data["embryos"]:
            embryo_flags = self.data["embryos"][entity_id].get("flags", {})
            if flag_type in embryo_flags:
                return embryo_flags[flag_type]
        
        return None
