"""
Embryo Treatment Manager
Handles all treatment-related operations for EmbryoMetadata.
Supports multiple treatments per embryo with warning system.
Designed as a mixin class to keep embryo_metadata.py manageable.
"""

from typing import Dict, List, Optional
from embryo_metadata_models import Treatment


class EmbryoTreatmentManager:
    """
    Mixin class for treatment management operations.
    
    Supports multiple treatments per embryo (e.g., chemical + temperature)
    with warning system for complex experimental designs.
    
    Required attributes from parent class:
    - self.data: Dict containing embryo metadata
    - self.schema_manager: PermittedValuesManager instance
    - self.verbose: bool for logging
    - self.get_timestamp(): method for timestamps
    - self.log_operation(): method for operation logging
    - self.validate_id_format(): method for ID validation
    """
    
    def add_treatment(self, embryo_id: str, treatment_name: str, 
                     concentration: Optional[str] = None, 
                     duration: Optional[str] = None,
                     temperature: Optional[str] = None,
                     confidence: float = 1.0, notes: str = "", 
                     overwrite: bool = False) -> bool:
        """
        Add a treatment to an embryo. Supports multiple treatments per embryo.
        
        Note: Multiple treatments per embryo are allowed (e.g., chemical + temperature).
        A warning will be issued if more than one treatment is detected.
        """
        # Validate embryo ID
        if not self.validate_id_format(embryo_id, "embryo"):
            raise ValueError(f"Invalid embryo ID format: {embryo_id}")
        
        # Validate treatment name
        if not self.schema_manager.is_valid_treatment(treatment_name):
            available = list(self.schema_manager.get_treatments().keys())
            raise ValueError(f"Invalid treatment '{treatment_name}'. Available: {available}")
        
        # Validate confidence
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
        
        # Check if embryo exists in data
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        # Initialize treatments dict if not exists
        if "treatments" not in self.data["embryos"][embryo_id]:
            self.data["embryos"][embryo_id]["treatments"] = {}
        
        # Check for existing treatment
        existing_treatments = self.data["embryos"][embryo_id]["treatments"]
        if treatment_name in existing_treatments and not overwrite:
            raise ValueError(f"Treatment '{treatment_name}' already exists for {embryo_id}. Use overwrite=True to replace")
        
        # Create treatment data
        treatment_data = {
            "concentration": concentration, "duration": duration, "temperature": temperature,
            "confidence": confidence, "notes": notes, "author": "manual",
            "timestamp": self.get_timestamp(), "last_updated": self.get_timestamp()
        }
        
        # Add treatment
        self.data["embryos"][embryo_id]["treatments"][treatment_name] = treatment_data
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        # Check for multiple treatments and warn
        treatment_count = len(existing_treatments) + (0 if treatment_name in existing_treatments else 1)
        if treatment_count > 1:
            treatment_list = list(existing_treatments.keys())
            if treatment_name not in treatment_list:
                treatment_list.append(treatment_name)
            if self.verbose:
                print(f"⚠️  WARNING: Embryo {embryo_id} now has {treatment_count} treatments: {treatment_list}")
                print("   Multiple treatments may indicate complex experimental design.")
        
        # Log operation
        self.log_operation("add_treatment", embryo_id, treatment=treatment_name, data=treatment_data, total_treatments=treatment_count)
        
        if self.verbose:
            print(f"✅ Added treatment '{treatment_name}' to {embryo_id}")
        
        return True
    
    def edit_treatment(self, embryo_id: str, treatment_name: str, concentration: Optional[str] = None,
                      duration: Optional[str] = None, temperature: Optional[str] = None,
                      confidence: Optional[float] = None, notes: Optional[str] = None) -> bool:
        """Edit an existing treatment."""
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        treatments = self.data["embryos"][embryo_id].get("treatments", {})
        if treatment_name not in treatments:
            available = list(treatments.keys())
            raise ValueError(f"Treatment '{treatment_name}' not found for {embryo_id}. Available: {available}")
        
        current = treatments[treatment_name]
        
        if concentration is not None:
            current["concentration"] = concentration
        if duration is not None:
            current["duration"] = duration
        if temperature is not None:
            current["temperature"] = temperature
        if confidence is not None:
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
            current["confidence"] = confidence
        if notes is not None:
            current["notes"] = notes
        
        current["last_updated"] = self.get_timestamp()
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        changes = {k:v for k,v in locals().items() if v is not None and k not in ['self', 'embryo_id', 'treatment_name', 'current', 'treatments']}
        self.log_operation("edit_treatment", embryo_id, treatment=treatment_name, changes=changes)
        
        if self.verbose:
            print(f"💊 Updated treatment '{treatment_name}' for {embryo_id}")
        
        return True
    
    def remove_treatment(self, embryo_id: str, treatment_name: str) -> bool:
        """Remove a treatment from an embryo."""
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        treatments = self.data["embryos"][embryo_id].get("treatments", {})
        if treatment_name not in treatments:
            available = list(treatments.keys())
            raise ValueError(f"Treatment '{treatment_name}' not found for {embryo_id}. Available: {available}")
        
        removed = treatments.pop(treatment_name)
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        self.log_operation("remove_treatment", embryo_id, treatment=treatment_name, removed_data=removed)
        
        if self.verbose:
            print(f"🗑️ Removed treatment '{treatment_name}' from {embryo_id}")
        
        return True
    
    def get_treatments(self, embryo_id: str) -> dict:
        """Get all treatments for an embryo."""
        if embryo_id not in self.data["embryos"]:
            return {}
        return self.data["embryos"][embryo_id].get("treatments", {}).copy()
    
    def list_treatments_by_type(self, treatment_name: str) -> list:
        """Find all embryos with a specific treatment type."""
        embryos_with_treatment = []
        for embryo_id, embryo_data in self.data["embryos"].items():
            if treatment_name in embryo_data.get("treatments", {}):
                embryos_with_treatment.append(embryo_id)
        return embryos_with_treatment
    
    def get_multi_treatment_embryos(self) -> dict:
        """Get all embryos with multiple treatments."""
        multi_treatment = {}
        for embryo_id, embryo_data in self.data["embryos"].items():
            treatments = embryo_data.get("treatments", {})
            if len(treatments) > 1:
                multi_treatment[embryo_id] = list(treatments.keys())
        return multi_treatment
    
    def get_treatment_combinations(self) -> dict:
        """Get all unique treatment combinations in the dataset."""
        combinations = {}
        for embryo_id, embryo_data in self.data["embryos"].items():
            treatments = embryo_data.get("treatments", {})
            if treatments:
                combo = tuple(sorted(treatments.keys()))
                if combo not in combinations:
                    combinations[combo] = []
                combinations[combo].append(embryo_id)
        return combinations
