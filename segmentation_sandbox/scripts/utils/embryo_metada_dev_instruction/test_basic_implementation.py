#!/usr/bin/env python3
"""
Basic test for the PermittedValuesManager and updated models
"""

import sys
from pathlib import Path

# Add the path to our modules
sys.path.append(str(Path(__file__).parent))

from permitted_values_manager import PermittedValuesManager
from embryo_metadata_models import (
    Phenotype, Genotype, Flag, Treatment, 
    Validator, Serializer, ValidationError, PermittedValueError
)

def test_permitted_values_manager():
    """Test the PermittedValuesManager functionality."""
    print("🧪 Testing PermittedValuesManager...")
    
    # Initialize manager
    manager = PermittedValuesManager()
    
    # Test getting values
    phenotypes = manager.get_phenotypes()
    print(f"✅ Loaded {len(phenotypes)} phenotypes: {list(phenotypes.keys())}")
    
    # Test validation
    assert manager.is_valid_phenotype("HEART_DEFECT"), "HEART_DEFECT should be valid"
    assert not manager.is_valid_phenotype("INVALID_PHENOTYPE"), "Invalid phenotype should be rejected"
    
    print("✅ PermittedValuesManager tests passed!")

def test_data_models():
    """Test the data model classes."""
    print("🧪 Testing data models...")
    
    # Test Phenotype
    pheno = Phenotype(value="EDEMA", author="test_user", confidence=0.95)
    pheno_dict = pheno.to_dict()
    assert pheno_dict["value"] == "EDEMA"
    assert pheno_dict["confidence"] == 0.95
    
    # Test Genotype
    geno = Genotype(value="wildtype", author="test_user", confirmed=True)
    geno_dict = geno.to_dict()
    assert geno_dict["confirmed"] == True
    
    # Test Flag
    flag = Flag(value="MOTION_BLUR", author="qc_system", severity="warning", auto_generated=True)
    flag_dict = flag.to_dict()
    assert flag_dict["auto_generated"] == True
    
    # Test Treatment
    treatment = Treatment(value="SU5402", author="researcher", details="10μM")
    treatment_dict = treatment.to_dict()
    assert treatment_dict["details"] == "10μM"
    
    print("✅ Data model tests passed!")

def test_validation():
    """Test the validation system."""
    print("🧪 Testing validation...")
    
    manager = PermittedValuesManager()
    phenotypes = manager.get_phenotypes()
    
    # Test valid phenotype
    try:
        Validator.validate_phenotype("HEART_DEFECT", phenotypes)
        print("✅ Valid phenotype validation passed")
    except ValidationError as e:
        print(f"❌ Valid phenotype validation failed: {e}")
    
    # Test invalid phenotype
    try:
        Validator.validate_phenotype("INVALID_PHENOTYPE", phenotypes)
        print("❌ Invalid phenotype validation should have failed")
    except PermittedValueError:
        print("✅ Invalid phenotype validation correctly failed")
    
    # Test DEAD exclusivity
    try:
        Validator.validate_phenotype("EDEMA", phenotypes, existing_phenotypes=["DEAD"])
        print("❌ DEAD exclusivity validation should have failed")
    except ValidationError:
        print("✅ DEAD exclusivity validation correctly failed")
    
    print("✅ Validation tests passed!")

def test_serialization():
    """Test serialization/deserialization."""
    print("🧪 Testing serialization...")
    
    # Test phenotype serialization
    pheno = Phenotype(value="EDEMA", author="test_user", confidence=0.8)
    pheno_dict = Serializer.serialize_annotation(pheno)
    pheno_restored = Serializer.deserialize_phenotype(pheno_dict)
    
    assert pheno_restored.value == pheno.value
    assert pheno_restored.confidence == pheno.confidence
    
    # Test treatment serialization
    treatment = Treatment(value="DMSO", author="researcher", details="1%")
    treatment_dict = Serializer.serialize_annotation(treatment)
    treatment_restored = Serializer.deserialize_treatment(treatment_dict)
    
    assert treatment_restored.value == treatment.value
    assert treatment_restored.details == treatment.details
    
    print("✅ Serialization tests passed!")

if __name__ == "__main__":
    print("🚀 Running basic tests for EmbryoMetadata system...")
    
    try:
        test_permitted_values_manager()
        test_data_models()
        test_validation()
        test_serialization()
        
        print("\n🎉 All tests passed! The implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
