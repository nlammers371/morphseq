#!/usr/bin/env python3
"""
Demo: BaseAnnotationParser Usage Across MorphSeq Pipeline
Shows how different annotation classes can inherit from the base parser
"""

import sys
from pathlib import Path
import json
import tempfile

# Add the path to our modules
sys.path.append(str(Path(__file__).parent))

from base_annotation_parser import BaseAnnotationParser

class MockGroundedSamAnnotations(BaseAnnotationParser):
    """Example of how GroundedSamAnnotations would inherit from BaseAnnotationParser."""
    
    def __init__(self, filepath):
        super().__init__(filepath, verbose=True)
        self.annotations = self._load_or_create()
    
    def _load_or_create(self):
        if self.filepath.exists():
            return self.load_json()
        else:
            return {
                "experiments": {},
                "embryo_ids": [],
                "snip_ids": [],
                "file_info": {
                    "created": self.get_timestamp(),
                    "version": "1.0"
                }
            }
    
    def add_embryo_annotation(self, embryo_id: str, snip_id: str, bbox: list):
        """Add embryo annotation using base parser validation."""
        # Validate IDs using base parser
        if not self.validate_id_format(embryo_id, "embryo"):
            raise ValueError(f"Invalid embryo ID: {embryo_id}")
        
        if not self.validate_id_format(snip_id, "snip"):
            raise ValueError(f"Invalid snip ID: {snip_id}")
        
        # Extract components using base parser
        embryo_info = self.parse_id(embryo_id)
        snip_info = self.parse_id(snip_id)
        
        exp_id = embryo_info["experiment_id"]
        
        # Add annotation structure
        if exp_id not in self.annotations["experiments"]:
            self.annotations["experiments"][exp_id] = {"embryos": {}}
        
        if embryo_id not in self.annotations["experiments"][exp_id]["embryos"]:
            self.annotations["experiments"][exp_id]["embryos"][embryo_id] = {"snips": {}}
        
        self.annotations["experiments"][exp_id]["embryos"][embryo_id]["snips"][snip_id] = {
            "bbox": bbox,
            "frame": self.extract_frame_number(snip_id)
        }
        
        # Track IDs
        if embryo_id not in self.annotations["embryo_ids"]:
            self.annotations["embryo_ids"].append(embryo_id)
        
        if snip_id not in self.annotations["snip_ids"]:
            self.annotations["snip_ids"].append(snip_id)
        
        # Log operation using base parser
        self.log_operation("add_embryo_annotation", embryo_id, 
                          snip_id=snip_id, frame=snip_info["frame"])
    
    def save(self):
        """Save using base parser atomic save."""
        self.annotations["file_info"]["last_updated"] = self.get_timestamp()
        self.save_json(self.annotations)
        self.mark_saved()

class MockExperimentQC(BaseAnnotationParser):
    """Example of how ExperimentQC would inherit from BaseAnnotationParser."""
    
    def __init__(self, filepath):
        super().__init__(filepath, verbose=True)
        self.qc_data = self._load_or_create()
    
    def _load_or_create(self):
        if self.filepath.exists():
            return self.load_json()
        else:
            return {
                "experiments": {},
                "summary": {"total_experiments": 0, "total_issues": 0}
            }
    
    def add_qc_flag(self, entity_id: str, flag: str, severity: str = "warning"):
        """Add QC flag using base parser ID parsing."""
        # Parse entity using base parser
        parsed = self.parse_id(entity_id)
        entity_type = parsed["type"]
        
        if entity_type == "unknown":
            raise ValueError(f"Invalid entity ID: {entity_id}")
        
        exp_id = self.get_experiment_id_from_entity(entity_id)
        
        # Initialize experiment if needed
        if exp_id not in self.qc_data["experiments"]:
            self.qc_data["experiments"][exp_id] = {"flags": {}}
        
        # Add flag
        if entity_id not in self.qc_data["experiments"][exp_id]["flags"]:
            self.qc_data["experiments"][exp_id]["flags"][entity_id] = []
        
        flag_entry = {
            "flag": flag,
            "severity": severity,
            "entity_type": entity_type,
            "timestamp": self.get_timestamp()
        }
        
        self.qc_data["experiments"][exp_id]["flags"][entity_id].append(flag_entry)
        
        # Update summary
        self.qc_data["summary"]["total_issues"] += 1
        
        # Log using base parser
        self.log_operation("add_qc_flag", entity_id, flag=flag, severity=severity)
    
    def save(self):
        """Save using base parser."""
        self.save_json(self.qc_data)
        self.mark_saved()

def demo_pipeline_integration():
    """Demonstrate how BaseAnnotationParser unifies the pipeline."""
    print("🚀 Demo: BaseAnnotationParser in MorphSeq Pipeline")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # 1. GroundedSam Annotations
        print("\n📍 Step 1: GroundedSam Annotations")
        sam_file = tmp_path / "sam_annotations.json"
        sam = MockGroundedSamAnnotations(sam_file)
        
        # Add some annotations
        sam.add_embryo_annotation("20240411_A01_e01", "20240411_A01_e01_0001", [10, 20, 50, 60])
        sam.add_embryo_annotation("20240411_A01_e01", "20240411_A01_e01_0002", [12, 22, 52, 62])
        sam.add_embryo_annotation("20240411_A01_e02", "20240411_A01_e02_0001", [100, 200, 150, 250])
        
        print(f"✅ Added annotations for {len(sam.annotations['embryo_ids'])} embryos")
        print(f"   Total snips: {len(sam.annotations['snip_ids'])}")
        print(f"   Recent changes: {len(sam.get_recent_changes())}")
        
        sam.save()
        print(f"💾 Saved SAM annotations ({sam.get_file_stats()['size']} bytes)")
        
        # 2. Experiment QC
        print("\n🔍 Step 2: Experiment QC")
        qc_file = tmp_path / "experiment_qc.json"
        qc = MockExperimentQC(qc_file)
        
        # Add QC flags using the same IDs
        qc.add_qc_flag("20240411_A01_e01_0001", "MOTION_BLUR", "warning")
        qc.add_qc_flag("20240411_A01", "LOW_QUALITY", "warning")
        qc.add_qc_flag("20240411", "INCOMPLETE_DATA", "error")
        
        print(f"✅ Added QC flags, total issues: {qc.qc_data['summary']['total_issues']}")
        print(f"   Recent changes: {len(qc.get_recent_changes())}")
        
        qc.save()
        print(f"💾 Saved QC data ({qc.get_file_stats()['size']} bytes)")
        
        # 3. Demonstrate ID parsing consistency
        print("\n🧬 Step 3: ID Parsing Consistency")
        test_ids = ["20240411", "20240411_A01", "20240411_A01_e01", "20240411_A01_e01_0001"]
        
        for test_id in test_ids:
            sam_parsed = sam.parse_id(test_id)
            qc_parsed = qc.parse_id(test_id)
            
            # Both should parse the same way
            assert sam_parsed == qc_parsed
            
            exp_id_sam = sam.get_experiment_id_from_entity(test_id)
            exp_id_qc = qc.get_experiment_id_from_entity(test_id)
            
            assert exp_id_sam == exp_id_qc
            
            print(f"   {test_id} -> {sam_parsed['type']} (exp: {exp_id_sam})")
        
        # 4. Demonstrate change tracking
        print("\n📝 Step 4: Change Tracking")
        
        print("SAM Recent Changes:")
        for change in sam.get_recent_changes(3):
            op = change['operation']
            details = change['details']
            print(f"   - {op}: {details.get('entity_id', 'N/A')}")
        
        print("QC Recent Changes:")
        for change in qc.get_recent_changes(3):
            op = change['operation']
            details = change['details']
            print(f"   - {op}: {details.get('entity_id', 'N/A')} ({details.get('flag', 'N/A')})")
        
        # 5. Show file backup functionality
        print("\n💾 Step 5: Backup Functionality")
        
        # Modify and save again to create backup
        sam.add_embryo_annotation("20240411_A01_e03", "20240411_A01_e03_0001", [300, 400, 350, 450])
        sam.save()  # This should create a backup
        
        # List backup files
        backup_files = list(tmp_path.glob("*.backup.*"))
        print(f"   Created {len(backup_files)} backup files")
        for backup in backup_files:
            print(f"   📦 {backup.name}")
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("\n💡 Key Benefits of BaseAnnotationParser:")
        print("   ✅ Consistent ID parsing across all pipeline components")
        print("   ✅ Unified file I/O with atomic saves and backups")
        print("   ✅ Built-in change tracking for audit trails")
        print("   ✅ Common utilities reduce code duplication")
        print("   ✅ Easy to extend for new annotation types")

if __name__ == "__main__":
    demo_pipeline_integration()
