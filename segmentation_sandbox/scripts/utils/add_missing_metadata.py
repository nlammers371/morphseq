#!/usr/bin/env python3
"""
Add Missing Model Metadata to Existing Annotations
==================================================

This script adds model metadata to existing high-quality annotations for backward compatibility.
It extracts the model metadata from regular annotations and adds it to high-quality annotations.

Usage:
    python add_missing_metadata.py <annotations_file>
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

def extract_model_metadata_from_regular_annotations(annotations_data: Dict) -> Dict[str, Dict]:
    """Extract model metadata from regular annotations organized by model."""
    model_metadata_by_weights = {}
    
    for image_id, image_data in annotations_data.get("images", {}).items():
        for annotation in image_data.get("annotations", []):
            if annotation.get("prompt") == "individual embryo":
                model_metadata = annotation.get("model_metadata", {})
                if model_metadata:
                    weights_path = model_metadata.get("model_weights_path", "unknown")
                    weights_name = Path(weights_path).name if weights_path != "unknown" else "unknown"
                    
                    if weights_name not in model_metadata_by_weights:
                        model_metadata_by_weights[weights_name] = model_metadata
    
    return model_metadata_by_weights

def add_metadata_to_high_quality_annotations(annotations_file: Path, 
                                            base_model_weights: str = "groundingdino_swint_ogc.pth",
                                            finetuned_model_weights: str = "checkpoint_best_regular.pth"):
    """Add missing model metadata to high-quality annotations."""
    
    print(f"🔧 Adding missing metadata to: {annotations_file}")
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Extract model metadata from regular annotations
    model_metadata_by_weights = extract_model_metadata_from_regular_annotations(data)
    
    print(f"📊 Found model metadata for {len(model_metadata_by_weights)} different models:")
    for weights_name, metadata in model_metadata_by_weights.items():
        print(f"   • {weights_name}: {metadata.get('model_architecture', 'Unknown')}")
    
    # Check high-quality annotations
    hq_annotations = data.get("high_quality_annotations", {})
    
    if not hq_annotations:
        print("❌ No high-quality annotations found")
        return False
    
    print(f"📋 Processing {len(hq_annotations)} high-quality annotation experiments...")
    
    # Determine which model metadata to use based on filename or user preference
    base_metadata = None
    finetuned_metadata = None
    
    for weights_name, metadata in model_metadata_by_weights.items():
        if base_model_weights in weights_name or "swint" in weights_name.lower():
            base_metadata = metadata
        elif finetuned_model_weights in weights_name or "checkpoint" in weights_name.lower():
            finetuned_metadata = metadata
    
    # Determine which metadata to use
    if "finetuned" in str(annotations_file).lower():
        preferred_metadata = finetuned_metadata or base_metadata
        model_type = "finetuned"
    else:
        preferred_metadata = base_metadata or finetuned_metadata
        model_type = "base"
    
    if not preferred_metadata:
        print("❌ No suitable model metadata found in regular annotations")
        return False
    
    print(f"🎯 Using {model_type} model metadata:")
    print(f"   Architecture: {preferred_metadata.get('model_architecture')}")
    print(f"   Config: {preferred_metadata.get('model_config_path')}")
    print(f"   Weights: {preferred_metadata.get('model_weights_path')}")
    
    # Add metadata to high-quality annotations
    updated_count = 0
    for exp_id, exp_data in hq_annotations.items():
        if "model_metadata" not in exp_data:
            exp_data["model_metadata"] = preferred_metadata.copy()
            updated_count += 1
            print(f"   ✅ Added metadata to experiment {exp_id}")
        else:
            print(f"   ⏭️  Experiment {exp_id} already has metadata")
    
    if updated_count > 0:
        # Update last_updated timestamp
        if "file_info" in data:
            data["file_info"]["last_updated"] = datetime.now().isoformat()
        
        # Backup original file
        backup_file = annotations_file.with_suffix('.json.backup')
        if backup_file.exists():
            backup_file.unlink()
        
        annotations_file.rename(backup_file)
        print(f"💾 Created backup: {backup_file.name}")
        
        # Save updated annotations
        with open(annotations_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Updated {updated_count} experiments with model metadata")
        print(f"💾 Saved updated annotations to: {annotations_file}")
        return True
    else:
        print("✅ All experiments already have model metadata")
        return False

def main():
    parser = argparse.ArgumentParser(description="Add missing model metadata to annotations")
    parser.add_argument("annotations_file", help="Path to annotations JSON file")
    parser.add_argument("--base-weights", default="groundingdino_swint_ogc.pth",
                       help="Base model weights filename pattern")
    parser.add_argument("--finetuned-weights", default="checkpoint_best_regular.pth",
                       help="Finetuned model weights filename pattern")
    
    args = parser.parse_args()
    
    annotations_file = Path(args.annotations_file)
    
    if not annotations_file.exists():
        print(f"❌ File not found: {annotations_file}")
        return 1
    
    success = add_metadata_to_high_quality_annotations(
        annotations_file,
        args.base_weights,
        args.finetuned_weights
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
