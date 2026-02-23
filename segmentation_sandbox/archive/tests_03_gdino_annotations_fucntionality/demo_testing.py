#!/usr/bin/env python3
"""
GroundedDINO Pipeline Testing Demo
=================================

This script demonstrates the production-ready testing framework for the
GroundedDINO detection pipeline.

Usage:
    python demo_testing.py
"""

import sys
import subprocess
from pathlib import Path

def run_demo():
    """Run testing demonstration."""
    
    print("🎯 GroundedDINO Pipeline Testing Demo")
    print("=" * 60)
    
    test_dir = Path(__file__).parent
    
    print("\n1. 🧪 Running Unit Tests (subset)")
    print("-" * 30)
    result = subprocess.run([
        sys.executable, str(test_dir / "test_grounded_sam_utils.py"), 
        "--subset", "3", "--verbose"
    ], cwd=test_dir)
    
    print("\n2. 🔬 Running Integration Tests (subset)")
    print("-" * 30)
    result = subprocess.run([
        sys.executable, str(test_dir / "test_gdino_integration.py"), 
        "--subset", "2", "--verbose"
    ], cwd=test_dir)
    
    print("\n3. 📜 Running Script Tests")
    print("-" * 30)
    result = subprocess.run([
        sys.executable, str(test_dir / "test_03_initial_gdino_detections.py"), 
        "--verbose"
    ], cwd=test_dir)
    
    print("\n4. 🚀 Running Complete Test Suite (quick)")
    print("-" * 30)
    result = subprocess.run([
        sys.executable, str(test_dir / "run_production_tests.py"), 
        "--quick", "--verbose"
    ], cwd=test_dir)
    
    print("\n✅ Demo completed!")
    print("\nAvailable test commands:")
    print("  python run_production_tests.py --quick      # Quick validation")
    print("  python run_production_tests.py --subset 5   # Run 5 random tests")
    print("  python run_production_tests.py              # Complete test suite")
    print("  python run_production_tests.py --report     # Detailed report")
    
    print("\nIndividual test files:")
    print("  python test_grounded_sam_utils.py           # Unit tests")
    print("  python test_gdino_integration.py            # Integration tests")
    print("  python test_03_initial_gdino_detections.py  # Script tests")


if __name__ == "__main__":
    run_demo()
