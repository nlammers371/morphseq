#!/usr/bin/env python3
"""
Production Readiness Test Suite for GroundedDINO Pipeline
=========================================================

This comprehensive test suite validates the production readiness of the
GroundedDINO pipeline with:

1. Unit tests for core utilities
2. Integration tests for the complete pipeline
3. Script validation tests
4. Performance and reliability tests
5. Error handling and edge case tests

Features:
- Subset testing for quick validation
- Mock-based testing for CI/CD environments
- Detailed reporting and analysis
- Performance benchmarking
- Production environment simulation

Usage:
    python run_production_tests.py                    # Run all tests
    python run_production_tests.py --subset 5         # Run 5 random tests
    python run_production_tests.py --quick           # Run quick validation
    python run_production_tests.py --integration     # Run integration tests only
    python run_production_tests.py --report          # Generate detailed report
"""

import os
import sys
import json
import time
import random
import unittest
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import tempfile
import shutil

# Add scripts directory to path
TEST_DIR = Path(__file__).parent
SCRIPTS_DIR = TEST_DIR.parent
SANDBOX_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

# Import test modules
from test_grounded_sam_utils import (
    TestIoUCalculation, TestModelMetadata, TestGroundedDinoAnnotations, 
    TestConfigLoading, TestSubsetRunner
)
from test_gdino_integration import TestGDinoIntegration, TestSubsetIntegration
from test_03_initial_gdino_detections import (
    TestGDinoDetectionScript, TestScriptProductionReadiness, TestScriptSubset
)


class ProductionTestRunner:
    """Main test runner for production readiness validation."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = time.time()
        self.results = {}
        self.temp_dir = tempfile.mkdtemp()
        
    def __del__(self):
        """Clean up temporary directory."""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def run_unit_tests(self, subset_size: Optional[int] = None) -> Dict:
        """Run unit tests for core utilities."""
        print("\n🔬 Running Unit Tests")
        print("=" * 50)
        
        test_classes = [
            TestIoUCalculation,
            TestModelMetadata,
            TestGroundedDinoAnnotations,
            TestConfigLoading
        ]
        
        if subset_size:
            result = TestSubsetRunner.run_subset_tests(
                test_classes, subset_size, self.verbose
            )
        else:
            suite = unittest.TestSuite()
            for test_class in test_classes:
                tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
                suite.addTests(tests)
            
            runner = unittest.TextTestRunner(verbosity=2 if self.verbose else 1)
            result = runner.run(suite)
        
        unit_results = {
            "passed": result.wasSuccessful(),
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0
        }
        
        self.results["unit_tests"] = unit_results
        return unit_results
    
    def run_integration_tests(self, subset_size: Optional[int] = None) -> Dict:
        """Run integration tests."""
        print("\n🧪 Running Integration Tests")
        print("=" * 50)
        
        if subset_size:
            result = TestSubsetIntegration.run_subset_integration_tests(
                subset_size, self.verbose
            )
        else:
            suite = unittest.TestLoader().loadTestsFromTestCase(TestGDinoIntegration)
            runner = unittest.TextTestRunner(verbosity=2 if self.verbose else 1)
            result = runner.run(suite)
        
        integration_results = {
            "passed": result.wasSuccessful(),
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0
        }
        
        self.results["integration_tests"] = integration_results
        return integration_results
    
    def run_script_tests(self, subset_size: Optional[int] = None) -> Dict:
        """Run script validation tests."""
        print("\n📜 Running Script Tests")
        print("=" * 50)
        
        if subset_size:
            result = TestScriptSubset.run_subset_script_tests(
                subset_size, self.verbose
            )
        else:
            test_classes = [TestGDinoDetectionScript, TestScriptProductionReadiness]
            suite = unittest.TestSuite()
            for test_class in test_classes:
                tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
                suite.addTests(tests)
            
            runner = unittest.TextTestRunner(verbosity=2 if self.verbose else 1)
            result = runner.run(suite)
        
        script_results = {
            "passed": result.wasSuccessful(),
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0
        }
        
        self.results["script_tests"] = script_results
        return script_results
    
    def run_performance_tests(self) -> Dict:
        """Run performance and reliability tests."""
        print("\n⚡ Running Performance Tests")
        print("=" * 50)
        
        performance_results = {
            "passed": True,
            "tests_run": 0,
            "failures": 0,
            "errors": 0,
            "metrics": {}
        }
        
        try:
            # Test 1: IoU calculation performance
            print("🔍 Testing IoU calculation performance...")
            start_time = time.time()
            
            from utils.grounded_sam_utils import calculate_detection_iou
            
            # Run many IoU calculations
            num_tests = 10000
            for i in range(num_tests):
                box1 = [random.random(), random.random(), random.random() * 0.5, random.random() * 0.5]
                box2 = [random.random(), random.random(), random.random() * 0.5, random.random() * 0.5]
                iou = calculate_detection_iou(box1, box2)
            
            iou_time = time.time() - start_time
            performance_results["metrics"]["iou_calculation_time"] = iou_time
            performance_results["tests_run"] += 1
            
            if self.verbose:
                print(f"   ✅ IoU calculation: {num_tests} operations in {iou_time:.3f}s")
            
            # Test 2: Memory usage test
            print("💾 Testing memory usage...")
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create large annotation data structure
            from utils.grounded_sam_utils import GroundedDinoAnnotations
            annotations_file = Path(self.temp_dir) / "perf_test_annotations.json"
            annotations = GroundedDinoAnnotations(annotations_file, verbose=False)
            
            # Add many annotations
            import numpy as np
            from unittest.mock import Mock
            
            mock_model = Mock()
            mock_model._annotation_metadata = {
                "model_config_path": "config.py",
                "model_weights_path": "weights.pth",
                "loading_timestamp": "2024-01-01T00:00:00",
                "model_architecture": "GroundedDINO"
            }
            
            for i in range(100):
                boxes = np.random.rand(10, 4)
                logits = np.random.rand(10)
                phrases = ["test object"] * 10
                
                annotations.add_annotation(
                    f"img_{i:03d}", "test prompt", mock_model, boxes, logits, phrases
                )
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = final_memory - initial_memory
            
            performance_results["metrics"]["memory_usage_mb"] = memory_usage
            performance_results["tests_run"] += 1
            
            if self.verbose:
                print(f"   ✅ Memory usage: {memory_usage:.1f} MB for 100 images × 10 detections")
            
            # Test 3: File I/O performance
            print("💾 Testing file I/O performance...")
            start_time = time.time()
            
            # Save and load multiple times
            for i in range(10):
                annotations.save()
                new_annotations = GroundedDinoAnnotations(annotations_file, verbose=False)
            
            io_time = time.time() - start_time
            performance_results["metrics"]["file_io_time"] = io_time
            performance_results["tests_run"] += 1
            
            if self.verbose:
                print(f"   ✅ File I/O: 10 save/load cycles in {io_time:.3f}s")
            
        except Exception as e:
            performance_results["passed"] = False
            performance_results["errors"] += 1
            if self.verbose:
                print(f"   ❌ Performance test error: {e}")
        
        self.results["performance_tests"] = performance_results
        return performance_results
    
    def run_dependency_tests(self) -> Dict:
        """Test dependency availability and versions."""
        print("\n📦 Testing Dependencies")
        print("=" * 50)
        
        dependency_results = {
            "passed": True,
            "tests_run": 0,
            "failures": 0,
            "errors": 0,
            "dependencies": {}
        }
        
        required_packages = [
            "numpy", "torch", "cv2", "matplotlib", "yaml", "pathlib", "json"
        ]
        
        for package in required_packages:
            try:
                if package == "cv2":
                    import cv2
                    version = cv2.__version__
                elif package == "yaml":
                    import yaml
                    version = getattr(yaml, '__version__', 'unknown')
                elif package == "torch":
                    import torch
                    version = torch.__version__
                elif package == "numpy":
                    import numpy as np
                    version = np.__version__
                elif package == "matplotlib":
                    import matplotlib
                    version = matplotlib.__version__
                else:
                    exec(f"import {package}")
                    version = "available"
                
                dependency_results["dependencies"][package] = {
                    "available": True,
                    "version": version
                }
                dependency_results["tests_run"] += 1
                
                if self.verbose:
                    print(f"   ✅ {package}: {version}")
                    
            except ImportError as e:
                dependency_results["dependencies"][package] = {
                    "available": False,
                    "error": str(e)
                }
                dependency_results["failures"] += 1
                dependency_results["passed"] = False
                
                if self.verbose:
                    print(f"   ❌ {package}: {e}")
        
        self.results["dependency_tests"] = dependency_results
        return dependency_results
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        duration = time.time() - self.start_time
        
        report = f"""
{'='*80}
PRODUCTION READINESS TEST REPORT
{'='*80}

Test Run Information:
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Duration: {duration:.2f} seconds
- Environment: {sys.platform}
- Python: {sys.version.split()[0]}

"""
        
        # Summary table
        total_tests = 0
        total_failures = 0
        total_errors = 0
        
        for test_type, results in self.results.items():
            if isinstance(results, dict) and "tests_run" in results:
                total_tests += results["tests_run"]
                total_failures += results["failures"]
                total_errors += results["errors"]
        
        report += f"""
SUMMARY:
- Total Tests: {total_tests}
- Passed: {total_tests - total_failures - total_errors}
- Failed: {total_failures}
- Errors: {total_errors}
- Success Rate: {((total_tests - total_failures - total_errors) / max(total_tests, 1)) * 100:.1f}%

"""
        
        # Detailed results
        for test_type, results in self.results.items():
            if isinstance(results, dict):
                status = "✅ PASSED" if results.get("passed", False) else "❌ FAILED"
                report += f"""
{test_type.upper().replace('_', ' ')}:
- Status: {status}
- Tests Run: {results.get('tests_run', 0)}
- Failures: {results.get('failures', 0)}
- Errors: {results.get('errors', 0)}
"""
                
                # Add specific metrics
                if "metrics" in results:
                    report += "- Metrics:\n"
                    for metric, value in results["metrics"].items():
                        report += f"  • {metric}: {value}\n"
                
                if "dependencies" in results:
                    report += "- Dependencies:\n"
                    for dep, info in results["dependencies"].items():
                        status = "✅" if info["available"] else "❌"
                        version = info.get("version", "unknown")
                        report += f"  • {dep}: {status} {version}\n"
        
        report += f"""
{'='*80}
RECOMMENDATIONS:
"""
        
        # Generate recommendations
        recommendations = []
        
        if total_errors > 0:
            recommendations.append("🔧 Fix errors before production deployment")
        
        if total_failures > 0:
            recommendations.append("🔧 Address test failures")
        
        if self.results.get("dependency_tests", {}).get("passed", True) is False:
            recommendations.append("📦 Install missing dependencies")
        
        if "performance_tests" in self.results:
            perf = self.results["performance_tests"]["metrics"]
            if perf.get("memory_usage_mb", 0) > 500:
                recommendations.append("💾 Monitor memory usage in production")
            if perf.get("file_io_time", 0) > 5:
                recommendations.append("💾 Optimize file I/O operations")
        
        if not recommendations:
            recommendations.append("✅ All tests passed - ready for production!")
        
        for rec in recommendations:
            report += f"- {rec}\n"
        
        report += f"""
{'='*80}
"""
        
        return report
    
    def run_all_tests(self, subset_size: Optional[int] = None) -> Dict:
        """Run all test suites."""
        print(f"\n🚀 Starting Production Readiness Tests")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if subset_size:
            print(f"🎯 Running subset tests (size: {subset_size})")
        else:
            print("🎯 Running complete test suite")
        
        # Run all test suites
        self.run_dependency_tests()
        self.run_unit_tests(subset_size)
        self.run_integration_tests(subset_size)
        self.run_script_tests(subset_size)
        self.run_performance_tests()
        
        return self.results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Production readiness test suite for GroundedDINO pipeline"
    )
    parser.add_argument("--subset", type=int, help="Run tests on random subset of this size")
    parser.add_argument("--quick", action="store_true", help="Run quick validation (subset=5)")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--script", action="store_true", help="Run script tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set subset size
    subset_size = args.subset
    if args.quick:
        subset_size = 5
    
    # Create test runner
    runner = ProductionTestRunner(verbose=args.verbose)
    
    # Run specific test suites
    if args.unit:
        runner.run_unit_tests(subset_size)
    elif args.integration:
        runner.run_integration_tests(subset_size)
    elif args.script:
        runner.run_script_tests(subset_size)
    elif args.performance:
        runner.run_performance_tests()
    else:
        # Run all tests
        runner.run_all_tests(subset_size)
    
    # Generate and display report
    if args.report or not any([args.unit, args.integration, args.script, args.performance]):
        report = runner.generate_report()
        print(report)
        
        # Save report to file
        report_file = Path("production_test_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file}")
    
    # Return appropriate exit code
    all_passed = all(
        results.get("passed", False) 
        for results in runner.results.values()
        if isinstance(results, dict)
    )
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
