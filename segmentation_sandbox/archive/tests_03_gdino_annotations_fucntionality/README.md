# GroundedDINO Pipeline Test Suite

This comprehensive test suite validates the production readiness of the GroundedDINO detection pipeline.

## Test Structure

```
tests/
├── test_grounded_sam_utils.py         # Unit tests for core utilities
├── test_gdino_integration.py          # Integration tests
├── test_03_initial_gdino_detections.py # Script validation tests
├── run_production_tests.py            # Main test runner
└── README.md                          # This file
```

## Quick Start

### Run All Tests
```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/tests
python run_production_tests.py
```

### Run Quick Validation (5 Random Tests)
```bash
python run_production_tests.py --quick
```

### Run Specific Test Suites
```bash
# Unit tests only
python run_production_tests.py --unit

# Integration tests only  
python run_production_tests.py --integration

# Script validation tests only
python run_production_tests.py --script

# Performance tests only
python run_production_tests.py --performance
```

### Run Tests on Random Subset
```bash
# Run 10 random tests
python run_production_tests.py --subset 10

# Run 3 random integration tests
python run_production_tests.py --integration --subset 3
```

### Generate Detailed Report
```bash
python run_production_tests.py --report
```

## Test Coverage

### Unit Tests (`test_grounded_sam_utils.py`)
- ✅ IoU calculation accuracy and edge cases
- ✅ Model metadata extraction
- ✅ Annotation management and persistence
- ✅ High-quality annotation filtering
- ✅ Configuration file loading
- ✅ Error handling and validation

### Integration Tests (`test_gdino_integration.py`)
- ✅ End-to-end detection workflow
- ✅ Model loading and inference pipeline
- ✅ Annotation generation and filtering
- ✅ Experiment metadata integration
- ✅ File I/O operations
- ✅ Multi-session persistence

### Script Tests (`test_03_initial_gdino_detections.py`)
- ✅ Command-line argument parsing
- ✅ Required argument validation
- ✅ Help functionality
- ✅ Error handling patterns
- ✅ Documentation completeness
- ✅ Output formatting

### Performance Tests
- ✅ IoU calculation performance
- ✅ Memory usage monitoring
- ✅ File I/O performance
- ✅ Dependency availability

## Individual Test Files

### Running Individual Test Files

```bash
# Unit tests
python test_grounded_sam_utils.py
python test_grounded_sam_utils.py --subset 5 --verbose

# Integration tests
python test_gdino_integration.py
python test_gdino_integration.py --subset 3 --mock-model

# Script tests
python test_03_initial_gdino_detections.py
python test_03_initial_gdino_detections.py --verbose
```

## Mock Testing

For environments without the actual GroundingDINO model, tests include comprehensive mocking:

```bash
# Use mocked model for integration tests
python test_gdino_integration.py --mock-model

# Use mocked inference for script tests
python test_03_initial_gdino_detections.py --mock-inference
```

## Test Features

### 🎯 Subset Testing
- Run tests on random subsets for quick validation
- Useful for CI/CD pipelines
- Configurable subset sizes

### 🔧 Mock Support
- Comprehensive mocking for model-dependent tests
- Test business logic without requiring actual models
- Simulate various scenarios and edge cases

### 📊 Performance Monitoring
- Memory usage tracking
- Execution time measurements
- I/O performance analysis

### 📋 Detailed Reporting
- Comprehensive test reports
- Performance metrics
- Dependency analysis
- Production readiness recommendations

## Expected Output

### Successful Test Run
```
🚀 Starting Production Readiness Tests
Time: 2024-01-01 12:00:00
🎯 Running complete test suite

📦 Testing Dependencies
==================================================
   ✅ numpy: 1.21.0
   ✅ torch: 1.12.0
   ✅ cv2: 4.5.0
   ✅ matplotlib: 3.5.0
   ✅ yaml: 6.0

🔬 Running Unit Tests
==================================================
test_calculate_detection_iou_perfect_overlap ... ok
test_calculate_detection_iou_no_overlap ... ok
test_get_model_metadata_with_annotation_metadata ... ok
test_annotations_initialization_new_file ... ok
...

🧪 Running Integration Tests
==================================================
test_end_to_end_detection_workflow ... ok
test_high_quality_filtering_integration ... ok
...

📜 Running Script Tests
==================================================
test_script_argument_parsing ... ok
test_script_help ... ok
...

⚡ Running Performance Tests
==================================================
🔍 Testing IoU calculation performance...
   ✅ IoU calculation: 10000 operations in 0.123s
💾 Testing memory usage...
   ✅ Memory usage: 45.2 MB for 100 images × 10 detections
...

================================================================================
PRODUCTION READINESS TEST REPORT
================================================================================

SUMMARY:
- Total Tests: 32
- Passed: 32
- Failed: 0
- Errors: 0
- Success Rate: 100.0%

RECOMMENDATIONS:
- ✅ All tests passed - ready for production!
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the correct directory
   ```bash
   cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/tests
   ```

2. **Missing Dependencies**: Install required packages
   ```bash
   pip install numpy torch opencv-python matplotlib pyyaml psutil
   ```

3. **Permission Errors**: Check file permissions for test directories

4. **Memory Issues**: Use subset testing for resource-constrained environments
   ```bash
   python run_production_tests.py --subset 5
   ```

### Debug Mode
```bash
# Run with verbose output
python run_production_tests.py --verbose

# Run specific failing test
python test_grounded_sam_utils.py --verbose
```

## Production Deployment Checklist

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Script tests pass
- [ ] Performance tests meet requirements
- [ ] All dependencies available
- [ ] Memory usage within limits
- [ ] Error handling validated
- [ ] Documentation complete

## Contributing

When adding new features:
1. Add corresponding unit tests
2. Update integration tests if needed
3. Run full test suite
4. Update documentation

## Contact

For questions or issues with the test suite, please contact the development team.
