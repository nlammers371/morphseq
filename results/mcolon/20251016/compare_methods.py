#!/usr/bin/env python3
"""
Compare classification and distribution methods.

This script will run both methods and create comparative visualizations.

Currently a placeholder - will be implemented after distribution method is ready.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from difference_detection import run_classification_test, run_distribution_test


def main():
    """Compare both methods."""
    print("Method comparison not yet implemented.")
    print("Run run_classification.py to test the classification approach.")
    return


if __name__ == "__main__":
    main()
