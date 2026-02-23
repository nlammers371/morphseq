#!/usr/bin/env python3
"""
Run distribution-based phenotype emergence analysis.

This script uses energy distance and Hotelling's T² tests
for detecting phenotypic differences.

Currently a placeholder - not yet implemented.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from difference_detection import run_distribution_test


def main():
    """Run distribution-based analysis."""
    print("Distribution-based testing not yet implemented.")
    print("Use run_classification.py for now.")
    return


if __name__ == "__main__":
    main()
