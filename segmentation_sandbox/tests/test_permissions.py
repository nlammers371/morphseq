#!/usr/bin/env python3
"""
Simple test script to verify Python execution permissions
"""
import sys
import os
from datetime import datetime

print("=" * 50)
print("Python Execution Permission Test")
print("=" * 50)
print(f"Timestamp: {datetime.now()}")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Script file: {__file__}")
print("=" * 50)
print("✅ Python script execution successful!")
print("✅ You can grant permission for Python scripts in this directory")
print("=" * 50)