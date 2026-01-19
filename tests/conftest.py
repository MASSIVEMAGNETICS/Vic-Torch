"""
Pytest configuration for Vic-Torch tests.

This file sets up the Python path to allow imports from the Vic-Torch directory.
"""

import sys
import os

# Add Vic-Torch directory to the Python path
test_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(test_dir)
vic_torch_dir = os.path.join(repo_root, 'Vic-Torch')

if os.path.exists(vic_torch_dir):
    sys.path.insert(0, vic_torch_dir)
