#!/usr/bin/env python3
"""
Vic-Torch Deterministic Training Wrapper

This script provides a CLI wrapper for running training scripts with
deterministic seeding and reproducibility features.

Usage:
    python scripts/train.py [--config CONFIG] [--script SCRIPT]

Examples:
    python scripts/train.py
    python scripts/train.py --config configs/custom.yaml
    python scripts/train.py --script Vic-Torch/train_victor.py
"""

import os
import sys
import random
import subprocess
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml
except ImportError:
    print("Warning: pyyaml not installed. Using default seed=42")
    yaml = None

try:
    import numpy as np
except ImportError:
    print("Warning: numpy not installed. Skipping numpy seeding")
    np = None


def load_config(config_path):
    """Load configuration from YAML file."""
    if yaml is None:
        return {"seed": 42}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using default seed=42")
        return {"seed": 42}


def set_deterministic_seed(seed):
    """Set deterministic seeds for reproducibility."""
    print(f"🌱 Setting deterministic seed: {seed}")
    
    # Set Python random seed
    random.seed(seed)
    
    # Set numpy seed if available
    if np is not None:
        np.random.seed(seed)
    
    # Set PYTHONHASHSEED environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✅ Deterministic seeding complete (seed={seed})")


def run_training_script(script_path, seed):
    """Run the training script as a subprocess."""
    print(f"\n🚀 Launching training script: {script_path}")
    print(f"📍 Working directory: {os.getcwd()}")
    print("-" * 60)
    
    # Ensure PYTHONHASHSEED is propagated to subprocess
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = str(seed)
    
    try:
        # Run the training script as a subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            env=env,
            check=False
        )
        
        print("-" * 60)
        print(f"✅ Training script completed with exit code: {result.returncode}")
        return result.returncode
        
    except FileNotFoundError:
        print(f"❌ Error: Training script not found: {script_path}")
        return 1
    except Exception as e:
        print(f"❌ Error running training script: {e}")
        return 1


def main():
    """Main entry point for the training wrapper."""
    parser = argparse.ArgumentParser(
        description="Vic-Torch Deterministic Training Wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration YAML file (default: configs/default.yaml)'
    )
    
    parser.add_argument(
        '--script',
        type=str,
        default='Vic-Torch/train_victor_godcore_v2.py',
        help='Path to training script to run (default: Vic-Torch/train_victor_godcore_v2.py)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔥 Vic-Torch Deterministic Training Wrapper")
    print("=" * 60)
    
    # Load configuration
    config = load_config(args.config)
    seed = config.get('seed', 42)
    
    # Set deterministic seeds
    set_deterministic_seed(seed)
    
    # Run the training script
    exit_code = run_training_script(args.script, seed)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
