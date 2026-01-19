#!/usr/bin/env python3
"""
Deterministic Training Wrapper

This script provides a CLI wrapper for reproducible training runs.
It loads configuration, sets deterministic seeds, and runs the training
script as a subprocess to ensure isolation and reproducibility.

Usage:
    python scripts/train.py --config configs/default.yaml --script Vic-Torch/train_victor_godcore.py
"""

import argparse
import random
import sys
import subprocess
import os
import yaml


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        print("Warning: numpy not available, skipping numpy seed")


def load_config(config_path):
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Deterministic training wrapper')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--script', type=str, required=True,
                        help='Path to training script to execute')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine seed
    seed = args.seed if args.seed is not None else config.get('seed', 42)
    
    print(f"[Deterministic Training Wrapper]")
    print(f"  Config: {args.config}")
    print(f"  Script: {args.script}")
    print(f"  Seed: {seed}")
    
    # Set seeds for reproducibility
    set_seeds(seed)
    
    # Set environment variables for subprocess
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = str(seed)
    
    # Run training script as subprocess
    print(f"\n[Running training script...]")
    try:
        result = subprocess.run(
            [sys.executable, args.script],
            env=env,
            check=True
        )
        print(f"\n[Training completed successfully]")
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"\n[Training failed with exit code {e.returncode}]")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"Error: Training script not found: {args.script}")
        sys.exit(1)


if __name__ == '__main__':
    main()
