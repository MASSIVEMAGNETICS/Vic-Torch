# ============================================
# FILE: victorch/utils/checkpoint.py
# VERSION: v1.0.0-GODCORE-SAVELOAD
# NAME: Victor Checkpoint System
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Save and Load Model + Optimizer States.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

import numpy as np
import os

def save_checkpoint(model, optimizer, epoch, path="victor_checkpoint.npz"):
    params = {}

    # Save model params
    model_params = [p.data for p in model.parameters()]
    params['model'] = model_params

    # Save optimizer params
    if hasattr(optimizer, 'm') and hasattr(optimizer, 'v'):
        params['optimizer_m'] = optimizer.m
        params['optimizer_v'] = optimizer.v
        params['optimizer_t'] = optimizer.t

    # Save epoch
    params['epoch'] = epoch

    # Save to file
    np.savez_compressed(path, **params)
    print(f"[Victor Checkpoint] Saved to {path}")

def load_checkpoint(model, optimizer, path="victor_checkpoint.npz"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint {path} not found!")

    data = np.load(path, allow_pickle=True)

    # Load model params
    model_params = data['model']
    for p, saved_p in zip(model.parameters(), model_params):
        p.data = saved_p

    # Load optimizer params
    if hasattr(optimizer, 'm') and 'optimizer_m' in data:
        optimizer.m = data['optimizer_m']
    if hasattr(optimizer, 'v') and 'optimizer_v' in data:
        optimizer.v = data['optimizer_v']
    if hasattr(optimizer, 't') and 'optimizer_t' in data:
        optimizer.t = int(data['optimizer_t'])

    # Load epoch
    epoch = int(data['epoch'])

    print(f"[Victor Checkpoint] Loaded from {path}")
    return epoch


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
