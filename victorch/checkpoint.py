"""
Checkpoint Helper for Vic-Torch

This module provides utilities for saving and loading model checkpoints
with support for multiple backend formats (numpy, pickle, state_dict).

Usage:
    from victorch.checkpoint import save_checkpoint, load_checkpoint
    
    # Save checkpoint
    save_checkpoint(model, optimizer, epoch=10, path="checkpoint.npz")
    
    # Load checkpoint
    epoch = load_checkpoint(model, optimizer, path="checkpoint.npz")
"""

import numpy as np
import pickle
import os


def save_checkpoint(model, optimizer=None, epoch=0, metadata=None, path="checkpoint.npz"):
    """
    Save model checkpoint with weights and metadata.
    
    Attempts multiple strategies:
    1. Try model.save_weights() if available
    2. Try to extract state_dict() if available
    3. Fallback to pickle for full object serialization
    
    Args:
        model: Model instance to save
        optimizer: Optional optimizer instance to save
        epoch: Current epoch number
        metadata: Optional dictionary of additional metadata
        path: Path to save checkpoint file
    """
    checkpoint_data = {
        'epoch': epoch,
        'metadata': metadata or {}
    }
    
    # Strategy 1: Try to save via model.save_weights()
    if hasattr(model, 'save_weights'):
        try:
            weights_path = path.replace('.npz', '_weights.npz')
            model.save_weights(weights_path)
            checkpoint_data['weights_path'] = weights_path
            print(f"[Checkpoint] Saved model weights to {weights_path}")
        except Exception as e:
            print(f"[Checkpoint] Warning: save_weights() failed: {e}")
    
    # Strategy 2: Try to extract state_dict
    if hasattr(model, 'state_dict'):
        try:
            checkpoint_data['model_state_dict'] = model.state_dict()
            print(f"[Checkpoint] Saved model state_dict")
        except Exception as e:
            print(f"[Checkpoint] Warning: state_dict() failed: {e}")
    
    # Strategy 3: Try to save parameters
    if hasattr(model, 'parameters'):
        try:
            params = [p.data for p in model.parameters()]
            checkpoint_data['model_params'] = params
            print(f"[Checkpoint] Saved {len(params)} model parameters")
        except Exception as e:
            print(f"[Checkpoint] Warning: parameters() failed: {e}")
    
    # Save optimizer state if provided
    if optimizer is not None:
        if hasattr(optimizer, 'state_dict'):
            try:
                checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
            except Exception as e:
                print(f"[Checkpoint] Warning: optimizer state_dict failed: {e}")
        
        # Save Adam-specific parameters if available
        if hasattr(optimizer, 'm') and hasattr(optimizer, 'v'):
            checkpoint_data['optimizer_m'] = optimizer.m
            checkpoint_data['optimizer_v'] = optimizer.v
            if hasattr(optimizer, 't'):
                checkpoint_data['optimizer_t'] = optimizer.t
    
    # Save to file using numpy's compressed format
    try:
        np.savez_compressed(path, **checkpoint_data)
        print(f"[Checkpoint] Successfully saved to {path}")
    except Exception as e:
        # Fallback to pickle if numpy fails
        print(f"[Checkpoint] numpy save failed: {e}, trying pickle...")
        pickle_path = path.replace('.npz', '.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        print(f"[Checkpoint] Saved to {pickle_path} using pickle")


def load_checkpoint(model, optimizer=None, path="checkpoint.npz"):
    """
    Load model checkpoint from file.
    
    Args:
        model: Model instance to load weights into
        optimizer: Optional optimizer instance to load state into
        path: Path to checkpoint file
        
    Returns:
        epoch: The epoch number from the checkpoint
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    try:
        # Try loading with numpy
        data = np.load(path, allow_pickle=True)
        
        # Load model weights/parameters
        if 'model_state_dict' in data:
            if hasattr(model, 'load_state_dict'):
                model.load_state_dict(data['model_state_dict'].item())
                print(f"[Checkpoint] Loaded model state_dict")
        
        if 'model_params' in data:
            if hasattr(model, 'parameters'):
                model_params = data['model_params']
                for p, saved_p in zip(model.parameters(), model_params):
                    p.data = saved_p
                print(f"[Checkpoint] Loaded model parameters")
        
        if 'weights_path' in data:
            weights_path = str(data['weights_path'])
            if hasattr(model, 'load_weights') and os.path.exists(weights_path):
                model.load_weights(weights_path)
                print(f"[Checkpoint] Loaded model weights from {weights_path}")
        
        # Load optimizer state if provided
        if optimizer is not None:
            if 'optimizer_state_dict' in data:
                if hasattr(optimizer, 'load_state_dict'):
                    optimizer.load_state_dict(data['optimizer_state_dict'].item())
                    print(f"[Checkpoint] Loaded optimizer state_dict")
            
            # Load Adam-specific parameters
            if 'optimizer_m' in data and hasattr(optimizer, 'm'):
                optimizer.m = data['optimizer_m']
            if 'optimizer_v' in data and hasattr(optimizer, 'v'):
                optimizer.v = data['optimizer_v']
            if 'optimizer_t' in data and hasattr(optimizer, 't'):
                optimizer.t = int(data['optimizer_t'])
        
        epoch = int(data['epoch']) if 'epoch' in data else 0
        print(f"[Checkpoint] Loaded from {path} (epoch {epoch})")
        
        return epoch
        
    except Exception as e:
        # Try pickle as fallback
        pickle_path = path.replace('.npz', '.pkl')
        if os.path.exists(pickle_path):
            print(f"[Checkpoint] Trying pickle fallback...")
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            epoch = data.get('epoch', 0)
            print(f"[Checkpoint] Loaded from {pickle_path} (epoch {epoch})")
            return epoch
        else:
            raise Exception(f"Failed to load checkpoint: {e}")
