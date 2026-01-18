"""
Vic-Torch Checkpoint Helper

Simple checkpoint save/load utilities for model persistence.

Usage:
    from victorch.checkpoint import save_checkpoint, load_checkpoint
    
    # Save a checkpoint
    save_checkpoint('model.npy', model, metadata={'epoch': 10, 'loss': 0.5})
    
    # Load a checkpoint
    checkpoint = load_checkpoint('model.npy')
    weights = checkpoint['weights']
    metadata = checkpoint['metadata']
"""

import numpy as np
from datetime import datetime


def save_checkpoint(path, model, metadata=None):
    """
    Save model checkpoint to disk.
    
    Args:
        path (str): Path where checkpoint will be saved
        model: Model object to save (should have save_weights() or state_dict() method)
        metadata (dict, optional): Additional metadata to save with checkpoint
    
    Returns:
        str: Path where checkpoint was saved
    """
    if metadata is None:
        metadata = {}
    
    # Try to extract weights from model
    weights = None
    
    # Try model.save_weights() method
    if hasattr(model, 'save_weights') and callable(getattr(model, 'save_weights')):
        try:
            weights = model.save_weights()
        except Exception:
            pass
    
    # Try model.state_dict() method (PyTorch-style)
    if weights is None and hasattr(model, 'state_dict') and callable(getattr(model, 'state_dict')):
        try:
            weights = model.state_dict()
        except Exception:
            pass
    
    # Fallback: pickle the entire model object
    if weights is None:
        import pickle
        weights = pickle.dumps(model)
    
    # Create checkpoint payload
    checkpoint = {
        'weights': weights,
        'metadata': metadata,
        'saved_at': datetime.now().isoformat()
    }
    
    # Save using numpy
    np.save(path, checkpoint, allow_pickle=True)
    
    print(f"✅ Checkpoint saved to: {path}")
    return path


def load_checkpoint(path):
    """
    Load model checkpoint from disk.
    
    Args:
        path (str): Path to checkpoint file
    
    Returns:
        dict: Dictionary containing 'weights', 'metadata', and 'saved_at'
    """
    try:
        checkpoint = np.load(path, allow_pickle=True).item()
        
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint file does not contain a valid checkpoint dictionary")
        
        # Validate checkpoint structure
        if 'weights' not in checkpoint:
            raise ValueError("Checkpoint missing 'weights' field")
        
        print(f"✅ Checkpoint loaded from: {path}")
        if 'saved_at' in checkpoint:
            print(f"   Saved at: {checkpoint['saved_at']}")
        if 'metadata' in checkpoint and checkpoint['metadata']:
            print(f"   Metadata: {checkpoint['metadata']}")
        
        return checkpoint
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
