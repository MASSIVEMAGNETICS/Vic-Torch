"""
Minimal test for victorch.core.tensor.Tensor

Tests basic Tensor functionality with defensive imports.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add Vic-Torch to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Vic-Torch"))


def test_tensor_basic():
    """Test basic Tensor instantiation and properties."""
    try:
        from victorch.core.tensor import Tensor
    except ImportError:
        pytest.skip("victorch.core.tensor not available")
    
    # Create a Tensor with a small numpy array
    data = np.array([1, 2, 3, 4, 5])
    t = Tensor(data)
    
    # Assert that the tensor has data attribute
    assert hasattr(t, 'data'), "Tensor should have 'data' attribute"
    assert t.data is not None, "Tensor data should not be None"
    
    # Assert shape matches
    assert t.data.shape == (5,), f"Expected shape (5,), got {t.data.shape}"
    
    # Assert data values match
    assert np.array_equal(t.data, data), "Tensor data should match input array"


def test_tensor_2d():
    """Test 2D Tensor instantiation."""
    try:
        from victorch.core.tensor import Tensor
    except ImportError:
        pytest.skip("victorch.core.tensor not available")
    
    # Create a 2D Tensor
    data = np.array([[1, 2], [3, 4]])
    t = Tensor(data)
    
    # Assert correct shape
    assert t.data.shape == (2, 2), f"Expected shape (2, 2), got {t.data.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
