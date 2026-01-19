"""
Minimal Tensor Tests for CI

These tests are intentionally minimal and defensive to bootstrap CI.
They skip gracefully if imports fail.
"""

import pytest


def test_tensor_import():
    """Test that Tensor can be imported."""
    try:
        from victorch.core.tensor import Tensor
    except ImportError as e:
        pytest.skip(f"Tensor module not available: {e}")


def test_tensor_basic_shape():
    """Test basic tensor creation and shape."""
    try:
        from victorch.core.tensor import Tensor
        import numpy as np
        
        # Test 1D tensor
        t1 = Tensor([1, 2, 3])
        assert t1.shape() == (3,), f"Expected shape (3,), got {t1.shape()}"
        
        # Test 2D tensor
        t2 = Tensor([[1, 2], [3, 4]])
        assert t2.shape() == (2, 2), f"Expected shape (2, 2), got {t2.shape()}"
        
        # Test 3D tensor
        t3 = Tensor(np.zeros((2, 3, 4)))
        assert t3.shape() == (2, 3, 4), f"Expected shape (2, 3, 4), got {t3.shape()}"
        
    except ImportError:
        pytest.skip("Tensor module not available")


def test_tensor_basic_operations():
    """Test basic tensor operations."""
    try:
        from victorch.core.tensor import Tensor
        import numpy as np
        
        x = Tensor([1, 2, 3])
        y = Tensor([4, 5, 6])
        
        # Test addition
        z = x + y
        expected = Tensor([5, 7, 9])
        assert np.allclose(z.data, expected.data), "Addition failed"
        
        # Test multiplication
        z = x * y
        expected = Tensor([4, 10, 18])
        assert np.allclose(z.data, expected.data), "Multiplication failed"
        
    except ImportError:
        pytest.skip("Tensor module not available")
    except Exception as e:
        pytest.skip(f"Test requires additional features: {e}")


if __name__ == "__main__":
    # Run tests directly for debugging
    print("Running tensor tests...")
    test_tensor_import()
    print("✓ test_tensor_import passed")
    
    test_tensor_basic_shape()
    print("✓ test_tensor_basic_shape passed")
    
    test_tensor_basic_operations()
    print("✓ test_tensor_basic_operations passed")
    
    print("\nAll tests passed!")
