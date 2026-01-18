"""
Minimal test for MemoryBank

Tests basic MemoryBank functionality with defensive imports.
"""

import pytest


def test_memory_bank_basic():
    """Test basic MemoryBank creation and storage."""
    try:
        from Vic_Torch.victorch.memory.memory_bank import MemoryBank
    except ImportError:
        # Try alternative import path
        try:
            from Vic_Torch.victorch.memory.godcore_memory_bank import GodcoreMemoryBank as MemoryBank
        except ImportError:
            pytest.skip("MemoryBank not available")
    
    # Create a capacity-limited MemoryBank
    capacity = 5
    bank = MemoryBank(capacity=capacity)
    
    # Assert bank was created
    assert bank is not None, "MemoryBank should be created"
    
    # Store items
    bank.store("item1")
    bank.store("item2")
    bank.store("item3")
    
    # Assert len() behavior
    assert len(bank.buffer) == 3, f"Expected 3 items, got {len(bank.buffer)}"


def test_memory_bank_capacity():
    """Test MemoryBank capacity limits."""
    try:
        from Vic_Torch.victorch.memory.memory_bank import MemoryBank
    except ImportError:
        try:
            from Vic_Torch.victorch.memory.godcore_memory_bank import GodcoreMemoryBank as MemoryBank
        except ImportError:
            pytest.skip("MemoryBank not available")
    
    # Create a small capacity bank
    capacity = 3
    bank = MemoryBank(capacity=capacity)
    
    # Store more items than capacity
    for i in range(5):
        bank.store(f"item{i}")
    
    # Assert capacity is respected
    assert len(bank.buffer) <= capacity, f"Buffer size {len(bank.buffer)} exceeds capacity {capacity}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
