# ============================================
# FILE: victorch/tests/test_tensor.py
# VERSION: v1.0.0-GODCORE-TESTS
# NAME: Tensor Unit Tests
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Full Tensor functional validation suite.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

import numpy as np
from victorch.core.tensor import Tensor

def assert_close(a, b, tol=1e-5):
    assert np.allclose(a.data, b.data, atol=tol), f"Mismatch: {a.data} vs {b.data}"

def test_basic_ops():
    x = Tensor([1,2,3])
    y = Tensor([4,5,6])

    assert_close(x + y, Tensor([5,7,9]))
    assert_close(x - y, Tensor([-3,-3,-3]))
    assert_close(x * y, Tensor([4,10,18]))
    assert_close(y / x, Tensor([4,2.5,2]))

def test_matmul():
    a = Tensor(np.ones((2,3)))
    b = Tensor(np.ones((3,4)))
    out = a.matmul(b)
    assert out.shape() == (2,4)

def test_mean_sum():
    t = Tensor([[1,2],[3,4]])
    assert_close(t.mean(), Tensor(2.5))
    assert_close(t.sum(), Tensor(10))

def test_min_max():
    t = Tensor([1,8,3,-2])
    assert_close(t.min(), Tensor(-2))
    assert_close(t.max(), Tensor(8))


def test_exp_log_pow():
    t = Tensor([1.0,2.0,3.0])
    assert_close(t.exp(), Tensor(np.exp([1.0,2.0,3.0])))
    assert_close(t.log(), Tensor(np.log([1.0,2.0,3.0])))
    assert_close(t.pow(2), Tensor([1.0,4.0,9.0]))

def test_softmax():
    t = Tensor([[1,2,3],[1,2,3]])
    softmax_out = t.softmax(axis=1)
    assert np.allclose(softmax_out.data.sum(axis=1), np.array([1.0,1.0]), atol=1e-5)

def test_backward():
    x = Tensor([1.0,2.0,3.0], requires_grad=True)
    y = (x * 2).sum()
    y.backward()
    assert_close(x.grad, Tensor([2.0,2.0,2.0]))

def test_zero_grad():
    x = Tensor([1.0,2.0,3.0], requires_grad=True)
    y = (x * 2).sum()
    y.backward()
    x.zero_grad()
    assert x.grad is None

# ============================================
# Test Runner
# ============================================

if __name__ == "__main__":
    test_basic_ops()
    test_matmul()
    test_mean_sum()
    test_min_max()
    test_exp_log_pow()
    test_softmax()
    test_backward()
    test_zero_grad()
    print("✅ All VictorTensor unit tests passed!")


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
