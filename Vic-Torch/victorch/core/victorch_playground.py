# ============================================
# FILE: victorch_playground.py
# VERSION: v0.1.0-GODCORE-ELITE
# NAME: VICTORCH Playground
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Modular Tensor + Ops + Autograd system in one file for battle-testing.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

import numpy as np

# =====================
# AUTOGRAD CORE
# =====================

class Function:
    """
    Base class for all differentiable operations.
    """
    def __init__(self, *parents):
        self.parents = parents

    def backward(self, grad_output):
        raise NotImplementedError


class Add(Function):
    def backward(self, grad_output):
        return grad_output, grad_output  # dL/da = 1, dL/db = 1


class Mul(Function):
    def backward(self, grad_output):
        a, b = self.parents
        return grad_output * b.data, grad_output * a.data

# =====================
# TENSOR CORE
# =====================

class Tensor:
    """
    Core Tensor object for Victorch.
    Lightweight wrapper over numpy arrays with optional autograd tracking.
    """

    def __init__(self, data, requires_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self.creator = None

    def set_creator(self, creator):
        self.creator = creator
        if self.requires_grad:
            for parent in creator.parents:
                parent.requires_grad = True

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    # =====================
    # Arithmetic Operations
    # =====================

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        out.set_creator(Add(self, other))
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        # (Subtraction autograd can be improved later)
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out.set_creator(Mul(self, other))
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
        # (Division autograd later — inverse chain rule)
        return out

    def matmul(self, other):
        other = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data @ other, requires_grad=self.requires_grad)

    # =====================
    # Reduction Operations
    # =====================

    def sum(self):
        return Tensor(self.data.sum(), requires_grad=self.requires_grad)

    def mean(self):
        return Tensor(self.data.mean(), requires_grad=self.requires_grad)

    # =====================
    # Structural Operations
    # =====================

    def shape(self):
        return self.data.shape

    def reshape(self, *shape):
        return Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad)

    def transpose(self, *axes):
        return Tensor(self.data.transpose(*axes), requires_grad=self.requires_grad)

    # =====================
    # Autograd - Backward
    # =====================

    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on tensor without requires_grad=True.")

        if grad is None:
            grad = np.ones_like(self.data)  # Default to dL/dout = 1

        self.grad = grad

        if self.creator is not None:
            grads = self.creator.backward(grad)
            if len(self.creator.parents) == 1:
                grads = [grads]
            for parent, grad_parent in zip(self.creator.parents, grads):
                parent.backward(grad_parent)

# =====================
# OPS MODULE
# =====================

def add(a, b):
    return a + b

def sub(a, b):
    return a - b

def mul(a, b):
    return a * b

def div(a, b):
    return a / b

def matmul(a, b):
    return a.matmul(b)

def sum(tensor):
    return tensor.sum()

def mean(tensor):
    return tensor.mean()

# =====================
# TESTING BLOCK
# =====================

if __name__ == "__main__":
    print("=== VICTORCH GODCORE TEST START ===\n")

    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)

    print(f"a: {a}")
    print(f"b: {b}")

    c = mul(a, b)  # a * b
    d = add(c, b)  # (a * b) + b

    print(f"d (forward result): {d.data}")

    d.backward()

    print(f"a.grad (should be b.data): {a.grad}")
    print(f"b.grad (should be a.data + 1): {b.grad}")

    print("\n=== VICTORCH GODCORE TEST END ===")


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
