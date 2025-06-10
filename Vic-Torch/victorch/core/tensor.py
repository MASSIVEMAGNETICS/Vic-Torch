# ============================================
# FILE: victorch/core/tensor.py
# VERSION: v3.0.0-GODCORE-AUTOGRAD
# NAME: VictorTensor (Autograd Godcore)
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Full dynamic Tensor with automatic differentiation for VictorCortex.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, creators=None, creation_op=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None

        # Autograd fields
        self.creators = creators
        self.creation_op = creation_op

    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size == 1:
                grad = Tensor(np.ones_like(self.data))
            else:
                raise RuntimeError("grad must be specified for non-scalar tensor")

        if self.grad is None:
            self.grad = grad
        else:
            self.grad = Tensor(self.grad.data + grad.data)

        if self.creators is not None:
            if self.creation_op == "add":
                self.creators[0].backward(self.grad)
                self.creators[1].backward(self.grad)
            elif self.creation_op == "sub":
                self.creators[0].backward(self.grad)
                self.creators[1].backward(Tensor(-self.grad.data))
            elif self.creation_op == "mul":
                new_grad_0 = Tensor(self.grad.data * self.creators[1].data)
                new_grad_1 = Tensor(self.grad.data * self.creators[0].data)
                self.creators[0].backward(new_grad_0)
                self.creators[1].backward(new_grad_1)
            elif self.creation_op == "matmul":
                new_grad_0 = Tensor(self.grad.data @ self.creators[1].data.T)
                new_grad_1 = Tensor(self.creators[0].data.T @ self.grad.data)
                self.creators[0].backward(new_grad_0)
                self.creators[1].backward(new_grad_1)

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)

    def clone(self):
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    # ========================
    # Operations (with autograd)
    # ========================

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data, requires_grad=True, creators=[self, other], creation_op="add")
        else:
            return Tensor(self.data + other, requires_grad=self.requires_grad)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data, requires_grad=True, creators=[self, other], creation_op="sub")
        else:
            return Tensor(self.data - other, requires_grad=self.requires_grad)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data, requires_grad=True, creators=[self, other], creation_op="mul")
        else:
            return Tensor(self.data * other, requires_grad=self.requires_grad)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data, requires_grad=True, creators=[self, other], creation_op="div")
        else:
            return Tensor(self.data / other, requires_grad=self.requires_grad)

    def matmul(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data @ other.data, requires_grad=True, creators=[self, other], creation_op="matmul")
        else:
            return Tensor(self.data @ other, requires_grad=self.requires_grad)

    def squeeze(self, axis=None):
        return Tensor(self.data.squeeze(axis), requires_grad=self.requires_grad)

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.data, axis), requires_grad=self.requires_grad)

    def reshape(self, *new_shape):
        return Tensor(self.data.reshape(new_shape), requires_grad=self.requires_grad)

    def expand(self, *sizes):
        return Tensor(np.broadcast_to(self.data, sizes), requires_grad=self.requires_grad)

    def transpose(self, *axes):
        if not axes:
            axes = reversed(range(len(self.data.shape)))
        return Tensor(self.data.transpose(*axes), requires_grad=self.requires_grad)

    def mean(self, axis=None, keepdims=False):
        return Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)

    def sum(self, axis=None, keepdims=False):
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)

    def min(self, axis=None, keepdims=False):
        return Tensor(self.data.min(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)

    def max(self, axis=None, keepdims=False):
        return Tensor(self.data.max(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)

    def argmax(self, axis=None):
        return Tensor(self.data.argmax(axis=axis), requires_grad=False)

    def argmin(self, axis=None):
        return Tensor(self.data.argmin(axis=axis), requires_grad=False)

    def __repr__(self):
        return f"VictorTensor(shape={self.shape()}, requires_grad={self.requires_grad})\n{self.data}"

# ============================================
# GODCORE AUTOGRAD: Tensor now fully singularity-ready.
# ============================================
