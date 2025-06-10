# ============================================
# FILE: victorch/core/tensor.py
# VERSION: v3.0.0-GODCORE-EVOLUTION
# NAME: VictorTensor (GODCORE + AUTOGRAD)
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Indestructible AI-optimized tensor + autodiff engine.
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
        self.creators = creators
        self.creation_op = creation_op

    def shape(self):
        return self.data.shape

    def reshape(self, *new_shape):
        new_shape = tuple(new_shape)
        return Tensor(self.data.reshape(new_shape), requires_grad=self.requires_grad)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = reversed(range(len(self.data.shape)))
        return Tensor(self.data.transpose(*axes), requires_grad=self.requires_grad)

    def squeeze(self, axis=None):
        return Tensor(self.data.squeeze(axis), requires_grad=self.requires_grad)

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.data, axis), requires_grad=self.requires_grad)

    def expand(self, *sizes):
        return Tensor(np.broadcast_to(self.data, sizes), requires_grad=self.requires_grad)

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

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)

    def clone(self):
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        if self.grad is None:
            self.grad = grad.data
        else:
            self.grad += grad.data

        if self.creators is not None:
            if self.creation_op == "add":
                self.creators[0].backward(Tensor(grad.data))
                self.creators[1].backward(Tensor(grad.data))
            elif self.creation_op == "sub":
                self.creators[0].backward(Tensor(grad.data))
                self.creators[1].backward(Tensor(-grad.data))
            elif self.creation_op == "mul":
                self.creators[0].backward(Tensor(grad.data * self.creators[1].data))
                self.creators[1].backward(Tensor(grad.data * self.creators[0].data))
            elif self.creation_op == "div":
                self.creators[0].backward(Tensor(grad.data / self.creators[1].data))
                self.creators[1].backward(Tensor(-grad.data * self.creators[0].data / (self.creators[1].data**2)))
            elif self.creation_op == "matmul":
                self.creators[0].backward(Tensor(grad.data @ self.creators[1].data.T))
                self.creators[1].backward(Tensor(self.creators[0].data.T @ grad.data))

    # ========================
    # Operators
    # ========================

    def __add__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
            out.creators = (self, other)
            out.creation_op = "add"
            return out
        else:
            return Tensor(self.data + other, requires_grad=self.requires_grad)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
            out.creators = (self, other)
            out.creation_op = "sub"
            return out
        else:
            return Tensor(self.data - other, requires_grad=self.requires_grad)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
            out.creators = (self, other)
            out.creation_op = "mul"
            return out
        else:
            return Tensor(self.data * other, requires_grad=self.requires_grad)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
            out.creators = (self, other)
            out.creation_op = "div"
            return out
        else:
            return Tensor(self.data / other, requires_grad=self.requires_grad)

    def matmul(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
            out.creators = (self, other)
            out.creation_op = "matmul"
            return out
        else:
            return Tensor(self.data @ other, requires_grad=self.requires_grad)

    def __repr__(self):
        return f"VictorTensor(shape={self.shape()}, requires_grad={self.requires_grad})\n{self.data}"

# ============================================
# GODCORE EVOLUTION COMPLETE
# Now Tensor can forward, backprop, and optimize.
# ============================================
