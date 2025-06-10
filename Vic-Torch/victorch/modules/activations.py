# ============================================
# FILE: victorch/modules/activations.py
# VERSION: v0.0.2-GODCORE-ELITE
# NAME: Activations (Softmax + ReLU)
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Core activation functions for VICTORCH models.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

import numpy as np
from ..core.tensor import Tensor

class Softmax:
    """
    Softmax activation over the last axis.
    """
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, x: Tensor) -> Tensor:
        x_stable = x.data - np.max(x.data, axis=self.axis, keepdims=True)
        exp_x = np.exp(x_stable)
        softmax_x = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
        return Tensor(softmax_x, requires_grad=x.requires_grad)

class ReLU:
    """
    ReLU activation: max(0, x)
    """
    def __call__(self, x: Tensor) -> Tensor:
        return Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
