# ============================================
# FILE: victorch/core/loss.py
# VERSION: v0.0.1-GODCORE-ELITE
# NAME: Loss Functions (MSE, CrossEntropy)
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Core loss functions for VICTORCH training systems.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

import numpy as np
from .tensor import Tensor

class MSELoss:
    """
    Mean Squared Error Loss
    """

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred.data - target.data
        loss = np.mean(diff ** 2)
        return Tensor(loss, requires_grad=True)

class CrossEntropyLoss:
    """
    Cross-Entropy Loss for classification tasks
    """

    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        # logits: (batch_size, num_classes)
        # target: (batch_size,)
        probs = self._softmax(logits.data)
        batch_size = target.data.shape[0]
        # Select correct class probabilities
        correct_logprobs = -np.log(probs[np.arange(batch_size), target.data.astype(int)])
        loss = np.sum(correct_logprobs) / batch_size
        return Tensor(loss, requires_grad=True)

    def _softmax(self, x):
        x_stable = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
