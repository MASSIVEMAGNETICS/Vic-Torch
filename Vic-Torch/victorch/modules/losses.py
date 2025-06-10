# ============================================
# FILE: victorch/modules/losses.py
# VERSION: v0.0.1-GODCORE-ELITE
# NAME: VICTORCH Loss Functions (MSELoss, CrossEntropyLoss)
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Core loss functions for training models in VICTORCH.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

import numpy as np
from ..core.tensor import Tensor

class MSELoss:
    """
    Mean Squared Error Loss.
    """
    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        loss = (prediction - target) * (prediction - target)
        return loss.mean()


class CrossEntropyLoss:
    """
    Cross Entropy Loss for classification tasks.
    Assumes inputs are raw logits (not softmaxed yet).
    """
    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        # Stabilize logits for numerical safety
        exp_preds = np.exp(prediction.data - np.max(prediction.data, axis=1, keepdims=True))
        probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        
        # Compute log-likelihood
        batch_size = prediction.data.shape[0]
        log_likelihood = -np.log(probs[range(batch_size), target.data.astype(int)])
        loss = np.sum(log_likelihood) / batch_size
        
        return Tensor(loss, requires_grad=prediction.requires_grad)


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
