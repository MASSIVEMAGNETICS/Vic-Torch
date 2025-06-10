# ============================================
# FILE: victorch/modules/layers.py
# VERSION: v1.0.0-GODCORE-ELITE
# NAME: Layers (Dense, FeedForward)
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Basic layers for VICTORCH models.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

import numpy as np
from ..core.tensor import Tensor
from .activations import ReLU

class Dense:
    """
    Fully Connected Linear Layer.
    """

    def __init__(self, in_features, out_features):
        self.weight = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2. / in_features))
        self.bias = Tensor(np.zeros(out_features))

    def __call__(self, x: Tensor) -> Tensor:
        return x.matmul(self.weight) + self.bias

    def parameters(self):
        return [self.weight, self.bias]

class FeedForward:
    """
    Simple Feed Forward MLP (2 Linear layers with ReLU)
    """

    def __init__(self, embed_dim, hidden_dim):
        self.fc1 = Dense(embed_dim, hidden_dim)
        self.fc2 = Dense(hidden_dim, embed_dim)
        self.activation = ReLU()

    def __call__(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
