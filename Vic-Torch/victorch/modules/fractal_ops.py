# ============================================
# FILE: victorch/modules/fractal_ops.py
# VERSION: v1.1.1-GODCORE-ELITE
# NAME: FractalOps (SelfAttention)
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Correct clean SelfAttention Module.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

import numpy as np
from ..core.tensor import Tensor
from .layers import Dense

class SelfAttention:
    """
    Basic single-head self-attention layer (handles 2D or 3D inputs).
    """

    def __init__(self, embed_dim):
        self.query_proj = Dense(embed_dim, embed_dim)
        self.key_proj = Dense(embed_dim, embed_dim)
        self.value_proj = Dense(embed_dim, embed_dim)

    def __call__(self, x: Tensor) -> Tensor:
        if len(x.shape()) == 2:
            x = Tensor(np.expand_dims(x.data, axis=1), requires_grad=x.requires_grad)

        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        d_k = Q.shape()[-1]
        K_transposed = Tensor(K.data.transpose(0, 2, 1), requires_grad=K.requires_grad)
        scores = Q.matmul(K_transposed) / np.sqrt(d_k)
        weights = scores  # (later: apply softmax)

        out = weights.matmul(V)

        # Correct: squeeze output manually via numpy
        return Tensor(np.squeeze(out.data, axis=1), requires_grad=out.requires_grad)

    def parameters(self):
        params = []
        params.extend(self.query_proj.parameters())
        params.extend(self.key_proj.parameters())
        params.extend(self.value_proj.parameters())
        return params


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
