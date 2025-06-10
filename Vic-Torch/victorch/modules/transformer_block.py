# ============================================
# FILE: victorch/modules/transformer_block.py
# VERSION: v1.0.0-GODCORE-ELITE
# NAME: Transformer Encoder Block (Self-Attention + FeedForward)
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Core Transformer Block module for VICTORCH architectures.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

import numpy as np
from ..core.tensor import Tensor
from .fractal_ops import SelfAttention
from .layers import FeedForward
from .activations import ReLU

class LayerNorm:
    """
    Simple Layer Normalization across last dimension.
    """

    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = Tensor(np.ones(dim))
        self.beta = Tensor(np.zeros(dim))

    def __call__(self, x: Tensor) -> Tensor:
        mean = x.data.mean(axis=-1, keepdims=True)
        variance = ((x.data - mean) ** 2).mean(axis=-1, keepdims=True)
        x_norm = (x.data - mean) / np.sqrt(variance + self.eps)
        out = self.gamma.data * x_norm + self.beta.data
        return Tensor(out, requires_grad=x.requires_grad)

class TransformerBlock:
    """
    Transformer Encoder Block:
    - Self-Attention
    - Add & LayerNorm
    - FeedForward
    - Add & LayerNorm
    """

    def __init__(self, embed_dim, hidden_dim):
        self.attention = SelfAttention(embed_dim)
        self.norm1 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, hidden_dim)
        self.norm2 = LayerNorm(embed_dim)

    def __call__(self, x: Tensor) -> Tensor:
        # Self-attention + residual + norm
        attn_out = self.attention(x)
        x = Tensor(x.data + attn_out.data, requires_grad=x.requires_grad)
        x = self.norm1(x)

        # FeedForward + residual + norm
        ff_out = self.ff(x)
        x = Tensor(x.data + ff_out.data, requires_grad=x.requires_grad)
        x = self.norm2(x)

        return x

    def parameters(self):
        params = []
        params.extend(self.attention.parameters())
        params.extend(self.ff.parameters())
        return params


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
