# ============================================
# FILE: victorch/modules/multihead_attention.py
# VERSION: v0.0.2-GODCORE-ELITE
# NAME: MultiHeadAttention (Fixed Tensor Handling)
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Multi-head attention mechanism for VICTORCH models (batch + grad safe).
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

import numpy as np
from ..core.tensor import Tensor
from ..modules.layers import Dense
from ..modules.activations import Softmax

class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.
    Splits input into multiple heads, applies attention independently, and concatenates outputs.
    """

    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0, "Embed dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V
        self.query_proj = Dense(embed_dim, embed_dim)
        self.key_proj = Dense(embed_dim, embed_dim)
        self.value_proj = Dense(embed_dim, embed_dim)

        # Final output projection
        self.out_proj = Dense(embed_dim, embed_dim)

        self.softmax = Softmax(axis=-1)

    def __call__(self, x: Tensor) -> Tensor:
        batch_size, seq_len, embed_dim = x.shape()

        # Project inputs
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        # Split into heads
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Scaled dot-product attention
        K_transposed = Tensor(np.transpose(K.data, (0, 1, 3, 2)), requires_grad=K.requires_grad)
        scores = Q.matmul(K_transposed) / np.sqrt(self.head_dim)
        attention_weights = self.softmax(scores)
        attention_output = attention_weights.matmul(V)

        # Merge heads
        output = self._merge_heads(attention_output)

        # Final output projection
        output = self.out_proj(output)

        return output

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        Split the last dimension into (num_heads, head_dim)
        """
        batch_size, seq_len, embed_dim = x.shape()
        reshaped = x.data.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        transposed = np.transpose(reshaped, (0, 2, 1, 3))  # (batch, heads, seq_len, head_dim)
        return Tensor(transposed, requires_grad=x.requires_grad)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """
        Merge multiple heads back into the original embed_dim
        """
        batch_size, num_heads, seq_len, head_dim = x.shape()
        transposed = np.transpose(x.data, (0, 2, 1, 3))  # (batch, seq_len, heads, head_dim)
        merged = transposed.reshape(batch_size, seq_len, num_heads * head_dim)
        return Tensor(merged, requires_grad=x.requires_grad)


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
