# ============================================
# FILE: victorch/utils/model_summary.py
# VERSION: v1.0.0-GODCORE-VIEW
# NAME: VictorModelSummary
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Instant layer/param summary for any VICTORCH model.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

def summarize(model):
    print("\n============================================")
    print(f"Model Summary: {model.__class__.__name__}")
    print("============================================")
    total_params = 0

    # Embedding
    if hasattr(model, 'embedding'):
        embed_params = model.embedding.weight_size()
        print(f"Embedding Layer: {embed_params} params")
        total_params += embed_params

    # Transformer blocks
    if hasattr(model, 'transformer_blocks'):
        for i, block in enumerate(model.transformer_blocks):
            block_params = block.count_parameters()
            print(f"TransformerBlock-{i+1}: {block_params} params")
            total_params += block_params

    # Output layer
    if hasattr(model, 'output_layer'):
        output_params = model.output_layer.weight_size()
        print(f"Output Layer: {output_params} params")
        total_params += output_params

    print("============================================")
    print(f"Total Parameters: {total_params}")
    print("============================================\n")

# ============================================
# Extend Dense + TransformerBlock classes to support parameter counting
# ============================================

def patch_layers():
    from victorch.modules.layers import Dense
    from victorch.modules.transformer_block import TransformerBlock

    def dense_weight_size(self):
        return self.weights.data.size + self.bias.data.size

    def transformer_count_parameters(self):
        return self.attention.count_parameters() + self.ff.count_parameters()

    Dense.weight_size = dense_weight_size
    TransformerBlock.count_parameters = transformer_count_parameters

# ============================================
# GODCORE UPGRADE: Layers and blocks now summarizable.
# ============================================

if __name__ == "__main__":
    from victorch.models.victor_model import VictorTransformerModel
    patch_layers()
    model = VictorTransformerModel(vocab_size=32, embed_dim=32, num_layers=4, hidden_dim=64, num_classes=10)
    summarize(model)


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
