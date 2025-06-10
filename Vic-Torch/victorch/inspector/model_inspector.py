# ============================================
# FILE: victorch/inspector/model_inspector.py
# VERSION: v1.0.0-GODCORE-INSIGHT
# NAME: VictorModelInspector
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Deep dive analyzer for Victor models (param counts, layer stats, memory est).
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

import numpy as np

def count_parameters(module):
    """
    Recursively count parameters in a module.
    """
    if hasattr(module, 'parameters'):
        return sum(p.data.size for p in module.parameters())
    elif isinstance(module, list):
        return sum(count_parameters(m) for m in module)
    else:
        return 0

def summarize_model(model):
    """
    Summarizes model: total params, per-layer params, est. memory.
    """
    summary = {}
    total_params = 0
    total_memory_bytes = 0

    def walk(name, module):
        nonlocal total_params, total_memory_bytes

        if hasattr(module, 'parameters'):
            params = module.parameters()
            param_count = sum(p.data.size for p in params)
            memory = sum(p.data.nbytes for p in params)

            summary[name] = {
                "params": param_count,
                "memory_bytes": memory
            }
            total_params += param_count
            total_memory_bytes += memory

        if isinstance(module, list):
            for i, m in enumerate(module):
                walk(f"{name}[{i}]", m)

    walk(model.__class__.__name__, model)

    print("\n=== Victor Model Inspector ===")
    for layer, stats in summary.items():
        print(f"{layer}: {stats['params']} params | {stats['memory_bytes']/1024:.2f} KB")

    print("------------------------------")
    print(f"Total Parameters: {total_params}")
    print(f"Estimated Memory: {total_memory_bytes/1024/1024:.2f} MB")
    print("==============================\n")

# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    from victorch.models.victor_model import VictorTransformerModel

    model = VictorTransformerModel(
        vocab_size=32, 
        embed_dim=32, 
        num_layers=4, 
        hidden_dim=64, 
        num_classes=10
    )

    summarize_model(model)


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
