import torch
import torch.nn as nn
from typing import Optional, Any

def apply_mlp_multiplier(model: nn.Module, multiplier: float):
    """
    Direct constant scalar multiplication across MLP layers (Section 3.2.1 of the paper):
        W'_up = c * W_up
        W'_down = (1 / c) * W_down
    
    For SwiGLU / Gated MLP architectures (e.g. LLaMA), this exactly cancels out in the forward pass:
        W_down * (1/c) * [ c * W_up(x) ] = W_down * W_up(x)
    leaving outputs identical while scaling weight norms and altering gradient optimization dynamics.
    """
    if multiplier is None or float(multiplier) == 1.0:
        return

    c = float(multiplier)
    inv_c = 1.0 / c

    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    elif hasattr(model, "layers"):
        layers = model.layers

    if not layers:
        print(f"[apply_mlp_multiplier] Warning: Could not locate layers on {type(model).__name__}")
        return

    count = 0
    with torch.no_grad():
        for i, layer in enumerate(layers):
            if not hasattr(layer, "mlp"):
                continue
            mlp = layer.mlp
            if hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
                mlp.up_proj.weight.data.mul_(c)
                mlp.down_proj.weight.data.mul_(inv_c)
                count += 1
            elif hasattr(mlp, "fc1") and hasattr(mlp, "fc2"):
                mlp.fc1.weight.data.mul_(c)
                mlp.fc2.weight.data.mul_(inv_c)
                count += 1

    print(f"[apply_mlp_multiplier] Scaled {count}/{len(layers)} MLP layers by constant factor C={c} (up *= {c}, down /= {c}).")


def apply_attn_multiplier(model: nn.Module, multiplier: float):
    """
    Attention Query-Key symmetry transformation (Section 3.2.2 of the paper):
        W'_Q = c * W_Q
        b'_Q = c * b_Q (if bias exists)
        W'_K = (1 / c) * W_K
        b'_K = (1 / c) * b_K (if bias exists)

    Because attention scores are computed via:
        A = softmax( (Q K^T) / sqrt(d_k) )
    scaling Q by c and K by (1/c) exactly cancels out in the dot product:
        (c Q) (1/c K)^T = Q K^T
    leaving attention maps and all model forward outputs strictly invariant.
    Unlike MLP scaling on non-gated architectures, this holds across all Transformer models
    (including Phi-2 with GELU and RoPE).
    """
    if multiplier is None or float(multiplier) == 1.0:
        return

    c = float(multiplier)
    inv_c = 1.0 / c

    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    elif hasattr(model, "layers"):
        layers = model.layers

    if not layers:
        print(f"[apply_attn_multiplier] Warning: Could not locate layers on {type(model).__name__}")
        return

    count = 0
    with torch.no_grad():
        for i, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
            if attn is None:
                continue
            if hasattr(attn, "q_proj") and hasattr(attn, "k_proj"):
                attn.q_proj.weight.data.mul_(c)
                if attn.q_proj.bias is not None:
                    attn.q_proj.bias.data.mul_(c)
                attn.k_proj.weight.data.mul_(inv_c)
                if attn.k_proj.bias is not None:
                    attn.k_proj.bias.data.mul_(inv_c)
                count += 1

    print(f"[apply_attn_multiplier] Scaled {count}/{len(layers)} Attention layers by constant factor C={c} (Q *= {c}, K /= {c}).")
