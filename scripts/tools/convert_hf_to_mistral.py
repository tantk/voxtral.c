#!/usr/bin/env python3
"""Convert HuggingFace Voxtral safetensors to Mistral-native format.

Renames tensor keys from HF convention (audio_tower.*, language_model.*)
to Mistral convention (mm_streams_embeddings.*, layers.*) so voxtral.c
can load the weights.

Usage:
    python3 convert_hf_to_mistral.py input.safetensors output.safetensors
"""

import re
import sys
from safetensors import safe_open
from safetensors.torch import save_file


ENC = "mm_streams_embeddings.embedding_module.whisper_encoder"


def hf_to_mistral(hf_key: str) -> str:
    # Encoder Conv Stem
    m = re.match(r"audio_tower\.embedder\.conv([12])\.(weight|bias)$", hf_key)
    if m:
        conv_idx = int(m.group(1)) - 1
        return f"{ENC}.conv_layers.{conv_idx}.conv.{m.group(2)}"

    # Encoder Transformer Layers
    m = re.match(r"audio_tower\.layers\.(\d+)\.(.+)$", hf_key)
    if m:
        layer = m.group(1)
        suffix = m.group(2)
        lp = f"{ENC}.transformer.layers.{layer}"
        ENC_MAP = {
            "self_attn.q_proj.weight": "attention.wq.weight",
            "self_attn.q_proj.bias":   "attention.wq.bias",
            "self_attn.k_proj.weight": "attention.wk.weight",
            "self_attn.v_proj.weight": "attention.wv.weight",
            "self_attn.v_proj.bias":   "attention.wv.bias",
            "self_attn.o_proj.weight": "attention.wo.weight",
            "self_attn.o_proj.bias":   "attention.wo.bias",
            "self_attn_layer_norm.weight": "attention_norm.weight",
            "final_layer_norm.weight": "ffn_norm.weight",
            "mlp.gate_proj.weight": "feed_forward.w1.weight",
            "mlp.down_proj.weight": "feed_forward.w2.weight",
            "mlp.down_proj.bias":   "feed_forward.w2.bias",
            "mlp.up_proj.weight":   "feed_forward.w3.weight",
        }
        if suffix in ENC_MAP:
            return f"{lp}.{ENC_MAP[suffix]}"
        raise ValueError(f"Unknown encoder layer suffix: {suffix}")

    # Encoder Final Norm
    if hf_key == "audio_tower.norm.weight":
        return f"{ENC}.transformer.norm.weight"

    # Adapter
    m = re.match(r"multi_modal_projector\.linear_([12])\.weight$", hf_key)
    if m:
        proj_idx = 0 if m.group(1) == "1" else 2
        return f"mm_streams_embeddings.embedding_module.audio_language_projection.{proj_idx}.weight"

    # Token Embeddings
    if hf_key == "language_model.model.embed_tokens.weight":
        return "mm_streams_embeddings.embedding_module.tok_embeddings.weight"

    # Decoder Transformer Layers
    m = re.match(r"language_model\.model\.layers\.(\d+)\.(.+)$", hf_key)
    if m:
        layer = m.group(1)
        suffix = m.group(2)
        lp = f"layers.{layer}"
        DEC_MAP = {
            "ada_rms_norm.linear1.weight": "ada_rms_norm_t_cond.0.weight",
            "ada_rms_norm.linear2.weight": "ada_rms_norm_t_cond.2.weight",
            "self_attn.q_proj.weight": "attention.wq.weight",
            "self_attn.k_proj.weight": "attention.wk.weight",
            "self_attn.v_proj.weight": "attention.wv.weight",
            "self_attn.o_proj.weight": "attention.wo.weight",
            "input_layernorm.weight": "attention_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "mlp.gate_proj.weight": "feed_forward.w1.weight",
            "mlp.down_proj.weight": "feed_forward.w2.weight",
            "mlp.up_proj.weight":   "feed_forward.w3.weight",
        }
        if suffix in DEC_MAP:
            return f"{lp}.{DEC_MAP[suffix]}"
        raise ValueError(f"Unknown decoder layer suffix: {suffix}")

    # Decoder Final Norm
    if hf_key == "language_model.model.norm.weight":
        return "norm.weight"

    raise ValueError(f"Unknown HF key: {hf_key}")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.safetensors output.safetensors")
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]
    print(f"Loading {src}...")

    tensors = {}
    with safe_open(src, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"Found {len(keys)} tensors, converting names...")
        for i, hf_key in enumerate(keys):
            mistral_key = hf_to_mistral(hf_key)
            tensors[mistral_key] = f.get_tensor(hf_key)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(keys)}...")

    print(f"Writing {dst} ({len(tensors)} tensors)...")
    save_file(tensors, dst)
    print("Done!")


if __name__ == "__main__":
    main()
