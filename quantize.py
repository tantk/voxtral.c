#!/usr/bin/env python3
"""
quantize.py - Voxtral 4B Realtime weight quantizer (GPU-accelerated)

Reads safetensors BF16 weights and produces a pre-quantized VQF file.
Uses CUDA GPU for quantization when available, CPU as fallback.
Run once per model; output is reused on every app launch.

Usage:
    python quantize.py <model_dir> <output.vqf> [--type Q4_K|Q8_0|Q4_0]
"""

import argparse
import struct
import sys
import os
import json
import time
import numpy as np
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("Error: safetensors required. pip install safetensors")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("Error: PyTorch required. pip install torch")
    sys.exit(1)

# Device selection: GPU first, CPU fallback
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

# VQF format constants (must match voxtral_quant.h)
VQF_MAGIC = 0x31465156   # "VQF1"
VQF_VERSION = 1
VQF_TYPE_F32  = 0
VQF_TYPE_BF16 = 2
VQF_TYPE_Q8_0 = 8
VQF_TYPE_Q4_0 = 10
VQF_TYPE_Q4_K = 12

TYPE_NAMES = {
    "Q8_0": VQF_TYPE_Q8_0,
    "Q4_0": VQF_TYPE_Q4_0,
    "Q4_K": VQF_TYPE_Q4_K,
}


# ---------------------------------------------------------------------------
# Quantization functions (GPU-first, CPU fallback via torch)
# ---------------------------------------------------------------------------

def quantize_q8_0(t):
    """Q8_0: 32 values/block. scale(f32,4B) + int8[32](32B) = 36 bytes/block."""
    values = t.flatten().to(device=device, dtype=torch.float32)
    n = values.numel()
    pad = (32 - n % 32) % 32
    if pad:
        values = torch.nn.functional.pad(values, (0, pad))

    n_blocks = values.numel() // 32
    blocks = values.reshape(n_blocks, 32)

    amax = blocks.abs().amax(dim=1)
    scales = torch.where(amax > 0, amax / 127.0, amax.new_zeros(()))
    inv_scales = torch.where(scales > 0, 1.0 / scales, scales.new_zeros(()))
    quants = (blocks * inv_scales[:, None]).round().clamp(-127, 127).to(torch.int8)

    s = scales.cpu().numpy()
    q = quants.cpu().numpy()
    out = np.empty((n_blocks, 36), dtype=np.uint8)
    out[:, :4] = s.view(np.uint8).reshape(-1, 4)
    out[:, 4:] = q.view(np.uint8)
    return out.tobytes()


def quantize_q4_0(t):
    """Q4_0: 32 values/block. scale(f32,4B) + nibbles[16](16B) = 20 bytes/block."""
    values = t.flatten().to(device=device, dtype=torch.float32)
    n = values.numel()
    pad = (32 - n % 32) % 32
    if pad:
        values = torch.nn.functional.pad(values, (0, pad))

    n_blocks = values.numel() // 32
    blocks = values.reshape(n_blocks, 32)

    amax = blocks.abs().amax(dim=1)
    scales = torch.where(amax > 0, amax / 7.0, amax.new_zeros(()))
    inv_scales = torch.where(scales > 0, 1.0 / scales, scales.new_zeros(()))
    quants = (blocks * inv_scales[:, None] + 8).round().clamp(0, 15).to(torch.uint8)

    lo = quants[:, 0::2]
    hi = quants[:, 1::2]
    nibs = (lo & 0xF) | ((hi & 0xF) << 4)

    s = scales.cpu().numpy()
    n_np = nibs.cpu().numpy()
    out = np.empty((n_blocks, 20), dtype=np.uint8)
    out[:, :4] = s.view(np.uint8).reshape(-1, 4)
    out[:, 4:] = n_np
    return out.tobytes()


def quantize_q4_k(t):
    """Q4_K: 256 values/super-block (8 sub-blocks of 32).
    super_scale(f32,4B) + super_min(f32,4B) + packed_scales[12B] + nibs[128B] = 148 bytes."""
    values = t.flatten().to(device=device, dtype=torch.float32)
    n = values.numel()
    pad = (256 - n % 256) % 256
    if pad:
        values = torch.nn.functional.pad(values, (0, pad))

    n_blocks = values.numel() // 256
    subs = values.reshape(n_blocks, 8, 32)

    sub_min = subs.amin(dim=2)       # [n_blocks, 8]
    sub_max = subs.amax(dim=2)
    sub_range = sub_max - sub_min
    sub_scales = torch.where(sub_range > 0, sub_range / 15.0, sub_range.new_zeros(()))

    max_scale = sub_scales.abs().amax(dim=1)   # [n_blocks]
    max_min = sub_min.abs().amax(dim=1)
    super_scale = torch.where(max_scale > 0, max_scale / 63.0, max_scale.new_zeros(()))
    super_min = torch.where(max_min > 0, max_min / 63.0, max_min.new_zeros(()))

    inv_ss = torch.where(super_scale > 0, 1.0 / super_scale, super_scale.new_zeros(()))
    inv_sm = torch.where(super_min > 0, 1.0 / super_min, super_min.new_zeros(()))
    q_scales = (sub_scales * inv_ss[:, None]).round().clamp(0, 63).to(torch.uint8)
    q_mins = (sub_min.abs() * inv_sm[:, None]).round().clamp(0, 63).to(torch.uint8)

    eff_scales = q_scales.float() * super_scale[:, None]
    inv_eff = torch.where(eff_scales > 0, 1.0 / eff_scales, eff_scales.new_zeros(()))

    quants = ((subs - sub_min[:, :, None]) * inv_eff[:, :, None]).round().clamp(0, 15).to(torch.uint8)
    lo = quants[:, :, 0::2]
    hi = quants[:, :, 1::2]
    nibs = ((lo & 0xF) | ((hi & 0xF) << 4)).reshape(n_blocks, 128)

    # Move to CPU for final byte packing
    ss_np = super_scale.cpu().numpy()
    sm_np = super_min.cpu().numpy()
    qs_np = q_scales.cpu().numpy()
    qm_np = q_mins.cpu().numpy()
    nibs_np = nibs.cpu().numpy()

    # Pack 6-bit scales/mins into 12 bytes per block
    packed = np.zeros((n_blocks, 12), dtype=np.uint8)
    for i in range(4):
        s0, s1 = qs_np[:, i*2], qs_np[:, i*2+1]
        m0, m1 = qm_np[:, i*2], qm_np[:, i*2+1]
        packed[:, i*3]     = (s0 & 0x3F) | ((s1 & 0x03) << 6)
        packed[:, i*3 + 1] = ((s1 >> 2) & 0x0F) | ((m0 & 0x0F) << 4)
        packed[:, i*3 + 2] = ((m0 >> 4) & 0x03) | ((m1 & 0x3F) << 2)

    out = np.empty((n_blocks, 148), dtype=np.uint8)
    out[:, 0:4] = ss_np.view(np.uint8).reshape(-1, 4)
    out[:, 4:8] = sm_np.view(np.uint8).reshape(-1, 4)
    out[:, 8:20] = packed
    out[:, 20:148] = nibs_np
    return out.tobytes()


QUANTIZE_FN = {
    VQF_TYPE_Q8_0: quantize_q8_0,
    VQF_TYPE_Q4_0: quantize_q4_0,
    VQF_TYPE_Q4_K: quantize_q4_k,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def should_quantize(name):
    """Check if a tensor should be quantized (decoder/encoder linear weights only)."""
    if not name.endswith('.weight'):
        return False
    parts = name.split('.')
    if len(parts) < 3:
        return False
    weight_kind = parts[-2]
    if weight_kind not in ('wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3'):
        return False
    return any(p == 'layers' for p in parts)


def vqf_tensor_bytes(qtype, numel):
    """Compute output byte size for a tensor given quant type and element count."""
    if qtype == VQF_TYPE_F32:
        return numel * 4
    elif qtype == VQF_TYPE_BF16:
        return numel * 2
    elif qtype == VQF_TYPE_Q8_0:
        return ((numel + 31) // 32) * 36
    elif qtype == VQF_TYPE_Q4_0:
        return ((numel + 31) // 32) * 20
    elif qtype == VQF_TYPE_Q4_K:
        return ((numel + 255) // 256) * 148
    return numel * 2


def dtype_to_vqf(dtype_str):
    """Map safetensors dtype string to VQF type constant."""
    if dtype_str in ('F32', 'float32'):
        return VQF_TYPE_F32
    return VQF_TYPE_BF16


def dtype_element_size(dtype_str):
    """Bytes per element for a safetensors dtype."""
    if dtype_str in ('F32', 'float32'):
        return 4
    return 2  # BF16, F16


def tensor_raw_bytes(t):
    """Get raw bytes from a PyTorch tensor, preserving original dtype."""
    t = t.contiguous()
    if t.dtype == torch.float32:
        return t.numpy(force=True).tobytes()
    # BF16/F16: reinterpret as int16 (same 2-byte width) to get just this
    # tensor's bytes. Using untyped_storage() directly would copy the entire
    # mmap backing the safetensors file (~8 GB), not just this tensor's slice.
    return t.view(torch.int16).numpy().tobytes()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Voxtral 4B weight quantizer (GPU-accelerated)")
    parser.add_argument("model_dir", help="Directory containing consolidated.safetensors")
    parser.add_argument("output", help="Output VQF file path")
    parser.add_argument("--type", choices=list(TYPE_NAMES.keys()), default="Q4_K",
                        help="Quantization type (default: Q4_K)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    qtype = TYPE_NAMES[args.type]
    model_path = Path(args.model_dir) / "consolidated.safetensors"
    if not model_path.exists():
        print(f"Error: {model_path} not found")
        sys.exit(1)

    if USE_CUDA:
        gpu_name = torch.cuda.get_device_name()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: CUDA ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        print("Device: CPU (no CUDA available — will be slower)")
    print(f"Model:  {model_path}")
    print(f"Type:   {args.type}")

    # --- Pass 1: parse safetensors header for metadata (no data loading) ---
    with open(model_path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size))
    header.pop('__metadata__', None)

    tensor_names = sorted(header.keys())
    quantize_set = {n for n in tensor_names if should_quantize(n)}

    n_quant = len(quantize_set)
    n_keep = len(tensor_names) - n_quant
    print(f"\nTensors: {n_quant} to quantize, {n_keep} to keep")

    # Build descriptors and compute sizes/offsets from metadata only
    descriptors = []
    current_offset = 0
    total_original = 0
    total_quantized = 0

    for name in tensor_names:
        meta = header[name]
        dtype_str = meta['dtype']
        shape = meta['shape']
        numel = 1
        for d in shape:
            numel *= d
        ndim = len(shape)

        orig_bytes = numel * dtype_element_size(dtype_str)
        total_original += orig_bytes

        if name in quantize_set:
            q_bytes = vqf_tensor_bytes(qtype, numel)
            tensor_qtype = qtype
        else:
            q_bytes = orig_bytes
            tensor_qtype = dtype_to_vqf(dtype_str)

        total_quantized += q_bytes

        descriptors.append({
            'name': name,
            'qtype': tensor_qtype,
            'ndim': ndim,
            'shape': shape + [0] * (4 - ndim),
            'data_offset': current_offset,
            'data_size': q_bytes,
        })
        current_offset += q_bytes

    print(f"Original:   {total_original / 1e9:.2f} GB")
    print(f"Quantized:  {total_quantized / 1e9:.2f} GB")
    print(f"Ratio:      {total_original / total_quantized:.1f}x\n")

    # VQF header layout
    hdr_bytes = 4 + 4 + 4 + 4 + 8  # magic + version + qtype + n_tensors + data_offset
    desc_bytes = sum(2 + len(d['name'].encode('utf-8')) + 4 + 4 + 4*8 + 8 + 8 for d in descriptors)
    data_start = ((hdr_bytes + desc_bytes + 63) // 64) * 64

    # Open safetensors for lazy tensor loading
    sf = safe_open(str(model_path), framework="pt")

    # --- Pass 2: write VQF file, streaming one tensor at a time ---
    t_start = time.time()

    with open(args.output, 'wb') as f:
        # Header
        f.write(struct.pack('<I', VQF_MAGIC))
        f.write(struct.pack('<I', VQF_VERSION))
        f.write(struct.pack('<I', qtype))
        f.write(struct.pack('<I', len(descriptors)))
        f.write(struct.pack('<Q', data_start))

        # Tensor descriptors
        for d in descriptors:
            nb = d['name'].encode('utf-8')
            f.write(struct.pack('<H', len(nb)))
            f.write(nb)
            f.write(struct.pack('<I', d['qtype']))
            f.write(struct.pack('<I', d['ndim']))
            for i in range(4):
                f.write(struct.pack('<q', d['shape'][i]))
            f.write(struct.pack('<Q', d['data_offset']))
            f.write(struct.pack('<Q', d['data_size']))

        # Pad to data_start alignment
        pos = f.tell()
        if pos < data_start:
            f.write(b'\x00' * (data_start - pos))

        # Stream tensor data: quantize on GPU, write, free — one at a time
        n_done = 0
        n_total = len(tensor_names)
        bytes_written = 0
        for idx, name in enumerate(tensor_names):
            t = sf.get_tensor(name)

            if name in quantize_set:
                tq = time.time()
                qdata = QUANTIZE_FN[qtype](t)
                dt = time.time() - tq
                n_done += 1
                sz = len(qdata)
                print(f"  Q [{n_done}/{n_quant}] {name} {list(t.shape)} -> {args.type} ({sz/1e6:.1f} MB, {dt:.2f}s)")
                f.write(qdata)
                bytes_written += sz
                del qdata
            else:
                tq = time.time()
                raw = tensor_raw_bytes(t)
                dt = time.time() - tq
                sz = len(raw)
                print(f"  K [{idx+1}/{n_total}] {name} {list(t.shape)} ({sz/1e6:.1f} MB, {dt:.2f}s)")
                f.write(raw)
                bytes_written += sz
                del raw

            del t
            sys.stdout.flush()
            f.flush()

    elapsed = time.time() - t_start
    final_size = os.path.getsize(args.output)
    print(f"\nDone! {final_size / 1e9:.2f} GB written in {elapsed:.1f}s")
    if USE_CUDA:
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
