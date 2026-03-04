/*
 * voxtral_quant.h - Quantization types and VQF container format
 *
 * Defines GGML-compatible block quantization structures and the VQF
 * (Voxtral Quantized Format) container used by quantize.py and the runtime.
 *
 * Block formats match ggml-quants.h exactly so the quantization math
 * from llama.cpp is directly reusable.
 */

#ifndef VOXTRAL_QUANT_H
#define VOXTRAL_QUANT_H

#include <stdint.h>
#include <stddef.h>

/* ========================================================================
 * Quant type identifiers (match GGML type IDs for familiarity)
 * ======================================================================== */

#define VQF_TYPE_F32   0
#define VQF_TYPE_F16   1
#define VQF_TYPE_BF16  2
#define VQF_TYPE_Q8_0  8
#define VQF_TYPE_Q4_0  10
#define VQF_TYPE_Q4_K  12

/* ========================================================================
 * Block structures (must be packed, match GGML layout exactly)
 * ======================================================================== */

/* Q8_0: 32 values per block, 34 bytes
 * Dequant: w[i] = scale * quants[i] */
#pragma pack(push, 1)
typedef struct {
    float scale;           /* 4 bytes: block scale (f32 for simplicity) */
    int8_t quants[32];     /* 32 bytes: quantized values */
} vqf_block_q8_0;          /* 36 bytes total */
#pragma pack(pop)

#define VQF_Q8_0_BLOCK_SIZE  32
#define VQF_Q8_0_BLOCK_BYTES sizeof(vqf_block_q8_0)  /* 36 */

/* Q4_0: 32 values per block, 18 bytes
 * Dequant: w[i] = scale * (nibble[i] - 8) */
#pragma pack(push, 1)
typedef struct {
    float scale;           /* 4 bytes: block scale */
    uint8_t nibs[16];     /* 16 bytes: 32 nibbles packed (lo=even, hi=odd) */
} vqf_block_q4_0;          /* 20 bytes total */
#pragma pack(pop)

#define VQF_Q4_0_BLOCK_SIZE  32
#define VQF_Q4_0_BLOCK_BYTES sizeof(vqf_block_q4_0)  /* 20 */

/* Q4_K: 256 values per super-block
 * 8 sub-blocks of 32 values, each with 6-bit scale + 6-bit min
 * Dequant: w[i] = sub_scale[sb] * nibble[i] + sub_min[sb]
 *          where sub_scale and sub_min are derived from packed scales/mins */
#pragma pack(push, 1)
typedef struct {
    float super_scale;     /* 4 bytes: super-block scale for scales */
    float super_min;       /* 4 bytes: super-block scale for mins */
    uint8_t scales[12];    /* 12 bytes: packed 6-bit sub-block scales+mins */
    uint8_t nibs[128];     /* 128 bytes: 256 nibbles packed */
} vqf_block_q4_k;          /* 148 bytes total */
#pragma pack(pop)

#define VQF_Q4_K_BLOCK_SIZE  256
#define VQF_Q4_K_BLOCK_BYTES sizeof(vqf_block_q4_k)  /* 148 */

/* ========================================================================
 * VQF Container Format
 *
 * Layout:
 *   [VQF header]
 *   [Tensor descriptor 0]
 *   [Tensor descriptor 1]
 *   ...
 *   [Tensor descriptor N-1]
 *   [Tensor data ...]  (aligned to 64 bytes from file start)
 * ======================================================================== */

#define VQF_MAGIC 0x31465156  /* "VQF1" little-endian */
#define VQF_VERSION 1

typedef struct {
    uint32_t magic;        /* VQF_MAGIC */
    uint32_t version;      /* VQF_VERSION */
    uint32_t default_qtype;/* default quant type for linear layers */
    uint32_t num_tensors;  /* number of tensor descriptors */
    uint64_t data_offset;  /* byte offset from file start to tensor data */
} vqf_header_t;

#define VQF_MAX_NAME 256
#define VQF_MAX_DIMS 4

typedef struct {
    uint16_t name_len;
    char name[VQF_MAX_NAME];
    uint32_t qtype;        /* VQF_TYPE_* */
    uint32_t ndim;
    int64_t shape[VQF_MAX_DIMS];
    uint64_t data_offset;  /* relative to vqf_header_t.data_offset */
    uint64_t data_size;    /* bytes */
} vqf_tensor_desc_t;

/* ========================================================================
 * Helper functions
 * ======================================================================== */

static inline size_t vqf_block_size(int qtype) {
    switch (qtype) {
        case VQF_TYPE_Q8_0: return VQF_Q8_0_BLOCK_SIZE;
        case VQF_TYPE_Q4_0: return VQF_Q4_0_BLOCK_SIZE;
        case VQF_TYPE_Q4_K: return VQF_Q4_K_BLOCK_SIZE;
        default: return 0;
    }
}

static inline size_t vqf_block_bytes(int qtype) {
    switch (qtype) {
        case VQF_TYPE_Q8_0: return VQF_Q8_0_BLOCK_BYTES;
        case VQF_TYPE_Q4_0: return VQF_Q4_0_BLOCK_BYTES;
        case VQF_TYPE_Q4_K: return VQF_Q4_K_BLOCK_BYTES;
        default: return 0;
    }
}

/* Number of quantized blocks for a tensor with `numel` elements */
static inline int64_t vqf_num_blocks(int qtype, int64_t numel) {
    size_t bs = vqf_block_size(qtype);
    if (bs == 0) return 0;
    return (numel + (int64_t)bs - 1) / (int64_t)bs;
}

/* Total bytes for a quantized tensor */
static inline size_t vqf_tensor_bytes(int qtype, int64_t numel) {
    switch (qtype) {
        case VQF_TYPE_F32:  return (size_t)numel * sizeof(float);
        case VQF_TYPE_BF16: return (size_t)numel * sizeof(uint16_t);
        default: {
            int64_t nb = vqf_num_blocks(qtype, numel);
            return (size_t)nb * vqf_block_bytes(qtype);
        }
    }
}

/* Type name string */
static inline const char *vqf_type_name(int qtype) {
    switch (qtype) {
        case VQF_TYPE_F32:  return "F32";
        case VQF_TYPE_F16:  return "F16";
        case VQF_TYPE_BF16: return "BF16";
        case VQF_TYPE_Q8_0: return "Q8_0";
        case VQF_TYPE_Q4_0: return "Q4_0";
        case VQF_TYPE_Q4_K: return "Q4_K";
        default: return "UNKNOWN";
    }
}

/* ========================================================================
 * Quantized linear layer (CPU + CUDA dispatch)
 *
 * Drop-in replacement for vox_linear_nobias_bf16() when quantized weights
 * are available. Tries CUDA first, falls back to CPU.
 * Defined in voxtral_quant_kernels.c
 * ======================================================================== */

void vox_linear_nobias_quant(float *y, const float *x, const void *W_q,
                              int seq_len, int in_dim, int out_dim, int qtype);

#endif /* VOXTRAL_QUANT_H */
