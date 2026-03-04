/*
 * voxtral_quant_kernels.c - CPU dequant+matmul kernels for quantized weights
 *
 * Provides vox_linear_nobias_quant() which replaces vox_linear_nobias_bf16()
 * when quantized weights are available. Falls through to CUDA when possible.
 */

#include "voxtral_quant.h"
#include "voxtral_kernels.h"
#ifdef USE_CUDA
#include "voxtral_cuda_quant.h"
#endif
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern int vox_verbose;

/* ========================================================================
 * CPU dequant+matvec: out[N] = x[K] @ W^T[N,K]  (M=1, single token)
 * ======================================================================== */

static void matvec_q8_0(float *out, const float *x, const void *W, int K, int N) {
    int blocks_per_row = K / VQF_Q8_0_BLOCK_SIZE;
    size_t row_bytes = (size_t)blocks_per_row * VQF_Q8_0_BLOCK_BYTES;
    const uint8_t *wdata = (const uint8_t *)W;

    for (int row = 0; row < N; row++) {
        const uint8_t *rp = wdata + (size_t)row * row_bytes;
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const uint8_t *block = rp + (size_t)b * VQF_Q8_0_BLOCK_BYTES;
            float scale;
            memcpy(&scale, block, sizeof(float));
            const int8_t *quants = (const int8_t *)(block + 4);
            int k_base = b * VQF_Q8_0_BLOCK_SIZE;
            float partial = 0.0f;
            for (int i = 0; i < VQF_Q8_0_BLOCK_SIZE; i++) {
                partial += x[k_base + i] * (float)quants[i];
            }
            acc += scale * partial;
        }
        out[row] = acc;
    }
}

static void matvec_q4_0(float *out, const float *x, const void *W, int K, int N) {
    int blocks_per_row = K / VQF_Q4_0_BLOCK_SIZE;
    size_t row_bytes = (size_t)blocks_per_row * VQF_Q4_0_BLOCK_BYTES;
    const uint8_t *wdata = (const uint8_t *)W;

    for (int row = 0; row < N; row++) {
        const uint8_t *rp = wdata + (size_t)row * row_bytes;
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const uint8_t *block = rp + (size_t)b * VQF_Q4_0_BLOCK_BYTES;
            float scale;
            memcpy(&scale, block, sizeof(float));
            const uint8_t *nibs = block + 4;
            int k_base = b * VQF_Q4_0_BLOCK_SIZE;
            float partial = 0.0f;
            for (int i = 0; i < 16; i++) {
                uint8_t byte = nibs[i];
                int lo = (byte & 0xF) - 8;
                int hi = ((byte >> 4) & 0xF) - 8;
                partial += x[k_base + 2 * i] * (float)lo;
                partial += x[k_base + 2 * i + 1] * (float)hi;
            }
            acc += scale * partial;
        }
        out[row] = acc;
    }
}

static void matvec_q4_k(float *out, const float *x, const void *W, int K, int N) {
    int sblocks_per_row = K / VQF_Q4_K_BLOCK_SIZE;
    size_t row_bytes = (size_t)sblocks_per_row * VQF_Q4_K_BLOCK_BYTES;
    const uint8_t *wdata = (const uint8_t *)W;

    for (int row = 0; row < N; row++) {
        const uint8_t *rp = wdata + (size_t)row * row_bytes;
        float acc = 0.0f;
        for (int sb = 0; sb < sblocks_per_row; sb++) {
            const uint8_t *block = rp + (size_t)sb * VQF_Q4_K_BLOCK_BYTES;
            float super_scale, super_min;
            memcpy(&super_scale, block, sizeof(float));
            memcpy(&super_min, block + 4, sizeof(float));
            const uint8_t *packed = block + 8;
            const uint8_t *nibs = block + 20;

            /* Unpack 6-bit scales and mins */
            uint8_t q_scales[8], q_mins[8];
            for (int i = 0; i < 4; i++) {
                uint8_t b0 = packed[i * 3];
                uint8_t b1 = packed[i * 3 + 1];
                uint8_t b2 = packed[i * 3 + 2];
                q_scales[i * 2]     = b0 & 0x3F;
                q_scales[i * 2 + 1] = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2);
                q_mins[i * 2]       = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4);
                q_mins[i * 2 + 1]   = (b2 >> 2) & 0x3F;
            }

            int k_base = sb * VQF_Q4_K_BLOCK_SIZE;
            for (int sub = 0; sub < 8; sub++) {
                float s = (float)q_scales[sub] * super_scale;
                float m = (float)q_mins[sub] * super_min;
                const uint8_t *sub_nibs = nibs + sub * 16;
                int sk = k_base + sub * 32;
                float partial = 0.0f;
                for (int i = 0; i < 16; i++) {
                    uint8_t byte = sub_nibs[i];
                    int lo = byte & 0xF;
                    int hi = (byte >> 4) & 0xF;
                    partial += x[sk + 2 * i]     * (s * (float)lo - m);
                    partial += x[sk + 2 * i + 1] * (s * (float)hi - m);
                }
                acc += partial;
            }
        }
        out[row] = acc;
    }
}

/* ========================================================================
 * Unified dispatch: vox_linear_nobias_quant()
 *
 * Drop-in replacement for vox_linear_nobias_bf16() when quantized weights
 * are available. Tries CUDA first, falls back to CPU.
 * ======================================================================== */

void vox_linear_nobias_quant(float *y, const float *x, const void *W_q,
                              int seq_len, int in_dim, int out_dim, int qtype) {
#ifdef USE_CUDA
    /* Try CUDA quantized matmul (batched GEMV for any seq_len) */
    if (vox_cuda_quant_matmul_t(y, x, W_q, seq_len, in_dim, out_dim, qtype))
        return;
#endif

    /* CPU fallback */
    for (int s = 0; s < seq_len; s++) {
        const float *xi = x + (size_t)s * in_dim;
        float *yi = y + (size_t)s * out_dim;
        switch (qtype) {
            case VQF_TYPE_Q8_0: matvec_q8_0(yi, xi, W_q, in_dim, out_dim); break;
            case VQF_TYPE_Q4_0: matvec_q4_0(yi, xi, W_q, in_dim, out_dim); break;
            case VQF_TYPE_Q4_K: matvec_q4_k(yi, xi, W_q, in_dim, out_dim); break;
            default:
                /* Shouldn't happen — zero output as safety net */
                memset(yi, 0, (size_t)out_dim * sizeof(float));
                break;
        }
    }
}
