/*
 * voxtral_cuda_quant.c - CUDA integration for quantized (VQF) weights
 *
 * Provides:
 * - Permanent GPU upload of quantized weight blocks
 * - Dispatch wrappers that call quantized GEMV kernels
 * - Helper for CUDA graph capture with quantized weights
 *
 * This file is compiled only with -DUSE_CUDA and links against
 * the same cubin as voxtral_cuda.c (kernels are embedded).
 */

#include "voxtral_cuda_quant.h"
#include "voxtral_quant.h"
#include "voxtral.h"

#ifndef USE_CUDA
#error "voxtral_cuda_quant.c requires -DUSE_CUDA"
#endif

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int vox_verbose;

/* ========================================================================
 * External CUDA state (owned by voxtral_cuda.c)
 * ======================================================================== */

/* These are defined in voxtral_cuda.c and we access them via the
 * existing public API or by extern'ing the relevant globals. */
extern CUcontext g_ctx;
extern CUstream g_stream;
extern CUmodule g_mod;
extern int g_available;

/* ========================================================================
 * Quantized kernel function handles
 * ======================================================================== */

static CUfunction g_fn_gemv_q8_0 = 0;
static CUfunction g_fn_gemv_q4_0 = 0;
static CUfunction g_fn_gemv_q4_k = 0;
static CUfunction g_fn_gemv_q8_0_beta = 0;
static CUfunction g_fn_gemv_q4_0_beta = 0;
static CUfunction g_fn_gemv_q4_k_beta = 0;

static int g_quant_kernels_loaded = 0;

int vox_cuda_quant_load_kernels(void) {
    if (g_quant_kernels_loaded) return 1;
    if (!g_mod) return 0;

    (void)cuModuleGetFunction(&g_fn_gemv_q8_0, g_mod, "vox_gemv_q8_0");
    (void)cuModuleGetFunction(&g_fn_gemv_q4_0, g_mod, "vox_gemv_q4_0");
    (void)cuModuleGetFunction(&g_fn_gemv_q4_k, g_mod, "vox_gemv_q4_k");
    (void)cuModuleGetFunction(&g_fn_gemv_q8_0_beta, g_mod, "vox_gemv_q8_0_beta");
    (void)cuModuleGetFunction(&g_fn_gemv_q4_0_beta, g_mod, "vox_gemv_q4_0_beta");
    (void)cuModuleGetFunction(&g_fn_gemv_q4_k_beta, g_mod, "vox_gemv_q4_k_beta");

    if (g_fn_gemv_q8_0 || g_fn_gemv_q4_0 || g_fn_gemv_q4_k) {
        g_quant_kernels_loaded = 1;
        if (vox_verbose >= 1) {
            fprintf(stderr, "[cuda-quant] kernels loaded: q8_0=%s q4_0=%s q4_k=%s\n",
                    g_fn_gemv_q8_0 ? "yes" : "no",
                    g_fn_gemv_q4_0 ? "yes" : "no",
                    g_fn_gemv_q4_k ? "yes" : "no");
        }
    }
    return g_quant_kernels_loaded;
}

/* ========================================================================
 * Permanent GPU weight storage
 *
 * Quantized weights are small enough to keep ALL permanently in VRAM.
 * We use a simple array of (host_ptr, dev_ptr, bytes) entries.
 * ======================================================================== */

typedef struct {
    const void *host;
    CUdeviceptr dev;
    size_t bytes;
} quant_weight_entry_t;

static quant_weight_entry_t *g_qw_cache = NULL;
static int g_qw_cache_len = 0;
static int g_qw_cache_cap = 0;
static size_t g_qw_total_bytes = 0;

/* Look up a device pointer for a host quantized weight pointer */
CUdeviceptr vox_cuda_quant_weight_get(const void *host_ptr) {
    if (!host_ptr) return 0;
    for (int i = 0; i < g_qw_cache_len; i++) {
        if (g_qw_cache[i].host == host_ptr)
            return g_qw_cache[i].dev;
    }
    return 0;
}

/* Upload a quantized weight tensor to GPU (permanently resident) */
CUdeviceptr vox_cuda_quant_weight_upload(const void *host_ptr, size_t bytes) {
    if (!host_ptr || bytes == 0) return 0;
    if (!g_available) return 0;

    /* Check if already uploaded */
    CUdeviceptr existing = vox_cuda_quant_weight_get(host_ptr);
    if (existing) return existing;

    (void)cuCtxSetCurrent(g_ctx);

    /* Grow cache array if needed */
    if (g_qw_cache_len >= g_qw_cache_cap) {
        int new_cap = g_qw_cache_cap ? g_qw_cache_cap * 2 : 512;
        quant_weight_entry_t *tmp = (quant_weight_entry_t *)realloc(
            g_qw_cache, (size_t)new_cap * sizeof(quant_weight_entry_t));
        if (!tmp) return 0;
        g_qw_cache = tmp;
        g_qw_cache_cap = new_cap;
    }

    /* Allocate and upload */
    CUdeviceptr dev = 0;
    CUresult r = cuMemAlloc(&dev, bytes);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "[cuda-quant] cuMemAlloc failed for %zu bytes\n", bytes);
        return 0;
    }

    r = cuMemcpyHtoDAsync(dev, host_ptr, bytes, g_stream);
    if (r != CUDA_SUCCESS) {
        cuMemFree(dev);
        return 0;
    }

    g_qw_cache[g_qw_cache_len++] = (quant_weight_entry_t){
        .host = host_ptr,
        .dev = dev,
        .bytes = bytes,
    };
    g_qw_total_bytes += bytes;

    return dev;
}

/* Upload all quantized weights for a model to GPU */
int vox_cuda_quant_upload_all(vox_ctx_t *ctx) {
    if (!ctx || !ctx->use_quant) return 0;
    if (!g_available) return 0;

    (void)cuCtxSetCurrent(g_ctx);

    int uploaded = 0;
    size_t total = 0;

    /* Encoder layers */
    for (int i = 0; i < VOX_ENC_LAYERS; i++) {
        vox_enc_layer_t *l = &ctx->encoder.layers[i];

#define UPLOAD_ENC(field) do { \
    if (l->field##_weight_q && l->field##_qtype != VQF_TYPE_F32 && l->field##_qtype != VQF_TYPE_BF16) { \
        size_t bytes = vqf_tensor_bytes(l->field##_qtype, l->field##_numel); \
        CUdeviceptr d = vox_cuda_quant_weight_upload(l->field##_weight_q, bytes); \
        if (d) { uploaded++; total += bytes; } \
        else { fprintf(stderr, "[cuda-quant] failed to upload enc.%d.%s\n", i, #field); } \
    } \
} while(0)

        UPLOAD_ENC(wq);
        UPLOAD_ENC(wk);
        UPLOAD_ENC(wv);
        UPLOAD_ENC(wo);
        UPLOAD_ENC(w1);
        UPLOAD_ENC(w2);
        UPLOAD_ENC(w3);
#undef UPLOAD_ENC
    }

    /* Decoder layers */
    for (int i = 0; i < VOX_DEC_LAYERS; i++) {
        vox_dec_layer_t *l = &ctx->decoder.layers[i];

#define UPLOAD_DEC(field) do { \
    if (l->field##_weight_q && l->field##_qtype != VQF_TYPE_F32 && l->field##_qtype != VQF_TYPE_BF16) { \
        size_t bytes = vqf_tensor_bytes(l->field##_qtype, l->field##_numel); \
        CUdeviceptr d = vox_cuda_quant_weight_upload(l->field##_weight_q, bytes); \
        if (d) { uploaded++; total += bytes; } \
        else { fprintf(stderr, "[cuda-quant] failed to upload dec.%d.%s\n", i, #field); } \
    } \
} while(0)

        UPLOAD_DEC(wq);
        UPLOAD_DEC(wk);
        UPLOAD_DEC(wv);
        UPLOAD_DEC(wo);
        UPLOAD_DEC(w1);
        UPLOAD_DEC(w2);
        UPLOAD_DEC(w3);
#undef UPLOAD_DEC
    }

    /* Sync to ensure all uploads complete */
    cuStreamSynchronize(g_stream);

    if (vox_verbose >= 1) {
        fprintf(stderr, "[cuda-quant] uploaded %d tensors, %.1f MB total VRAM\n",
                uploaded, (double)total / (1024.0 * 1024.0));
    }

    return 1;
}

/* Free all permanently resident quantized weights */
void vox_cuda_quant_free_all(void) {
    if (!g_qw_cache) return;
    (void)cuCtxSetCurrent(g_ctx);
    for (int i = 0; i < g_qw_cache_len; i++) {
        if (g_qw_cache[i].dev)
            cuMemFree(g_qw_cache[i].dev);
    }
    free(g_qw_cache);
    g_qw_cache = NULL;
    g_qw_cache_len = 0;
    g_qw_cache_cap = 0;
    g_qw_total_bytes = 0;
}

/* ========================================================================
 * GEMV dispatch: quantized matmul on device
 *
 * These work on device pointers (weights already uploaded, activation
 * already on device). Used inside CUDA graph capture.
 * ======================================================================== */

static CUfunction gemv_fn_for_qtype(int qtype) {
    switch (qtype) {
        case VQF_TYPE_Q8_0: return g_fn_gemv_q8_0;
        case VQF_TYPE_Q4_0: return g_fn_gemv_q4_0;
        case VQF_TYPE_Q4_K: return g_fn_gemv_q4_k;
        default: return 0;
    }
}

static CUfunction gemv_beta_fn_for_qtype(int qtype) {
    switch (qtype) {
        case VQF_TYPE_Q8_0: return g_fn_gemv_q8_0_beta;
        case VQF_TYPE_Q4_0: return g_fn_gemv_q4_0_beta;
        case VQF_TYPE_Q4_K: return g_fn_gemv_q4_k_beta;
        default: return 0;
    }
}

static int rows_per_block_for_qtype(int qtype) {
    switch (qtype) {
        case VQF_TYPE_Q4_K: return 16;  /* 8 warps * 2 rows/warp */
        default: return 32;              /* 8 warps * 4 rows/warp */
    }
}

/* Launch quantized GEMV: out[N] = x[K] @ W^T[N,K]
 * All pointers are device pointers.
 * K must be aligned to the block size of the quant type. */
int vox_cuda_quant_gemv_dev(CUdeviceptr dOut, CUdeviceptr dX, CUdeviceptr dW,
                             int K, int N, int qtype) {
    CUfunction fn = gemv_fn_for_qtype(qtype);
    if (!fn) return 0;

    int rpb = rows_per_block_for_qtype(qtype);
    int grid = (N + rpb - 1) / rpb;
    int threads = 256;
    size_t shmem = (size_t)K * sizeof(float);

    void *params[] = { &dOut, &dX, &dW, &K, &N };
    CUresult r = cuLaunchKernel(fn, grid, 1, 1, threads, 1, 1,
                                 shmem, g_stream, params, NULL);
    return (r == CUDA_SUCCESS);
}

/* Launch quantized GEMV with beta: out[N] = beta*out[N] + x[K] @ W^T[N,K] */
int vox_cuda_quant_gemv_beta_dev(CUdeviceptr dOut, CUdeviceptr dX, CUdeviceptr dW,
                                  int K, int N, int qtype, float beta) {
    CUfunction fn = gemv_beta_fn_for_qtype(qtype);
    if (!fn) return 0;

    int rpb = rows_per_block_for_qtype(qtype);
    int grid = (N + rpb - 1) / rpb;
    int threads = 256;
    size_t shmem = (size_t)K * sizeof(float);

    void *params[] = { &dOut, &dX, &dW, &K, &N, &beta };
    CUresult r = cuLaunchKernel(fn, grid, 1, 1, threads, 1, 1,
                                 shmem, g_stream, params, NULL);
    return (r == CUDA_SUCCESS);
}

/* ========================================================================
 * Host-side dispatch: quantized matmul (upload activation, download result)
 *
 * Used by CPU fallback path when CUDA graph is not active.
 * ======================================================================== */

/* Scratch buffers for non-graph path */
static CUdeviceptr g_qx_dev = 0;
static size_t g_qx_cap = 0;
static CUdeviceptr g_qout_dev = 0;
static size_t g_qout_cap = 0;

static int ensure_qbuf(CUdeviceptr *buf, size_t *cap, size_t needed) {
    if (*cap >= needed) return 1;
    if (*buf) cuMemFree(*buf);
    *buf = 0; *cap = 0;
    CUresult r = cuMemAlloc(buf, needed);
    if (r != CUDA_SUCCESS) return 0;
    *cap = needed;
    return 1;
}

int vox_cuda_quant_matmul_t(float *out, const float *x, const void *W_q,
                             int M, int K, int N, int qtype) {
    if (!g_available) return 0;
    if (!vox_cuda_quant_load_kernels()) return 0;

    CUdeviceptr dW = vox_cuda_quant_weight_get(W_q);
    if (!dW) return 0;

    (void)cuCtxSetCurrent(g_ctx);

    size_t bytes_x = (size_t)M * K * sizeof(float);
    size_t bytes_out = (size_t)M * N * sizeof(float);

    if (!ensure_qbuf(&g_qx_dev, &g_qx_cap, bytes_x)) return 0;
    if (!ensure_qbuf(&g_qout_dev, &g_qout_cap, bytes_out)) return 0;

    if (cuMemcpyHtoDAsync(g_qx_dev, x, bytes_x, g_stream) != CUDA_SUCCESS) return 0;

    /* Launch M GEMV kernels (one per sequence position) */
    for (int s = 0; s < M; s++) {
        CUdeviceptr dXs = g_qx_dev + (size_t)s * K * sizeof(float);
        CUdeviceptr dOs = g_qout_dev + (size_t)s * N * sizeof(float);
        if (!vox_cuda_quant_gemv_dev(dOs, dXs, dW, K, N, qtype)) return 0;
    }

    if (cuMemcpyDtoHAsync(out, g_qout_dev, bytes_out, g_stream) != CUDA_SUCCESS) return 0;

    return (cuStreamSynchronize(g_stream) == CUDA_SUCCESS);
}
