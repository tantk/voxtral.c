/*
 * voxtral_cuda_quant.h - CUDA integration for quantized (VQF) weights
 */

#ifndef VOXTRAL_CUDA_QUANT_H
#define VOXTRAL_CUDA_QUANT_H

#include "voxtral.h"
#include <stddef.h>

#ifdef USE_CUDA
#include <cuda.h>

/* Load quantized GEMV kernel functions from the cubin module.
 * Must be called after cuda_load_kernel_module(). */
int vox_cuda_quant_load_kernels(void);

/* Upload all quantized weights for a model to GPU (permanently resident).
 * Call once after model load. */
int vox_cuda_quant_upload_all(vox_ctx_t *ctx);

/* Free all permanently resident quantized weights. */
void vox_cuda_quant_free_all(void);

/* Look up a device pointer for a host quantized weight pointer.
 * Returns 0 if not found. */
CUdeviceptr vox_cuda_quant_weight_get(const void *host_ptr);

/* Upload a single quantized weight tensor to GPU.
 * Returns device pointer, or 0 on failure. */
CUdeviceptr vox_cuda_quant_weight_upload(const void *host_ptr, size_t bytes);

/* Launch quantized GEMV on device: out[N] = x[K] @ W^T[N,K]
 * All pointers are device pointers. */
int vox_cuda_quant_gemv_dev(CUdeviceptr dOut, CUdeviceptr dX, CUdeviceptr dW,
                             int K, int N, int qtype);

/* Launch quantized GEMV with beta: out[N] = beta*out[N] + x[K] @ W^T[N,K] */
int vox_cuda_quant_gemv_beta_dev(CUdeviceptr dOut, CUdeviceptr dX, CUdeviceptr dW,
                                  int K, int N, int qtype, float beta);

/* Host-side quantized matmul (upload activation, dispatch, download result).
 * Currently M=1 only. */
int vox_cuda_quant_matmul_t(float *out, const float *x, const void *W_q,
                             int M, int K, int N, int qtype);

#else
/* Stubs when CUDA is not available */
static inline int vox_cuda_quant_load_kernels(void) { return 0; }
static inline int vox_cuda_quant_upload_all(vox_ctx_t *ctx) { (void)ctx; return 0; }
static inline void vox_cuda_quant_free_all(void) {}
static inline int vox_cuda_quant_matmul_t(float *out, const float *x, const void *W_q,
                                           int M, int K, int N, int qtype) {
    (void)out; (void)x; (void)W_q; (void)M; (void)K; (void)N; (void)qtype;
    return 0;
}
#endif

#endif /* VOXTRAL_CUDA_QUANT_H */
