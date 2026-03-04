/*
 * voxtral_quant_loader.c - VQF quantized model file reader
 *
 * Parses VQF container headers, mmaps tensor data, and populates
 * the quantized weight pointers in vox_enc_layer_t / vox_dec_layer_t.
 */

#include "voxtral_quant.h"
#include "voxtral.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

extern int vox_verbose;

/* ========================================================================
 * File handle
 * ======================================================================== */

typedef struct vqf_mapped_file {
    void *data;              /* mmap'd file */
    size_t file_size;
    vqf_header_t header;
    int num_tensors;
    vqf_tensor_desc_t *tensors;  /* allocated array */
} vqf_mapped_file_t;

/* ========================================================================
 * Open and parse VQF file
 * ======================================================================== */

static int vqf_parse_descriptors(vqf_mapped_file_t *vf) {
    const uint8_t *p = (const uint8_t *)vf->data;
    size_t offset = sizeof(vqf_header_t);

    vf->tensors = (vqf_tensor_desc_t *)calloc(vf->num_tensors, sizeof(vqf_tensor_desc_t));
    if (!vf->tensors) return -1;

    for (int i = 0; i < vf->num_tensors; i++) {
        vqf_tensor_desc_t *t = &vf->tensors[i];

        if (offset + 2 > vf->header.data_offset) {
            fprintf(stderr, "vqf: descriptor %d overflows header\n", i);
            return -1;
        }

        uint16_t name_len;
        memcpy(&name_len, p + offset, 2); offset += 2;
        if (name_len >= VQF_MAX_NAME) {
            fprintf(stderr, "vqf: tensor name too long (%d)\n", name_len);
            return -1;
        }
        t->name_len = name_len;
        memcpy(t->name, p + offset, name_len); offset += name_len;
        t->name[name_len] = '\0';

        memcpy(&t->qtype, p + offset, 4); offset += 4;
        memcpy(&t->ndim, p + offset, 4); offset += 4;
        for (int d = 0; d < VQF_MAX_DIMS; d++) {
            memcpy(&t->shape[d], p + offset, 8); offset += 8;
        }
        memcpy(&t->data_offset, p + offset, 8); offset += 8;
        memcpy(&t->data_size, p + offset, 8); offset += 8;

        /* Validate data bounds */
        size_t abs_start = vf->header.data_offset + t->data_offset;
        size_t abs_end = abs_start + t->data_size;
        if (abs_end < abs_start || abs_end > vf->file_size) {
            fprintf(stderr, "vqf: tensor '%s' data out of bounds\n", t->name);
            return -1;
        }
    }

    return 0;
}

vqf_mapped_file_t *vqf_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("vqf_open: open failed");
        return NULL;
    }

#ifdef _WIN32
    struct _stat64 st;
    if (_fstat64(fd, &st) < 0) {
#else
    struct stat st;
    if (fstat(fd, &st) < 0) {
#endif
        perror("vqf_open: fstat failed");
        close(fd);
        return NULL;
    }

    size_t file_size = (size_t)st.st_size;
    if (file_size < sizeof(vqf_header_t)) {
        fprintf(stderr, "vqf_open: file too small\n");
        close(fd);
        return NULL;
    }

    void *data = NULL;
#ifdef _WIN32
    HANDLE hFile = (HANDLE)_get_osfhandle(fd);
    HANDLE hMap = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hMap) {
        data = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(hMap);
    }
    close(fd);
    if (!data) {
        fprintf(stderr, "vqf_open: MapViewOfFile failed\n");
        return NULL;
    }
#else
    data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) {
        perror("vqf_open: mmap failed");
        return NULL;
    }
#endif

    vqf_mapped_file_t *vf = (vqf_mapped_file_t *)calloc(1, sizeof(vqf_mapped_file_t));
    if (!vf) {
#ifdef _WIN32
        UnmapViewOfFile(data);
#else
        munmap(data, file_size);
#endif
        return NULL;
    }
    vf->data = data;
    vf->file_size = file_size;

    /* Read header */
    memcpy(&vf->header, data, sizeof(vqf_header_t));

    if (vf->header.magic != VQF_MAGIC) {
        fprintf(stderr, "vqf_open: invalid magic (expected 0x%08X, got 0x%08X)\n",
                VQF_MAGIC, vf->header.magic);
        goto fail;
    }
    if (vf->header.version != VQF_VERSION) {
        fprintf(stderr, "vqf_open: unsupported version %d\n", vf->header.version);
        goto fail;
    }
    if (vf->header.data_offset > file_size) {
        fprintf(stderr, "vqf_open: data_offset beyond file\n");
        goto fail;
    }

    vf->num_tensors = (int)vf->header.num_tensors;
    if (vqf_parse_descriptors(vf) != 0) goto fail;

    if (vox_verbose >= 1)
        fprintf(stderr, "Opened VQF file: %d tensors, default quant=%s\n",
                vf->num_tensors, vqf_type_name(vf->header.default_qtype));

    return vf;

fail:
#ifdef _WIN32
    UnmapViewOfFile(data);
#else
    munmap(data, file_size);
#endif
    free(vf->tensors);
    free(vf);
    return NULL;
}

void vqf_close(vqf_mapped_file_t *vf) {
    if (!vf) return;
    if (vf->data) {
#ifdef _WIN32
        UnmapViewOfFile(vf->data);
#else
        munmap(vf->data, vf->file_size);
#endif
    }
    free(vf->tensors);
    free(vf);
}

/* Get pointer to tensor data within mmap'd region */
static const void *vqf_tensor_data(const vqf_mapped_file_t *vf, const vqf_tensor_desc_t *t) {
    return (const uint8_t *)vf->data + vf->header.data_offset + t->data_offset;
}

/* Find tensor by name */
static const vqf_tensor_desc_t *vqf_find(const vqf_mapped_file_t *vf, const char *name) {
    for (int i = 0; i < vf->num_tensors; i++) {
        if (strcmp(vf->tensors[i].name, name) == 0)
            return &vf->tensors[i];
    }
    return NULL;
}

/* ========================================================================
 * Weight loading helpers
 * ======================================================================== */

/* Load a quantized tensor: returns pointer to mmap'd data and sets qtype/numel */
static void *load_quant(vqf_mapped_file_t *vf, const char *name, int *out_qtype, int64_t *out_numel) {
    const vqf_tensor_desc_t *t = vqf_find(vf, name);
    if (!t) {
        fprintf(stderr, "vqf: weight not found: %s\n", name);
        return NULL;
    }

    int64_t numel = 1;
    for (int d = 0; d < (int)t->ndim; d++) numel *= t->shape[d];

    *out_qtype = (int)t->qtype;
    *out_numel = numel;

    return (void *)vqf_tensor_data(vf, t);
}

/* Load F32 tensor (allocates, converts if BF16) */
static float *load_f32(vqf_mapped_file_t *vf, const char *name) {
    const vqf_tensor_desc_t *t = vqf_find(vf, name);
    if (!t) {
        fprintf(stderr, "vqf: weight not found: %s\n", name);
        return NULL;
    }

    int64_t numel = 1;
    for (int d = 0; d < (int)t->ndim; d++) numel *= t->shape[d];

    const void *src = vqf_tensor_data(vf, t);
    float *out = (float *)malloc((size_t)numel * sizeof(float));
    if (!out) return NULL;

    if (t->qtype == VQF_TYPE_F32) {
        memcpy(out, src, (size_t)numel * sizeof(float));
    } else if (t->qtype == VQF_TYPE_BF16) {
        const uint16_t *bf16 = (const uint16_t *)src;
        for (int64_t i = 0; i < numel; i++) {
            uint32_t bits = ((uint32_t)bf16[i]) << 16;
            memcpy(&out[i], &bits, sizeof(float));
        }
    } else {
        fprintf(stderr, "vqf: unexpected type %s for f32 load of %s\n",
                vqf_type_name(t->qtype), name);
        free(out);
        return NULL;
    }

    return out;
}

/* Load BF16 tensor as direct pointer (no allocation) */
static uint16_t *load_bf16_direct(vqf_mapped_file_t *vf, const char *name) {
    const vqf_tensor_desc_t *t = vqf_find(vf, name);
    if (!t) {
        fprintf(stderr, "vqf: weight not found: %s\n", name);
        return NULL;
    }
    if (t->qtype != VQF_TYPE_BF16) {
        fprintf(stderr, "vqf: expected BF16 for %s, got %s\n",
                name, vqf_type_name(t->qtype));
        return NULL;
    }
    return (uint16_t *)vqf_tensor_data(vf, t);
}

/* ========================================================================
 * Populate weight structs from VQF file
 * ======================================================================== */

static int vqf_load_enc_layer_quant(vox_enc_layer_t *l, vqf_mapped_file_t *vf,
                                     const char *prefix, int layer_idx) {
    char name[512];

#define LOAD_ENC_Q(field, suffix) do { \
    snprintf(name, sizeof(name), "%s.%d.%s", prefix, layer_idx, suffix); \
    l->field##_weight_q = load_quant(vf, name, &l->field##_qtype, &l->field##_numel); \
    if (!l->field##_weight_q) return -1; \
} while(0)

    LOAD_ENC_Q(wq, "attention.wq.weight");
    LOAD_ENC_Q(wk, "attention.wk.weight");
    LOAD_ENC_Q(wv, "attention.wv.weight");
    LOAD_ENC_Q(wo, "attention.wo.weight");
    LOAD_ENC_Q(w1, "feed_forward.w1.weight");
    LOAD_ENC_Q(w2, "feed_forward.w2.weight");
    LOAD_ENC_Q(w3, "feed_forward.w3.weight");
#undef LOAD_ENC_Q

    /* Biases and norms: always F32 */
    snprintf(name, sizeof(name), "%s.%d.attention.wq.bias", prefix, layer_idx);
    l->wq_bias = load_f32(vf, name);
    snprintf(name, sizeof(name), "%s.%d.attention.wv.bias", prefix, layer_idx);
    l->wv_bias = load_f32(vf, name);
    snprintf(name, sizeof(name), "%s.%d.attention.wo.bias", prefix, layer_idx);
    l->wo_bias = load_f32(vf, name);
    snprintf(name, sizeof(name), "%s.%d.attention_norm.weight", prefix, layer_idx);
    l->attention_norm = load_f32(vf, name);
    snprintf(name, sizeof(name), "%s.%d.feed_forward.w2.bias", prefix, layer_idx);
    l->w2_bias = load_f32(vf, name);
    snprintf(name, sizeof(name), "%s.%d.ffn_norm.weight", prefix, layer_idx);
    l->ffn_norm = load_f32(vf, name);

    return 0;
}

static int vqf_load_dec_layer_quant(vox_dec_layer_t *l, vqf_mapped_file_t *vf, int layer_idx) {
    char name[512];

#define LOAD_DEC_Q(field, suffix) do { \
    snprintf(name, sizeof(name), "layers.%d.%s", layer_idx, suffix); \
    l->field##_weight_q = load_quant(vf, name, &l->field##_qtype, &l->field##_numel); \
    if (!l->field##_weight_q) return -1; \
} while(0)

    LOAD_DEC_Q(wq, "attention.wq.weight");
    LOAD_DEC_Q(wk, "attention.wk.weight");
    LOAD_DEC_Q(wv, "attention.wv.weight");
    LOAD_DEC_Q(wo, "attention.wo.weight");
    LOAD_DEC_Q(w1, "feed_forward.w1.weight");
    LOAD_DEC_Q(w2, "feed_forward.w2.weight");
    LOAD_DEC_Q(w3, "feed_forward.w3.weight");
#undef LOAD_DEC_Q

    /* Ada norm MLP (small, always f32) */
    snprintf(name, sizeof(name), "layers.%d.ada_rms_norm_t_cond.0.weight", layer_idx);
    l->ada_norm_down = load_f32(vf, name);
    snprintf(name, sizeof(name), "layers.%d.ada_rms_norm_t_cond.2.weight", layer_idx);
    l->ada_norm_up = load_f32(vf, name);

    /* Norms */
    snprintf(name, sizeof(name), "layers.%d.attention_norm.weight", layer_idx);
    l->attention_norm = load_f32(vf, name);
    snprintf(name, sizeof(name), "layers.%d.ffn_norm.weight", layer_idx);
    l->ffn_norm = load_f32(vf, name);

    return 0;
}

/* ========================================================================
 * Main loading entry points (called from voxtral.c)
 * ======================================================================== */

int vqf_load_encoder(vox_encoder_t *enc, vqf_mapped_file_t *vf) {
    char name[512];
    const char *prefix = "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers";

    /* Conv stem (always F32) */
    snprintf(name, sizeof(name), "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.weight");
    enc->conv0_weight = load_f32(vf, name);
    snprintf(name, sizeof(name), "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.bias");
    enc->conv0_bias = load_f32(vf, name);
    snprintf(name, sizeof(name), "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.1.conv.weight");
    enc->conv1_weight = load_f32(vf, name);
    snprintf(name, sizeof(name), "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.1.conv.bias");
    enc->conv1_bias = load_f32(vf, name);
    if (!enc->conv0_weight || !enc->conv1_weight) return -1;

    for (int i = 0; i < VOX_ENC_LAYERS; i++) {
        if (vqf_load_enc_layer_quant(&enc->layers[i], vf, prefix, i) != 0) {
            fprintf(stderr, "vqf: failed to load encoder layer %d\n", i);
            return -1;
        }
        if (vox_verbose >= 2)
            fprintf(stderr, "  Encoder layer %d/%d loaded (quantized)\n", i + 1, VOX_ENC_LAYERS);
    }

    snprintf(name, sizeof(name), "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.norm.weight");
    enc->norm = load_f32(vf, name);
    if (!enc->norm) return -1;

    return 0;
}

int vqf_load_decoder(vox_decoder_t *dec, vqf_mapped_file_t *vf) {
    /* Token embeddings: always BF16 (not quantized) */
    dec->tok_embeddings_bf16 = load_bf16_direct(vf,
        "mm_streams_embeddings.embedding_module.tok_embeddings.weight");
    if (!dec->tok_embeddings_bf16) return -1;

    for (int i = 0; i < VOX_DEC_LAYERS; i++) {
        if (vqf_load_dec_layer_quant(&dec->layers[i], vf, i) != 0) {
            fprintf(stderr, "vqf: failed to load decoder layer %d\n", i);
            return -1;
        }
        if (vox_verbose >= 2)
            fprintf(stderr, "  Decoder layer %d/%d loaded (quantized)\n", i + 1, VOX_DEC_LAYERS);
    }

    dec->norm = load_f32(vf, "norm.weight");
    if (!dec->norm) return -1;

    return 0;
}

int vqf_load_adapter(vox_adapter_t *adapter, vqf_mapped_file_t *vf) {
    /* Adapter stays BF16 (too small to quantize) */
    adapter->linear0_weight_bf16 = load_bf16_direct(vf,
        "mm_streams_embeddings.embedding_module.audio_language_projection.0.weight");
    adapter->linear1_weight_bf16 = load_bf16_direct(vf,
        "mm_streams_embeddings.embedding_module.audio_language_projection.2.weight");

    if (!adapter->linear0_weight_bf16 || !adapter->linear1_weight_bf16) return -1;
    return 0;
}
