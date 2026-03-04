/*
 * voxtral_quant_loader.h - VQF quantized model file reader
 */

#ifndef VOXTRAL_QUANT_LOADER_H
#define VOXTRAL_QUANT_LOADER_H

#include "voxtral.h"

typedef struct vqf_mapped_file vqf_mapped_file_t;

/* Open a VQF file for reading (memory-mapped) */
vqf_mapped_file_t *vqf_open(const char *path);

/* Close and free resources */
void vqf_close(vqf_mapped_file_t *vf);

/* Load encoder weights from VQF file */
int vqf_load_encoder(vox_encoder_t *enc, vqf_mapped_file_t *vf);

/* Load decoder weights from VQF file */
int vqf_load_decoder(vox_decoder_t *dec, vqf_mapped_file_t *vf);

/* Load adapter weights from VQF file */
int vqf_load_adapter(vox_adapter_t *adapter, vqf_mapped_file_t *vf);

#endif /* VOXTRAL_QUANT_LOADER_H */
