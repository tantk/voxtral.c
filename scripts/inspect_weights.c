/*
 * inspect_weights.c - Dump tensor names and shapes from safetensors file
 *
 * Usage: ./inspect_weights <path-to-safetensors>
 */

#include "voxtral_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <safetensors-file> [--prefix PREFIX] [--summary]\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    const char *prefix = NULL;
    int summary = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--prefix") == 0 && i + 1 < argc) {
            prefix = argv[++i];
        } else if (strcmp(argv[i], "--summary") == 0) {
            summary = 1;
        }
    }

    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) {
        fprintf(stderr, "Failed to open %s\n", path);
        return 1;
    }

    printf("File: %s\n", path);
    printf("Size: %.2f GB\n", (double)sf->file_size / (1024.0 * 1024.0 * 1024.0));
    printf("Header: %zu bytes\n", sf->header_size);
    printf("Tensors: %d\n\n", sf->num_tensors);

    if (summary) {
        /* Print unique prefixes (first two dot-separated components) */
        char prefixes[256][128];
        int counts[256];
        int n_prefixes = 0;

        for (int i = 0; i < sf->num_tensors; i++) {
            const char *name = sf->tensors[i].name;
            char pfx[128];

            /* Extract prefix: up to second dot */
            const char *dot1 = strchr(name, '.');
            if (dot1) {
                const char *dot2 = strchr(dot1 + 1, '.');
                if (dot2) {
                    int len = (int)(dot2 - name);
                    if (len > 127) len = 127;
                    memcpy(pfx, name, len);
                    pfx[len] = '\0';
                } else {
                    snprintf(pfx, sizeof(pfx), "%s", name);
                }
            } else {
                snprintf(pfx, sizeof(pfx), "%s", name);
            }

            /* Find or add prefix */
            int found = 0;
            for (int j = 0; j < n_prefixes; j++) {
                if (strcmp(prefixes[j], pfx) == 0) {
                    counts[j]++;
                    found = 1;
                    break;
                }
            }
            if (!found && n_prefixes < 256) {
                snprintf(prefixes[n_prefixes], sizeof(prefixes[0]), "%s", pfx);
                counts[n_prefixes] = 1;
                n_prefixes++;
            }
        }

        printf("Tensor prefixes:\n");
        for (int i = 0; i < n_prefixes; i++) {
            printf("  %-50s  (%d tensors)\n", prefixes[i], counts[i]);
        }
    } else {
        /* Print all tensors, optionally filtered by prefix */
        const char *dtype_names[] = {"F32", "F16", "BF16", "I32", "I64", "BOOL"};
        size_t total_bytes = 0;

        for (int i = 0; i < sf->num_tensors; i++) {
            const safetensor_t *t = &sf->tensors[i];

            if (prefix && strncmp(t->name, prefix, strlen(prefix)) != 0) {
                continue;
            }

            const char *dtype_name = t->dtype >= 0 && t->dtype <= 5 ?
                                     dtype_names[t->dtype] : "UNKNOWN";

            printf("%-70s  %4s  [", t->name, dtype_name);
            for (int j = 0; j < t->ndim; j++) {
                printf("%ld%s", (long)t->shape[j], j < t->ndim - 1 ? ", " : "");
            }
            printf("]");

            int64_t numel = safetensor_numel(t);
            if (numel > 1024 * 1024) {
                printf("  %.1fM", (double)numel / (1024.0 * 1024.0));
            } else if (numel > 1024) {
                printf("  %.1fK", (double)numel / 1024.0);
            } else {
                printf("  %ld", (long)numel);
            }
            printf("\n");

            total_bytes += t->data_size;
        }

        printf("\nTotal data: %.2f GB\n", (double)total_bytes / (1024.0 * 1024.0 * 1024.0));
    }

    safetensors_close(sf);
    return 0;
}
