#ifndef B12X_LOADER_POOL_API_H
#define B12X_LOADER_POOL_API_H

#include <stdbool.h>
#include <stdint.h>

#define B12X_POOL_API_CAPSULE "b12x.loader.pool_api.v1"
typedef struct {
    bool (*device_range)(uintptr_t pointer, uint64_t bytes, int device);
} b12x_pool_api_t;

#endif
