// CONVOLUCIÓN CUDA PARA IMÁGENES EN ESCALA DE GRISES
//
// Implementa convolución horizontal y vertical en GPU con memoria unificada.
// Las funciones `convolve_even` y `convolve_odd` usan kernels CUDA internos.
//
// Uso:
//  - `convolve_even`: convolución horizontal
//  - `convolve_odd`: convolución vertical

#ifndef CONVOLVE_CUH
#define CONVOLVE_CUH

#include <vector>
#include <algorithm>
#include <cmath>
#include "image.h"
#include <cuda_runtime.h>

// Verificación de errores CUDA
#ifndef cudaCheckError
#define cudaCheckError() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}
#endif

// Kernel: convolución horizontal
__global__ void convolve_horizontal_kernel(float *src, float *dst, const float *mask, int width, int height, int mask_len) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = mask[0] * src[y * width + x];
        for (int i = 1; i < mask_len; i++) {
            sum += mask[i] * (
                src[y * width + max(x - i, 0)] +
                src[y * width + min(x + i, width - 1)]
            );
        }
        dst[y * width + x] = sum;
    }
}

// Kernel: convolución vertical
__global__ void convolve_vertical_kernel(float *src, float *dst, const float *mask, int width, int height, int mask_len) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = mask[0] * src[y * width + x];
        for (int i = 1; i < mask_len; i++) {
            sum += mask[i] * (
                src[max(y - i, 0) * width + x] +
                src[min(y + i, height - 1) * width + x]
            );
        }
        dst[y * width + x] = sum;
    }
}

// Función general de convolución
static void convolve(image<float> *src, image<float> *dst, const std::vector<float> &mask, bool vertical = false) {
    int width = src->width();
    int height = src->height();
    int num_pixels = width * height;
    int mask_len = mask.size();

    float *d_src, *d_dst, *d_mask;
    cudaMallocManaged(&d_src, num_pixels * sizeof(float)); cudaCheckError();
    cudaMallocManaged(&d_dst, num_pixels * sizeof(float)); cudaCheckError();
    cudaMallocManaged(&d_mask, mask_len * sizeof(float)); cudaCheckError();

    std::copy(src->data, src->data + num_pixels, d_src);
    std::copy(mask.begin(), mask.end(), d_mask);

    dim3 block(32, 16);
    dim3 grid((width + 31) / 32, (height + 15) / 16);

    if (vertical) {
        convolve_vertical_kernel<<<grid, block>>>(d_src, d_dst, d_mask, width, height, mask_len);
    } else {
        convolve_horizontal_kernel<<<grid, block>>>(d_src, d_dst, d_mask, width, height, mask_len);
    }

    cudaDeviceSynchronize(); cudaCheckError();

    std::copy(d_dst, d_dst + num_pixels, dst->data);

    cudaFree(d_src); cudaCheckError();
    cudaFree(d_dst); cudaCheckError();
    cudaFree(d_mask); cudaCheckError();
}

// Alias: convolución horizontal
static void convolve_even(image<float> *src, image<float> *dst, std::vector<float> &mask) {
    convolve(src, dst, mask, false);
}

// Alias: convolución vertical
static void convolve_odd(image<float> *src, image<float> *dst, std::vector<float> &mask) {
    convolve(src, dst, mask, true);
}

#endif

