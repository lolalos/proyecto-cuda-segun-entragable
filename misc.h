// misc.h
#ifndef MISC_H
#define MISC_H

#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

typedef unsigned char uchar;

typedef struct { uchar r, g, b; } rgb;

// It's unlikely this operator will be used in device code, but if so, it would also need __host__ __device__
inline bool operator==(const rgb &a, const rgb &b) {
    return ((a.r == b.r) && (a.g == b.g) && (a.b == b.b));
}

template <class T>
// Change from __device__ to __host__ __device__
__host__ __device__ inline T abs(const T &x) { return (x > 0 ? x : -x); };

template <class T>
// Change from __device__ to __host__ __device__
__host__ __device__ inline int sign(const T &x) { return (x >= 0 ? 1 : -1); };

template <class T>
// Change from __device__ to __host__ __device__
__host__ __device__ inline T square(const T &x) { return x*x; };

template <class T>
// Change from __device__ to __host__ __device__
__host__ __device__ inline T bound(const T &x, const T &min, const T &max) {
    return (x < min ? min : (x > max ? max : x));
}

template <class T>
// Change from __device__ to __host__ __device__
__host__ __device__ inline bool check_bound(const T &x, const T&min, const T &max) {
    return ((x < min) || (x > max));
}

// These are typically host-only unless explicitly called from a kernel
inline int vlib_round(float x) { return (int)(x + 0.5F); }
inline int vlib_round(double x) { return (int)(x + 0.5); }


// gaussian calls square. Since square is now __host__ __device__, gaussian can also be __host__ __device__
// if it needs to be called from a kernel. If it's only called from host code (like in filter.h's make_fgauss),
// then it can remain a host-only function. However, to be safe if you move make_fgauss to device later,
// or if gaussian is used elsewhere on device, it's good to add __host__ __device__ now.
__host__ __device__ inline double gaussian(double val, double sigma) {
    return exp(-square(val/sigma)/2)/(sqrt(2*M_PI)*sigma);
}

// Macro for CUDA error checking (no change needed here)
#define cudaCheckError() {                                                                      \
    cudaError_t err = cudaGetLastError();                                                      \
    if (err != cudaSuccess) {                                                                  \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                                    \
    }                                                                                           \
}

#endif