#ifndef FP8_MOE_H_
#define FP8_MOE_H_

#include <stdint.h>

#if defined(_WIN32)
#define FP8_MOE_API __declspec(dllexport)
#else
#define FP8_MOE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Compute y = weights[N,K] @ x[K]. Weights are float8_e4m3fn bytes and scales
// are ue8m0 bytes laid out as [ceil(N/128), ceil(K/128)]. Output is float32.
//
// num_threads == 0 selects a conservative physical-core estimate. A positive
// value fixes the worker count. Returns zero on success and a negative value for
// invalid arguments.
FP8_MOE_API int fp8_moe_gemv(const uint8_t* weights, const uint8_t* scales,
                             const float* x, float* y, int64_t n, int64_t k,
                             int num_threads);

// The portable reference implementation. It uses the same persistent worker
// pool when num_threads > 1, but never executes SIMD-only code.
FP8_MOE_API int fp8_moe_gemv_scalar(const uint8_t* weights,
                                    const uint8_t* scales, const float* x,
                                    float* y, int64_t n, int64_t k,
                                    int num_threads);

// Name of the implementation selected by runtime CPU feature detection.
FP8_MOE_API const char* fp8_moe_isa_name(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // FP8_MOE_H_
