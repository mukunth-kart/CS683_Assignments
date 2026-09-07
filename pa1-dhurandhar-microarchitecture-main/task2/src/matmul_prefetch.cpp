// // matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>
#include "matmul.h"

#ifndef PD
#define PD 64 // default to 64 if not provided by Python
#endif

#ifndef FL
#define FL 0  // default to 0 if not provided by Python
#endif

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) 
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;
            
            int p = 0;
            for (; p <= K - 16; p += 16) {
                __builtin_prefetch(&a[p + PD],0,FL);
                __builtin_prefetch(&b[p + PD],0,FL);
                for (int pp = 0; pp < 16; ++pp) {
                    acc += a[p + pp] * b[p + pp];
                }
            }
            for (; p < K; ++p) {
                acc += a[p] * b[p];
            }
            
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}

