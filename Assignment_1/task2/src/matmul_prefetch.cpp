// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>
#include <algorithm>
#include "matmul.h"

#define PREFETCH_DIST 256

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) 
{
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.
    for (int i = 0; i < M; i++) {
        
        for (int j = 0; j < N; j++) {
            
            float acc = 0.0f;
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;
            for (int p = 0; p < K; ++p) {

                _mm_prefetch(reinterpret_cast<const char*>(&a[p + PREFETCH_DIST]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(&b[p + PREFETCH_DIST]), _MM_HINT_T0);

                acc += a[p] * b[p];
            }
            C[static_cast<long>(i) * ldc + j] = acc;
        
        }
    }
    // matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}
