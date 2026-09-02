// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // const float* a = A + static_cast<long>(i) * lda;
            // const float* b = B + static_cast<long>(j) * ldb;
            __m256 acc = _mm256_setzero_ps();
            // const float* a = A + static_cast<long>(i1) * lda;
            // const float* b = B + static_cast<long>(j1) * ldb;
            for (int k = 0; k < K; k+=8) {
                __m256 a = _mm256_load_ps(&A[static_cast<long>(i) * lda + k]);
                __m256 b = _mm256_load_ps(&B[static_cast<long>(j) * ldb + k]);

                acc = _mm256_fmadd_ps(a, b, acc); //A[static_cast<long>(i1) * lda + p] * B[static_cast<long>(j1) * ldb + p];
                
            }
            float temp[8];

            _mm256_store_ps(temp, acc);

            float total_sum = temp[0] + temp[1] + temp[2] + temp[3] + 
                            temp[4] + temp[5] + temp[6] + temp[7];

            C[static_cast<long>(i) * ldc + j] = total_sum;
        }
    }
    // matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}
