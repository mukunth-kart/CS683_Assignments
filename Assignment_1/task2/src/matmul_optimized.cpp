// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>
#include <algorithm>
#include "matmul.h"

#define PD 16

const int T = 32;
void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your best combined implementation.
    std::fill(C, C + (M * N), 0.0f);
    for(int i=0; i<M; i+=T)
    {
        for(int j=0; j<N; j+=T)
        {
            for(int k=0; k<K; k+=T)
            {
                // TxT mini matrix multiplication
                for(int i1=i; i1<i+T; i1+=PD)
                {

                    _mm_prefetch(&A[static_cast<long>(i1) * lda], _MM_HINT_T0);
                    for(int i2=i1; i2<i1+PD; i2++)
                    {

                    for(int j1=j; j1<j+T; j1+=PD)
                    {
                        
                        
                        // const float* a = A + static_cast<long>(i1) * lda;
                        // const float* b = B + static_cast<long>(j1) * ldb;
                        _mm_prefetch(&B[static_cast<long>(j1) * ldb], _MM_HINT_T0);
                        for(int j2=j1; j2<j1+PD; j2++)
                        {
                            __m256 acc = _mm256_setzero_ps();
                            for (int k1 = k; k1 < k+T; k1+=8) {
                                __m256 a = _mm256_load_ps(&A[static_cast<long>(i2) * lda + k1]);
                                __m256 b = _mm256_load_ps(&B[static_cast<long>(j2) * ldb + k1]);

                                acc = _mm256_fmadd_ps(a, b, acc); //A[static_cast<long>(i1) * lda + p] * B[static_cast<long>(j1) * ldb + p];
                            }
                            float temp[8];

                            _mm256_store_ps(temp, acc);

                            float total_sum = temp[0] + temp[1] + temp[2] + temp[3] + 
                                            temp[4] + temp[5] + temp[6] + temp[7];

                            C[static_cast<long>(i2) * ldc + j2] += total_sum;
                        }
                    }
                    }
                }
            }
        }
    }
    // matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}
