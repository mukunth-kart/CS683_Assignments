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
                for(int i1=i; i1<i+T; i1++)
                {
                    for(int j1=j; j1<j+T; j1++)
                    {
                        float acc = 0.0f;
                        // const float* a = A + static_cast<long>(i1) * lda;
                        // const float* b = B + static_cast<long>(j1) * ldb;
                        for (int p = k; p < k+T; p++) {
                            acc += A[static_cast<long>(i1) * lda + p] * B[static_cast<long>(j1) * ldb + p];
                        }
                        C[static_cast<long>(i1) * ldc + j1] += acc;
                    }
                }
            }
        }
    }
    // matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}
