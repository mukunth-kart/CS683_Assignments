#include <immintrin.h>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    constexpr int TILE_H = 16;
    constexpr int TILE_W = 64;

    for (int ty = 0; ty < H; ty += TILE_H) {
        int ty_end = (ty + TILE_H < H) ? ty + TILE_H : H;

        for (int tx = 0; tx < W; tx += TILE_W) {
            int tx_end = (tx + TILE_W < W) ? tx + TILE_W : W;

            for (int oy = ty; oy < ty_end; ++oy) {
                float* out_row = &out[oy * W];
                int ox = tx;

                // 4 Unrolled
                for (; ox <= tx_end - 32; ox += 32) {
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();

                    for (int ky = 0; ky < K; ++ky) {
                        const float* in_ptr = &in[(oy + ky) * in_stride + ox];
                        const float* ker_ptr = &ker[ky * K];

                        for (int kx = 0; kx < K; ++kx) {
                            __m256 v_ker = _mm256_set1_ps(ker_ptr[kx]);

                            __m256 in0 = _mm256_loadu_ps(in_ptr + kx + 0);
                            __m256 in1 = _mm256_loadu_ps(in_ptr + kx + 8);
                            __m256 in2 = _mm256_loadu_ps(in_ptr + kx + 16);
                            __m256 in3 = _mm256_loadu_ps(in_ptr + kx + 24);

                            acc0 = _mm256_fmadd_ps(in0, v_ker, acc0);
                            acc1 = _mm256_fmadd_ps(in1, v_ker, acc1);
                            acc2 = _mm256_fmadd_ps(in2, v_ker, acc2);
                            acc3 = _mm256_fmadd_ps(in3, v_ker, acc3);
                        }
                    }

                    _mm256_storeu_ps(out_row + ox + 0, acc0);
                    _mm256_storeu_ps(out_row + ox + 8, acc1);
                    _mm256_storeu_ps(out_row + ox + 16, acc2);
                    _mm256_storeu_ps(out_row + ox + 24, acc3);
                }

                //3 Unrolled Fallback
                for (; ox <= tx_end - 24; ox += 24) {
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();

                    for (int ky = 0; ky < K; ++ky) {
                        const float* in_ptr = &in[(oy + ky) * in_stride + ox];
                        const float* ker_ptr = &ker[ky * K];

                        for (int kx = 0; kx < K; ++kx) {
                            __m256 v_ker = _mm256_set1_ps(ker_ptr[kx]);

                            __m256 in0 = _mm256_loadu_ps(in_ptr + kx + 0);
                            __m256 in1 = _mm256_loadu_ps(in_ptr + kx + 8);
                            __m256 in2 = _mm256_loadu_ps(in_ptr + kx + 16);

                            acc0 = _mm256_fmadd_ps(in0, v_ker, acc0);
                            acc1 = _mm256_fmadd_ps(in1, v_ker, acc1);
                            acc2 = _mm256_fmadd_ps(in2, v_ker, acc2);
                        }
                    }

                    _mm256_storeu_ps(out_row + ox + 0, acc0);
                    _mm256_storeu_ps(out_row + ox + 8, acc1);
                    _mm256_storeu_ps(out_row + ox + 16, acc2);
                }

                //2 Unrolled Fallback
                for (; ox <= tx_end - 16; ox += 16) {
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();

                    for (int ky = 0; ky < K; ++ky) {
                        const float* in_ptr = &in[(oy + ky) * in_stride + ox];
                        const float* ker_ptr = &ker[ky * K];

                        for (int kx = 0; kx < K; ++kx) {
                            __m256 v_ker = _mm256_set1_ps(ker_ptr[kx]);

                            __m256 in0 = _mm256_loadu_ps(in_ptr + kx + 0);
                            __m256 in1 = _mm256_loadu_ps(in_ptr + kx + 8);

                            acc0 = _mm256_fmadd_ps(in0, v_ker, acc0);
                            acc1 = _mm256_fmadd_ps(in1, v_ker, acc1);
                        }
                    }

                    _mm256_storeu_ps(out_row + ox + 0, acc0);
                    _mm256_storeu_ps(out_row + ox + 8, acc1);
                }

                // Fallback
                for (; ox <= tx_end - 8; ox += 8) {
                    __m256 acc0 = _mm256_setzero_ps();

                    for (int ky = 0; ky < K; ++ky) {
                        const float* in_ptr = &in[(oy + ky) * in_stride + ox];
                        const float* ker_ptr = &ker[ky * K];

                        for (int kx = 0; kx < K; ++kx) {
                            __m256 v_ker = _mm256_set1_ps(ker_ptr[kx]);
                            __m256 in0 = _mm256_loadu_ps(in_ptr + kx);

                            acc0 = _mm256_fmadd_ps(in0, v_ker, acc0);
                        }
                    }

                    _mm256_storeu_ps(out_row + ox, acc0);
                }
            }
        }
    }
}