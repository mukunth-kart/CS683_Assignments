// conv_unroll.cpp  STAGE 2: LOOP UNROLLIN

#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        float* out_row = &out[oy * W];
        int ox = 0;

        for (; ox <= W - 8; ox += 8) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_ptr = &in[(oy + ky) * in_stride + ox];
                const float* ker_ptr = &ker[ky * K];

                for (int kx = 0; kx < K; ++kx) {
                    acc0 += in_ptr[kx + 0] * ker_ptr[kx];
                    acc1 += in_ptr[kx + 1] * ker_ptr[kx];
                    acc2 += in_ptr[kx + 2] * ker_ptr[kx];
                    acc3 += in_ptr[kx + 3] * ker_ptr[kx];
                    acc4 += in_ptr[kx + 4] * ker_ptr[kx];
                    acc5 += in_ptr[kx + 5] * ker_ptr[kx];
                    acc6 += in_ptr[kx + 6] * ker_ptr[kx];
                    acc7 += in_ptr[kx + 7] * ker_ptr[kx];
                }
            }

            out_row[ox + 0] = acc0;
            out_row[ox + 1] = acc1;
            out_row[ox + 2] = acc2;
            out_row[ox + 3] = acc3;
            out_row[ox + 4] = acc4;
            out_row[ox + 5] = acc5;
            out_row[ox + 6] = acc6;
            out_row[ox + 7] = acc7;
        }
    }
}