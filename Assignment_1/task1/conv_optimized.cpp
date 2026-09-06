// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Measure after every change -- not every "optimization" helps on every machine
// a few values and keeping what was fastest, not by intuition alone.

#include <immintrin.h>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                     int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int B = 128;
    int r;
    
    for(int ky=0;ky<K;++ky){
        for(int kx=0;kx<K;++kx){
            r = ker[ky*K+kx];
            for(int oy=0;oy<H;++oy){
                for(int ox=0;ox<W;++ox){
                    const int y_end = oy + B > H? H : oy + B;
                    const int x_end = ox + B > H? H : ox + B;
                    for (int y = oy; y < y_end  ; y++) {
                        for (int x = ox; x < x_end ; x++) {
                            out[oy*W+ox]+=in[(oy+ky)*in_stride + (ox+kx)]*r;
                        }
                    }
                }
            }
        }
    }

    // for(int ky=0;ky<K;++ky){
    //     for(int oy=0;oy<H;++oy){
    //         for(int kx=0;kx<K;++kx){
    //             r = ker[ky*K+kx];
    //             for(int ox=0;ox<W;++ox){
    //                 out[oy*W + ox]+=in[(oy+ky)*in_stride + (ox+kx)]*r;
    //             }
    //         }
    //     }
    // }
}
