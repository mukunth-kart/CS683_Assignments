// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
    const int p = K/2;
    const int in_stride = W+2*p; //padded row stride
    const int B=1024;
    int oy=0,ox=0;    
    for(oy=0;oy<H;oy+=B){
    	for(ox=0;ox<W;ox+=B){
            const int y_end = oy + B > H? H : oy + B;
            const int x_end = ox + B > H? H : ox + B;
    		for (int y = oy; y < y_end  ; y++) {
                for (int x = ox; x < x_end ; x++) {
                    float acc = 0.0f;
                    for (int ky = 0; ky < K; ky++) {
                        for (int kx = 0; kx < K; kx++) {
                            acc += in[(y + ky) * in_stride + (x + kx)] * ker[ky * K + kx];
                        }
                    }

                    out[y * W + x] = acc;
               }
    		}
    	}
    }
}