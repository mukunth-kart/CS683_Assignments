// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"
#include<iostream>

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    for (int oy = 0; oy < H; ++oy) {
	int ox=0;
        for (; ox < W-2; ox+=3) {
            float acc = 0.0f;
	    int ky=0;
            for (; ky < K-2; ky+=3) {
		int kx=0;
                for (; kx < K-2; kx+=3) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
		    acc += in[(oy + ky) * in_stride + (ox + kx+1)] * ker[ky * K + kx+1];
		    acc += in[(oy + ky) * in_stride + (ox + kx+2)] * ker[ky * K + kx+2];
		}
                for(;kx<K;++kx)
		    acc+= in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
		kx=0;
                for (; kx < K-2; kx+=3) {
                    acc += in[(oy + ky+1) * in_stride + (ox + kx)] * ker[(ky+1) * K + kx];
		    acc += in[(oy + ky+1) * in_stride + (ox + kx+1)] * ker[(ky+1) * K + kx+1];
		    acc += in[(oy + ky+1) * in_stride + (ox + kx+2)] * ker[(ky+1) * K + kx+2];
		}
                for(;kx<K;++kx)
		    acc+= in[(oy + ky+1) * in_stride + (ox + kx)] * ker[(ky+1) * K + kx];
		kx=0;
                for (; kx < K-2; kx+=3) {
                    acc += in[(oy + ky+2) * in_stride + (ox + kx)] * ker[(ky+2) * K + kx];
		    acc += in[(oy + ky+2) * in_stride + (ox + kx+1)] * ker[(ky+2) * K + kx+1];
		    acc += in[(oy + ky+2) * in_stride + (ox + kx+2)] * ker[(ky+2) * K + kx+2];
		}
                for(;kx<K;++kx)
		    acc+= in[(oy + ky+2) * in_stride + (ox + kx)] * ker[(ky+2) * K + kx];   
	    }
            for (; ky < K; ky++) {
		int kx=0;
                for (; kx < K-2; kx+=3) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
		    acc += in[(oy + ky) * in_stride + (ox + kx+1)] * ker[ky * K + kx+1];
		    acc += in[(oy + ky) * in_stride + (ox + kx+2)] * ker[ky * K + kx+2];
		}
                for(;kx<K;++kx)
		    acc+= in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
            }
            out[oy * W + ox] = acc;

            acc = 0.0f;
	    ky=0;
            for (; ky < K-2; ky+=3) {
		int kx=0;
                for (; kx < K-2; kx+=3) {
                    acc += in[(oy + ky) * in_stride + (ox+1 + kx)] * ker[ky * K + kx];
		    acc += in[(oy + ky) * in_stride + (ox+1 + kx+1)] * ker[ky * K + kx+1];
		    acc += in[(oy + ky) * in_stride + (ox+1 + kx+2)] * ker[ky * K + kx+2];
		}
                for(;kx<K;++kx)
		    acc+= in[(oy + ky) * in_stride + (ox+1 + kx)] * ker[ky * K + kx];
		kx=0;
                for (; kx < K-2; kx+=3) {
                    acc += in[(oy + ky+1) * in_stride + (ox+1 + kx)] * ker[(ky+1) * K + kx];
		    acc += in[(oy + ky+1) * in_stride + (ox+1 + kx+1)] * ker[(ky+1) * K + kx+1];
		    acc += in[(oy + ky+1) * in_stride + (ox+1 + kx+2)] * ker[(ky+1) * K + kx+2];
		}
                for(;kx<K;++kx)
		    acc+= in[(oy + ky+1) * in_stride + (ox+1 + kx)] * ker[(ky+1) * K + kx];
		kx=0;
                for (; kx < K-2; kx+=3) {
                    acc += in[(oy + ky+2) * in_stride + (ox+1 + kx)] * ker[(ky+2) * K + kx];
		    acc += in[(oy + ky+2) * in_stride + (ox+1 + kx+1)] * ker[(ky+2) * K + kx+1];
		    acc += in[(oy + ky+2) * in_stride + (ox+1 + kx+2)] * ker[(ky+2) * K + kx+2];
		}
                for(;kx<K;++kx)
		    acc+= in[(oy + ky+2) * in_stride + (ox+1 + kx)] * ker[(ky+2) * K + kx];   
	    }
            for (; ky < K; ky++) {
		int kx=0;
                for (; kx < K-2; kx+=3) {
                    acc += in[(oy + ky) * in_stride + (ox+1 + kx)] * ker[ky * K + kx];
		    acc += in[(oy + ky) * in_stride + (ox+1 + kx+1)] * ker[ky * K + kx+1];
		    acc += in[(oy + ky) * in_stride + (ox+1 + kx+2)] * ker[ky * K + kx+2];
		}
                for(;kx<K;++kx)
		    acc+= in[(oy + ky) * in_stride + (ox+1 + kx)] * ker[ky * K + kx];
            }
            out[oy * W + ox+1] = acc;

            acc = 0.0f;
	    ky=0;
            for (; ky < K-2; ky+=3) {
		int kx=0;
                for (; kx < K-2; kx+=3) {
                    acc += in[(oy + ky) * in_stride + (ox+2 + kx)] * ker[ky * K + kx];
		    acc += in[(oy + ky) * in_stride + (ox+2 + kx+1)] * ker[ky * K + kx+1];
		    acc += in[(oy + ky) * in_stride + (ox+2 + kx+2)] * ker[ky * K + kx+2];
		}
                for(;kx<K;++kx)
		    acc+= in[(oy + ky) * in_stride + (ox+2 + kx)] * ker[ky * K + kx];
		kx=0;
                for (; kx < K-2; kx+=3) {
                    acc += in[(oy + ky+1) * in_stride + (ox+2 + kx)] * ker[(ky+1) * K + kx];
		    acc += in[(oy + ky+1) * in_stride + (ox+2 + kx+1)] * ker[(ky+1) * K + kx+1];
		    acc += in[(oy + ky+1) * in_stride + (ox+2 + kx+2)] * ker[(ky+1) * K + kx+2];
		}
                for(;kx<K;++kx)
		    acc+= in[(oy + ky+1) * in_stride + (ox+2 + kx)] * ker[(ky+1) * K + kx];
		kx=0;
                for (; kx < K-2; kx+=3) {
                    acc += in[(oy + ky+2) * in_stride + (ox+2 + kx)] * ker[(ky+2) * K + kx];
		    acc += in[(oy + ky+2) * in_stride + (ox+2 + kx+1)] * ker[(ky+2) * K + kx+1];
		    acc += in[(oy + ky+2) * in_stride + (ox+2 + kx+2)] * ker[(ky+2) * K + kx+2];
		}
                for(;kx<K;++kx)
		    acc+= in[(oy + ky+2) * in_stride + (ox+2 + kx)] * ker[(ky+2) * K + kx];   
	    }
            for (; ky < K; ky++) {
		int kx=0;
                for (; kx < K-2; kx+=3) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
		    acc += in[(oy + ky) * in_stride + (ox + kx+1)] * ker[ky * K + kx+1];
		    acc += in[(oy + ky) * in_stride + (ox + kx+2)] * ker[ky * K + kx+2];
		}
                for(;kx<K;++kx)
		    acc+= in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
            }
            out[oy * W + ox] = acc;  
       	for(;ox<W;++ox){
            float acc = 0.0f;
	    int ky=0;
            for (; ky < K-2; ky+=3) {
		int kx=0;
                for (; kx < K-2; kx+=3) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
		    acc += in[(oy + ky) * in_stride + (ox + kx+1)] * ker[ky * K + kx+1];
		    acc += in[(oy + ky) * in_stride + (ox + kx+2)] * ker[ky * K + kx+2];
		}
                for(;kx<K;++kx)
		    acc+= in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
		kx=0;
                for (; kx < K-2; kx+=3) {
                    acc += in[(oy + ky+1) * in_stride + (ox + kx)] * ker[(ky+1) * K + kx];
		    acc += in[(oy + ky+1) * in_stride + (ox + kx+1)] * ker[(ky+1) * K + kx+1];
		    acc += in[(oy + ky+1) * in_stride + (ox + kx+2)] * ker[(ky+1) * K + kx+2];
		}
                for(;kx<K;++kx)
		    acc+= in[(oy + ky+1) * in_stride + (ox + kx)] * ker[(ky+1) * K + kx];
		kx=0;
                for (; kx < K-2; kx+=3) {
                    acc += in[(oy + ky+2) * in_stride + (ox + kx)] * ker[(ky+2) * K + kx];
		    acc += in[(oy + ky+2) * in_stride + (ox + kx+1)] * ker[(ky+2) * K + kx+1];
		    acc += in[(oy + ky+2) * in_stride + (ox + kx+2)] * ker[(ky+2) * K + kx+2];
		}
                for(;kx<K;++kx)
		    acc+= in[(oy + ky+2) * in_stride + (ox + kx)] * ker[(ky+2) * K + kx];   
	    }
            for (; ky < K; ky++) {
		int kx=0;
                for (; kx < K-2; kx+=3) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
		    acc += in[(oy + ky) * in_stride + (ox + kx+1)] * ker[ky * K + kx+1];
		    acc += in[(oy + ky) * in_stride + (ox + kx+2)] * ker[ky * K + kx+2];
		}
                for(;kx<K;++kx)
		    acc+= in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
            }
            out[oy * W + ox] = acc;  
	}	    
        }
    }
}