/*
Copyright 2011 Nghia Ho. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY NGHIA HO ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL BY NGHIA HO OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Nghia Ho.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <limits.h>
#include <cstdio>
#include "CUDA_BruteForceMatching.h"

using namespace std;

static const int NTHREADS = 512; // threads per block

static void CheckCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

__global__ void BruteForceMatching(unsigned char *patches1, char *patches_used1, int size1,
                                   unsigned char *patches2, char *patches_used2, int size2,
                                   int block_size_sq, int *ret_match_index, int *ret_match_score)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx >= size1) {
        return;
    }

    ret_match_index[idx] = -1;
    int best = INT_MAX;
    int best_idx = -1;

    if(!patches_used1[idx]) {
        return;
    }

    for(int i=0; i < size2; i++) {
        if(!patches_used2[i]) {
            continue;
        }

        int SAD = 0;

        for(int j=0; j < block_size_sq; j++) {
            SAD += abs(patches1[idx*block_size_sq + j] - patches2[i*block_size_sq + j]);
        }

        if(SAD < best) {
            best = SAD;
            best_idx = i;
        }
    }

    ret_match_index[idx] = best_idx;
    ret_match_score[idx] = best;
}

void CUDA_BruteForceMatching(const vector <Point2Df> &kp1, const vector <Point2Df> &kp2,
                             unsigned char *img1, int w1, int h1,
                             unsigned char *img2, int w2, int h2,
                             int block_size,
                             vector <int> *ret_match_index,
                             vector <float> *ret_match_score)
{
    // Extract patches from img1/img2 and load onto the GPU for matching.
    unsigned char *gpu_patches1;
    unsigned char *gpu_patches2;
    char *gpu_patches_used1;
    char *gpu_patches_used2;
    int *gpu_match_index;
    int *gpu_match_score;

    int block_size2 = block_size/2;
    int block_size_sq = block_size*block_size;

    vector <unsigned char> patches1(kp1.size()*block_size_sq);
    vector <unsigned char> patches2(kp2.size()*block_size_sq);
    vector <char> patches_used1(kp1.size(), 0);
    vector <char> patches_used2(kp2.size(), 0);
    vector <int> match_score(kp1.size());

    for(unsigned int i=0; i < kp1.size(); i++) {
        int x = kp1[i].x;
        int y = kp1[i].y;

        // Make sure the patch does not fall outside the image
        if(x - block_size2 < 0) continue;
        if(y - block_size2 < 0) continue;
        if(x + block_size2 >= w1) continue;
        if(y + block_size2 >= h1) continue;

        patches_used1[i] = 1;

        int k = 0;
        for(int yy=0; yy < block_size; yy++) {
            for(int xx=0; xx < block_size; xx++) {
                patches1[i*block_size_sq + k] = img1[(y-block_size2+yy)*w1 + x-block_size2+xx];
                k++;
            }
        }
    }

    for(unsigned int i=0; i < kp2.size(); i++) {
        int x = kp2[i].x;
        int y = kp2[i].y;

        // Make sure the patch does not fall outside the image
        if(x - block_size2 < 0) continue;
        if(y - block_size2 < 0) continue;
        if(x + block_size2 >= w2) continue;
        if(y + block_size2 >= h2) continue;

        patches_used2[i] = 1;

        int k = 0;
        for(int yy=0; yy < block_size; yy++) {
            for(int xx=0; xx < block_size; xx++) {
                patches2[i*block_size_sq + k] = img2[(y-block_size2+yy)*w2 + x-block_size2+xx];
                k++;
            }
        }
    }

    cudaMalloc((void**)&gpu_patches1, sizeof(unsigned char)*patches1.size());
    cudaMalloc((void**)&gpu_patches2, sizeof(unsigned char)*patches2.size());
    cudaMalloc((void**)&gpu_patches_used1, sizeof(char)*patches_used1.size());
    cudaMalloc((void**)&gpu_patches_used2, sizeof(char)*patches_used2.size());
    cudaMalloc((void**)&gpu_match_index, sizeof(int)*kp1.size());
    cudaMalloc((void**)&gpu_match_score, sizeof(int)*kp1.size());
    CheckCUDAError("cudaMalloc");

    cudaMemcpy(gpu_patches1, &patches1[0], sizeof(unsigned char)*patches1.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_patches2, &patches2[0], sizeof(unsigned char)*patches2.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_patches_used1, &patches_used1[0], sizeof(char)*patches_used1.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_patches_used2, &patches_used2[0], sizeof(char)*patches_used2.size(), cudaMemcpyHostToDevice);
    CheckCUDAError("cudaMemcpy");

    int threads = NTHREADS;
    int blocks = kp1.size()/threads + ((kp1.size() % threads)?1:0);

    BruteForceMatching<<<blocks, threads>>>(gpu_patches1, gpu_patches_used1, kp1.size(),
                                               gpu_patches2, gpu_patches_used2, kp2.size(),
                                               block_size_sq, gpu_match_index, gpu_match_score);
    cudaThreadSynchronize();
    CheckCUDAError("BruteForceMatching");

    vector <int> &_ret_match_index = *ret_match_index;
    vector <float> &_ret_match_score = *ret_match_score;

    _ret_match_index.resize(kp1.size());
    _ret_match_score.resize(kp1.size());

    cudaMemcpy(&_ret_match_index[0], gpu_match_index, sizeof(int)*kp1.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(&match_score[0], gpu_match_score, sizeof(int)*kp1.size(), cudaMemcpyDeviceToHost);

    // Transform the SAD score so that a low score is bad and high is good
    int smallest = *min_element(match_score.begin(), match_score.end());
    int biggest = *max_element(match_score.begin(), match_score.end());

    for(unsigned i=0; i < kp1.size(); i++) {
        _ret_match_score[i] = (biggest - match_score[i]) + smallest; // invert score;
    }

    cudaFree(gpu_patches1);
    cudaFree(gpu_patches2);
    cudaFree(gpu_patches_used1);
    cudaFree(gpu_patches_used2);
    cudaFree(gpu_match_index);
}
