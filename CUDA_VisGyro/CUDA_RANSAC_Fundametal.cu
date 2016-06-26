//from Nghia Ho website which presents a code for CUDA RANSAC Homography
#include "CUDA_RANSAC_Fundamental.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <numeric>
#include <omp.h>
#include <assert.h>
#include "CUDA_SVD.cu"

static const int NTHREADS = 512; // threads per block

#define SQ(x) (x)*(x)

static void CheckCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

__device__ int CalcFundamental(const Point2Df src[8], const Point2Df dst[8], float ret_F[9])
{
    // This version does not normalised the input data, which is contrary to what Multiple View Geometry says.
    // I included it to see what happens when you don't do this step.

    float X[M*N]; // M,N #define inCUDA_SVD.cu

    for(int i=0; i < 8; i++) {
        float srcx = src[i].x;
        float srcy = src[i].y;
        float dstx = dst[i].x;
        float dsty = dst[i].y;

        int y1 = i * N;

        // First row
        X[y1+0] = dstx*srcx;
        X[y1+1] = dstx*srcy;
        X[y1+2] = dstx;

        X[y1+3] = dsty*srcx;
        X[y1+4] = dsty*srcy;
        X[y1+5] = dsty;

        X[y1+6] = srcx;
        X[y1+7] = srcy;
        X[y1+8] = 1;

    }

    // Fill the last row


    int y = 8*N;
    X[y+0] = 0;
    X[y+1] = 0;
    X[y+2] = 0;

    X[y+3] = 0;
    X[y+4] = 0;
    X[y+5] = 0;

    X[y+6] = 0;
    X[y+7] = 0;
    X[y+8] = 0;

    float w[N];
    float v[N*N];

    int ret = dsvd(X, M, N, w, v);

    if(ret == 1) {
        // Sort
        float smallest = w[0];
        int col = 0;

        for(int i=1; i < N; i++) {
            if(w[i] < smallest) {
                smallest = w[i];
                col = i;
            }
        }

        ret_F[0] = v[0*N + col];
        ret_F[1] = v[1*N + col];
        ret_F[2] = v[2*N + col];
        ret_F[3] = v[3*N + col];
        ret_F[4] = v[4*N + col];
        ret_F[5] = v[5*N + col];
        ret_F[6] = v[6*N + col];
        ret_F[7] = v[7*N + col];
        ret_F[8] = v[8*N + col];
    }

    return ret;
}




__device__ int EvalFundamental(const Point2Df *src, const Point2Df *dst, int pts_num, const float F[9], float inlier_threshold)
{
    int inliers = 0;



  // extract fundamental matrix
  double f00 = F[0]; double f01 = F[1]; double f02 = F[2];
  double f10 = F[3]; double f11 = F[4]; double f12 = F[5];
  double f20 = F[6]; double f21 = F[7]; double f22 = F[8];

  // loop variables
  double u1,v1,u2,v2;
  double x2tFx1;
  double Fx1u,Fx1v,Fx1w;
  double Ftx2u,Ftx2v;

  // vector with inliers
  //vector<int32_t> inliers;

  // for all matches do
  for (int32_t i=0; i<(int32_t)pts_num; i++) {

    // extract matches
    u1 = src[i].x;
    v1 = src[i].y;
    u2 = dst[i].x;
    v2 = dst[i].y;

    // F*x1
    Fx1u = f00*u1+f01*v1+f02;
    Fx1v = f10*u1+f11*v1+f12;
    Fx1w = f20*u1+f21*v1+f22;

    // F'*x2
    Ftx2u = f00*u2+f10*v2+f20;
    Ftx2v = f01*u2+f11*v2+f21;

    // x2'*F*x1
    x2tFx1 = u2*Fx1u+v2*Fx1v+Fx1w;

    // sampson distance
    double d = x2tFx1*x2tFx1 / (Fx1u*Fx1u+Fx1v*Fx1v+Ftx2u*Ftx2u+Ftx2v*Ftx2v);

    // check threshold
    if(d<0)
        d=-d;
    if (d<inlier_threshold)
      inliers++;//inliers.push_back(i);
  }

//    for(int i=0; i < pts_num; i++) {
//        float x = H[0]*src[i].x + H[1]*src[i].y + H[2];
//        float y = H[3]*src[i].x + H[4]*src[i].y + H[5];
//        float z = H[6]*src[i].x + H[7]*src[i].y + H[8];
//
//        x /= z;
//        y /= z;
//
//        float dist_sq = (dst[i].x - x)*(dst[i].x- x) + (dst[i].y - y)*(dst[i].y - y);
//
//        if(dist_sq < inlier_threshold) {
//            inliers++;
//        }
//    }

    return inliers;
}

__global__ void RANSAC_Fundamental(const Point2Df *src, const Point2Df *dst,int pts_num, const int *rand_list, float inlier_threshold, int iterations, int *ret_inliers, float *ret_fundamental)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx >= iterations) {
        return;
    }

    ret_inliers[idx] = 0;

    int rand_idx[8];
    Point2Df _src[8];
    Point2Df _dst[8];
    float *F = &ret_homography[idx*9];

    rand_idx[0] = rand_list[idx*8];
    rand_idx[1] = rand_list[idx*8+1];
    rand_idx[2] = rand_list[idx*8+2];
    rand_idx[3] = rand_list[idx*8+3];
    rand_idx[4] = rand_list[idx*8+4];
    rand_idx[5] = rand_list[idx*8+5];
    rand_idx[6] = rand_list[idx*8+6];
    rand_idx[7] = rand_list[idx*8+7];

    // Check for duplicates
    if(rand_idx[0] == rand_idx[1]) return;
    if(rand_idx[0] == rand_idx[2]) return;
    if(rand_idx[0] == rand_idx[3]) return;
    if(rand_idx[0] == rand_idx[4]) return;
    if(rand_idx[0] == rand_idx[5]) return;
    if(rand_idx[0] == rand_idx[6]) return;
    if(rand_idx[0] == rand_idx[7]) return;
    if(rand_idx[1] == rand_idx[2]) return;
    if(rand_idx[1] == rand_idx[3]) return;
    if(rand_idx[1] == rand_idx[4]) return;
    if(rand_idx[1] == rand_idx[5]) return;
    if(rand_idx[1] == rand_idx[6]) return;
    if(rand_idx[1] == rand_idx[7]) return;
    if(rand_idx[2] == rand_idx[3]) return;
    if(rand_idx[2] == rand_idx[4]) return;
    if(rand_idx[2] == rand_idx[5]) return;
    if(rand_idx[2] == rand_idx[6]) return;
    if(rand_idx[2] == rand_idx[7]) return;
    if(rand_idx[3] == rand_idx[4]) return;
    if(rand_idx[3] == rand_idx[5]) return;
    if(rand_idx[3] == rand_idx[6]) return;
    if(rand_idx[3] == rand_idx[7]) return;
    if(rand_idx[4] == rand_idx[5]) return;
    if(rand_idx[4] == rand_idx[6]) return;
    if(rand_idx[4] == rand_idx[7]) return;
    if(rand_idx[5] == rand_idx[6]) return;
    if(rand_idx[5] == rand_idx[7]) return;
    if(rand_idx[6] == rand_idx[7]) return;

    for(int i=0; i < 8; i++) {
        _src[i].x = src[rand_idx[i]].x;
        _src[i].y = src[rand_idx[i]].y;
        _dst[i].x = dst[rand_idx[i]].x;
        _dst[i].y = dst[rand_idx[i]].y;
    }

#ifdef NORMALISE_INPUT_POINTS
    int ret = CalcHomography2(_src, _dst, H);
#else
    int ret = CalcFundamental(_src, _dst, F);
#endif

    ret_inliers[idx] = EvalFundamental(src, dst, pts_num, F, inlier_threshold);
}

void CUDA_RANSAC_Fundamental(const vector <Point2Df> &src, const vector <Point2Df> &dst, const vector <float> &match_score,
                            float RANSAC_threshold, int iterations,
                            int *best_inliers, float *best_F, vector <char> *inlier_mask)
{
    assert(src.size() == dst.size());
    assert(match_score.size() == dst.size());

    //int RANSAC_threshold = inlier_threshold*inlier_threshold;
    int threads = NTHREADS;
    int blocks = iterations/threads + ((iterations % threads)?1:0);

    Point2Df *gpu_src;
    Point2Df *gpu_dst;
    int *gpu_rand_list;
    int *gpu_ret_inliers;
    float *gpu_ret_F;
    vector <int> rand_list(iterations*8);
    vector <int> ret_inliers(iterations);
    vector <float> ret_F(iterations*8);

    cudaMalloc((void**)&gpu_src, sizeof(Point2Df)*src.size());
    cudaMalloc((void**)&gpu_dst, sizeof(Point2Df)*dst.size());
    cudaMalloc((void**)&gpu_rand_list, sizeof(int)*iterations*8);
    cudaMalloc((void**)&gpu_ret_inliers, sizeof(int)*iterations);
    cudaMalloc((void**)&gpu_ret_F, sizeof(float)*iterations*9);
    CheckCUDAError("cudaMalloc");

    cudaMemcpy(gpu_src, &src[0], sizeof(Point2Df)*src.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dst, &dst[0], sizeof(Point2Df)*dst.size(), cudaMemcpyHostToDevice);

    // Generate random numbers on host
    // Using a bias version when randomly selecting points
    // Point with better matching score have a highr chance of getting picked
    {
#ifdef BIAS_RANDOM_SELECTION
        vector <float> cummulative = match_score;
        double sum = accumulate(match_score.begin(), match_score.end(), 0.0);

        // Normalise the scores
        for(unsigned int i=0; i < cummulative.size(); i++) {
            cummulative[i] /= sum;
        }

        // Calc the cummulative distribution
        for(unsigned int i=1; i < cummulative.size(); i++) {
            cummulative[i] += cummulative[i-1];
        }

        for(unsigned int i=0; i < rand_list.size(); i++) {
            float x = rand()/(1.0 + RAND_MAX); // random between [0,1)

            // Binary search to find which index x lands on
            int min = 0;
            int max = src.size();
            int index = 0;

            while(true) {
                int mid = (min + max) / 2;

                if(min == max - 1) {
                    if(x < cummulative[min]) {
                        index = min;
                    }
                    else {
                        index = max;
                    }
                    break;
                }

                if(x > cummulative[mid]) {
                    min = mid;
                }
                else {
                    max = mid;
                }
            }

            rand_list[i] = index;
        }
#else
        for(unsigned int i=0; i < rand_list.size(); i++) {
            rand_list[i] = src.size() * (rand()/(1.0 + RAND_MAX));
        }
#endif
        cudaMemcpy(gpu_rand_list, &rand_list[0], sizeof(int)*rand_list.size(), cudaMemcpyHostToDevice);
        CheckCUDAError("cudaMemcpy");
    }

    RANSAC_Homography<<<blocks, threads>>>(gpu_src, gpu_dst, src.size(), gpu_rand_list, RANSAC_threshold, iterations, gpu_ret_inliers, gpu_ret_F);
    cudaThreadSynchronize();
    CheckCUDAError("RANSAC_Homography");

    cudaMemcpy(&ret_inliers[0], gpu_ret_inliers, sizeof(int)*ret_inliers.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ret_F[0], gpu_ret_F, sizeof(float)*ret_F.size(), cudaMemcpyDeviceToHost);

    *best_inliers = 0;
    int best_idx = 0;

    for(int i=0; i < ret_inliers.size(); i++) {
        /*
        printf("ret %d: %d\n", i, ret_inliers[i]);

        for(int j=0; j< 9; j++) {
            printf("%.3f ",  ret_H[i*9+j]);
        }
        printf("\n");
        */
        if(ret_inliers[i] > *best_inliers) {
            *best_inliers = ret_inliers[i];
            best_idx = i;
        }
    }

    memcpy(best_F, &ret_F[best_idx*9], sizeof(float)*9);

    // Fill the mask
    vector <char> &_inlier_mask = *inlier_mask;
    _inlier_mask.resize(src.size(), 0);
    // extract fundamental matrix
    double f00 = best_F[0]; double f01 = best_F[1]; double f02 = best_F[2];
    double f10 = best_F[3]; double f11 = best_F[4]; double f12 = best_F[5];
    double f20 = best_F[6]; double f21 = best_F[7]; double f22 = best_F[8];

    // loop variables
    double u1,v1,u2,v2;
    double x2tFx1;
    double Fx1u,Fx1v,Fx1w;
    double Ftx2u,Ftx2v;
    for(int i=0; i < src.size(); i++) {
         // extract matches
        u1 = src[i].x;
        v1 = src[i].y;
        u2 = dst[i].x;
        v2 = dst[i].y;

        // F*x1
        Fx1u = f00*u1+f01*v1+f02;
        Fx1v = f10*u1+f11*v1+f12;
        Fx1w = f20*u1+f21*v1+f22;

        // F'*x2
        Ftx2u = f00*u2+f10*v2+f20;
        Ftx2v = f01*u2+f11*v2+f21;

        // x2'*F*x1
        x2tFx1 = u2*Fx1u+v2*Fx1v+Fx1w;

        // sampson distance
        double d = x2tFx1*x2tFx1 / (Fx1u*Fx1u+Fx1v*Fx1v+Ftx2u*Ftx2u+Ftx2v*Ftx2v);

        // check threshold
        if(d<0)
            d=-d;
        if (d<inlier_threshold)
          _inlier_mask[i] = 1;
    }

    *best_inliers = accumulate(_inlier_mask.begin(), _inlier_mask.end(), 0);

    //printf("CUDA blocks/threads: %d %d\n", blocks, threads);

    cudaFree(gpu_src);
    cudaFree(gpu_dst);
    cudaFree(gpu_rand_list);
    cudaFree(gpu_ret_inliers);
    cudaFree(gpu_ret_F);
}
