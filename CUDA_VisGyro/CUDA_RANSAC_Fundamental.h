

#ifndef __CUDA_RANSAC_Fundamental_H__
#define __CUDA_RANSAC_Fundamental_H__

#include <vector>
#include "DataTypes.h"

using namespace std;


void CUDA_RANSAC_Fundamental(const vector <Point2Df> &src, const vector <Point2Df> &dst, const vector <float> &match_score,
                            float inlier_threshold, int iterations,
                            int *best_inliers, float *best_F, vector <char> *inlier_mask);

#endif
