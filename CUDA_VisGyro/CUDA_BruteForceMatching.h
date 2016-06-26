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

#ifndef __CUDA_BRUTE_FORCE_MATCHING__
#define __CUDA_BRUTE_FORCE_MATCHING__

#include <vector>
#include "DataTypes.h"

using namespace std;

// Finds matching points in src and dst using simple SAD block matching.
// The output is stored in match_index. Its size is the same as kp1, and each
// value is an index to the best match in kp2.

// ASSUMPTIONS
// - img1/img2 are expected to be grey level images
// - w1/w2 is the same value as the row stride
void CUDA_BruteForceMatching(const vector <Point2Df> &kp1, const vector <Point2Df> &kp2,
                             unsigned char *img1, int w1, int h1,
                             unsigned char *img2, int w2, int h2,
                             int block_size,
                             vector <int> *match_index,
                             vector <float> *match_score);

#endif
