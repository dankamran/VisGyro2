/* 
Copyright (c) 2013, Jan Roters, University of Muenster, Germany
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*! \file
 * \brief In this file there are some helping methods for the estimation framework.
 */
#include <iostream>

using namespace std;

typedef enum
{
	EstimationTypeRANSAC,
	EstimationTypeLMS
} EstimationType;

typedef enum
{
	ResultValueTypeSuccessful,
	ResultValueTypeNoCUDADeviceFound,
	ResultValueTypeWrongParameters,
	ResultValueTypeEstimationNameUnknown
} ResultValueType;

typedef enum
{
	DuplicateCheckTypeNoCheck,
	DuplicateCheckTypeIndexBased,
	DuplicateCheckTypeIndexAndValueBased
} DuplicateCheckType;


/**
 * \brief Method to execute some simple CUDA calls to initialize the context.
 *
 * \returns A result state which is either that the framework has been initialized successfully,
 * or it is the specified error code of type ResultValueType.
 */
ResultValueType FestGPU_Initialize()
{
	int devCount = 0;
	cudaGetDeviceCount(&devCount);

	if (devCount < 1)
    {
        return ResultValueTypeNoCUDADeviceFound;
    }

	float *gpu_test;
    cudaMalloc(&gpu_test, sizeof(float));
    cudaFree(gpu_test);

    return ResultValueTypeSuccessful;
}

/*!
 * \brief Retrieves an error message (string) for a given state
 *
 * \param[in] state The result state of a previous call.
 * \returns The error message as string.
 */
string FestGPU_Get_Error_Message(ResultValueType state)
{
	if (ResultValueTypeEstimationNameUnknown)
		return "The given estimation is unknown!";
	else if (ResultValueTypeNoCUDADeviceFound)
		return "No CUDA device has been found!";
	else if (ResultValueTypeWrongParameters)
		return "Wrong parameters have been given!";
	else
		return "Everything seem to be fine ...";
}



static double TimeDiff(timeval t1, timeval t2)
{
    double t;
    t = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    t += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms

    return t;
}

static void CheckCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        cerr << "--------------------------------" << endl;
    	cerr << "Cuda error: " << endl;
        cerr << "  " << msg << ": " << cudaGetErrorString(err) << "." << endl;
        cerr << "--------------------------------" << endl;
        exit(EXIT_FAILURE);
    }
}

static void printMemoryUsage(int number)
{
	size_t mem_free, mem_total;
    cudaMemGetInfo(&mem_free, &mem_total);
    cout << "Device Memory " << number << ": " << mem_free/(1024.0f*1024.0f) << "mb of " << mem_total/(1024.0f*1024.0f) << "mb" << endl;
    CheckCUDAError("cudaMemGetInfo");
}

// Print device properties
static void printDeviceProperties(cudaDeviceProp devProp)
{
	cout << "Major revision number:         " << devProp.major << endl;
	cout << "Minor revision number:         " << devProp.minor << endl;
	cout << "Name:                          " << devProp.name << endl;
	cout << "Total global memory:           " << devProp.totalGlobalMem << endl;
	cout << "Total shared memory per block: " << devProp.sharedMemPerBlock << endl;
	cout << "Total registers per block:     " << devProp.regsPerBlock << endl;
	cout << "Warp size:                     " << devProp.warpSize << endl;
	cout << "Maximum memory pitch:          " << devProp.memPitch << endl;
	cout << "Maximum threads per block:     " << devProp.maxThreadsPerBlock << endl;
	for (int i = 0; i < 3; ++i)
		cout << "Maximum dimension " << i << " of block:  " << devProp.maxThreadsDim[i] << endl;
	for (int i = 0; i < 3; ++i)
		cout << "Maximum dimension " << i << " of grid:   " << devProp.maxGridSize[i] << endl;
	cout << "Clock rate:                    " << devProp.clockRate << endl;
	cout << "Total constant memory:         " << devProp.totalConstMem << endl;
	cout << "Texture alignment:             " << devProp.textureAlignment << endl;
	cout << "Concurrent copy and execution: " << (devProp.deviceOverlap ? "Yes" : "No") << endl;
	cout << "Number of multiprocessors:     " << devProp.multiProcessorCount << endl;
	cout << "Kernel execution timeout:      " << (devProp.kernelExecTimeoutEnabled ? "Yes" : "No") << endl;
    return;
}




#define INSERTION_SORT_BOUND 8 /* boundary point to use insertion sort */

__device__ int CUDA_Compare(float a, float b)
{
	return 2*(a>b)-1;
}


__device__ void CUDA_ELEM_SWAP_DEVICE(float *f1, float *f2)
{
	float tmp;
	tmp = f1[0];
	f1[0] = f2[0];
	f2[0] = tmp;
}

__device__ float CUDA_MEDIAN_DEVICE(float *arr, int n)
{
	int low, high ;
	int median;
	int middle, ll, hh;

	low = 0 ; high = n-1 ; median = (int)(0.5f*(low + high));
	for (;;) {
		if (high <= low) /* One element only */
			return arr[median] ;

		if (high == low + 1) {  /* Two elements only */
			if (arr[low] > arr[high])
				CUDA_ELEM_SWAP_DEVICE(&(arr[low]), &(arr[high])) ;
			return arr[median] ;
		}

		/* Find median of low, middle and high items; swap into position low */
		middle = (int)(0.5f*(low + high));
		if (arr[middle] > arr[high])    CUDA_ELEM_SWAP_DEVICE(&(arr[middle]), &(arr[high])) ;
		if (arr[low] > arr[high])       CUDA_ELEM_SWAP_DEVICE(&(arr[low]), &(arr[high])) ;
		if (arr[middle] > arr[low])     CUDA_ELEM_SWAP_DEVICE(&(arr[middle]), &(arr[low])) ;

		/* Swap low item (now in position middle) into position (low+1) */
		CUDA_ELEM_SWAP_DEVICE(&(arr[middle]), &(arr[low+1])) ;

		/* Nibble from each end towards middle, swapping items when stuck */
		ll = low + 1;
		hh = high;
		for (;;) {
			do ll++; while (arr[low] > arr[ll]) ;
			do hh--; while (arr[hh]  > arr[low]) ;

			if (hh < ll)
				break;

			CUDA_ELEM_SWAP_DEVICE(&(arr[ll]), &(arr[hh])) ;
		}

		/* Swap middle item (in position low) back into correct position */
		CUDA_ELEM_SWAP_DEVICE(&(arr[low]), &(arr[hh])) ;

		/* Re-set active partition */
		if (hh <= median)
			low = ll;
		if (hh >= median)
			high = hh - 1;
	}
}

__device__ void CUDA_SORT_DEVICE(float *This, int the_len)
{
  int span;
  int lb;
  int ub;
  int indx;
  int indx2;

  if (the_len <= 1)
    return;

  span = INSERTION_SORT_BOUND;

  /* insertion sort the first pass */
  {
    float prev_val;
    float cur_val;
    float temp_val;

    for (lb = 0; lb < the_len; lb += span)
    {
      if ((ub = lb + span) > the_len) ub = the_len;

      prev_val = This[lb];

      for (indx = lb + 1; indx < ub; ++indx)
      {
        cur_val = This[indx];

        if (CUDA_Compare(prev_val, cur_val) > 0)
        {
          /* out of order: array[indx-1] > array[indx] */
          This[indx] = prev_val; /* move up the larger item first */

          /* find the insertion point for the smaller item */
          for (indx2 = indx - 1; indx2 > lb;)
          {
            temp_val = This[indx2 - 1];
            if (CUDA_Compare(temp_val, cur_val) > 0)
            {
              This[indx2--] = temp_val;
              /* still out of order, move up 1 slot to make room */
            }
            else
              break;
          }
          This[indx2] = cur_val; /* insert the smaller item right here */
        }
        else
        {
          /* in order, advance to next element */
          prev_val = cur_val;
        }
      }
    }
  }

  /* second pass merge sort */
  {
    int median;
    float* aux = &(This[the_len]);

    //aux = (float*) malloc(sizeof(float) * the_len / 2);

    while (span < the_len)
    {
      /* median is the start of second file */
      for (median = span; median < the_len;)
      {
        indx2 = median - 1;
        if (CUDA_Compare(This[indx2], This[median]) > 0)
        {
          /* the two files are not yet sorted */
          if ((ub = median + span) > the_len)
          {
            ub = the_len;
          }

          /* skip over the already sorted largest elements */
          while (CUDA_Compare(This[--ub], This[indx2]) >= 0)
          {
          }

          /* copy second file into buffer */
          for (indx = 0; indx2 < ub; ++indx)
          {
            *(aux + indx) = This[++indx2];
          }
          --indx;
          indx2 = median - 1;
          lb = median - span;
          /* merge two files into one */
          for (;;)
          {
            if (CUDA_Compare(*(aux + indx), This[indx2]) >= 0)
            {
              This[ub--] = *(aux + indx);
              if (indx > 0) --indx;
              else
              {
                /* second file exhausted */
                for (;;)
                {
                  This[ub--] = This[indx2];
                  if (indx2 > lb) --indx2;
                  else goto mydone; /* done */
                }
              }
            }
            else
            {
              This[ub--] = This[indx2];
              if (indx2 > lb) --indx2;
              else
              {
                /* first file exhausted */
                for (;;)
                {
                  This[ub--] = *(aux + indx);
                  if (indx > 0) --indx;
                  else goto mydone; /* done */
                }
              }
            }
          }
        }
        mydone:
        median += span + span;
      }
      span += span;
    }

    //free(aux);
  }
}




typedef enum
{
	RT_MAXIMUM = 0,
	RT_MINIMUM = 1
} ReductionType;


/**
 * CUDA kernel to search for maximum (or minimum) in an input array. Stores the result in the element of src
 * with index blockIdx.x
 *
 * \param[out] src Input array of data. Afterwards the maximums (minimums) of each block are stored in the first blockDim.x elements
 * \param[out] ret_indices Array of data indices. Afterwards the indices of the maximums (minimums) are stored in the first blockDim.x elements
 * \param[in] createIndices The first iteration of a complete maximum (minimum) search this flag must be set to create the indices in the extra array ret_indices
 */
template <typename T, ReductionType rt>
__global__ void CUDA_MAXIMUM_MINIMUM_SEARCH_KERNEL(T *src, int *ret_indices, bool createIndices)
{
	__shared__ T shsrc[2*NTHREADS];
	__shared__ int shidx[2*NTHREADS];

	unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (createIndices)
    {
    	shidx[tid] = idx;
    }
    else
    {
    	shidx[tid] = ret_indices[idx];
    }
    shsrc[tid] = src[idx];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    	if (tid < s) {
//    		if ((*device_func)<T>(shsrc[tid], shsrc[tid+s]))
    		if (   (shsrc[tid] < shsrc[tid+s] && rt == RT_MAXIMUM)
    		    || (shsrc[tid] > shsrc[tid+s] && rt == RT_MINIMUM) )
    		{
				shidx[tid] = shidx[tid + s];
				shsrc[tid] = shsrc[tid + s];
			}
    	}
    	__syncthreads();
    }


    // write result for this block to global mem
    if (tid == 0)
	{
    	ret_indices[blockIdx.x] = shidx[0];
    	src[blockIdx.x] = shsrc[0];
	}
}



/**
 * Calls CUDA kernels to search the maximum value. Returns the value and the index.
 *
 * \param blocks Count of values in the given arrays
 * \param maximum (Return) The maximum value in the array
 * \param maximum_idx (Return) The index of the maximum value in the array
 * \param gpu_maximum Array with values stored in device memory to search the maximum in
 * \param gpu_maximum_idx Array to store the indices of the array in (only used in this function). Must have the same size as gpu_maximum and of type int*
 */
template <typename T, ReductionType rt>
void CUDA_SEARCH_MAXIMUM_MINIMUM(int valueCount, T *result_value, int *result_value_idx, T *gpu_input, int *gpu_input_idx)
{
	int vc = valueCount;
	CUDA_MAXIMUM_MINIMUM_SEARCH_KERNEL<T, rt><<<vc, NTHREADS>>>(gpu_input, gpu_input_idx, true);

	do
	{
		if (vc >= NTHREADS)
		{
			vc /= NTHREADS;
			CUDA_MAXIMUM_MINIMUM_SEARCH_KERNEL<T, rt><<<vc, NTHREADS>>>(gpu_input, gpu_input_idx, false);
		}
	} while (vc > NTHREADS);
	if (vc > 1)
	{
		CUDA_MAXIMUM_MINIMUM_SEARCH_KERNEL<T, rt><<<1, vc>>>(gpu_input, gpu_input_idx, false);
	}
	cudaThreadSynchronize();
#ifdef CUDA_ERROR_CHECKS
	CheckCUDAError("CUDA_MAXIMUM_MINIMUM_SEARCH_KERNEL");
#endif

	// copy the return values from device memory to host memory
	cudaMemcpy(result_value, gpu_input, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(result_value_idx, gpu_input_idx, sizeof(int), cudaMemcpyDeviceToHost);
#ifdef CUDA_ERROR_CHECKS
	CheckCUDAError("cudaMemcpy");
#endif
}



template <int PARAMS_REQUIRED, int DATA_PT_SIZE>
__device__ void generateSamples(
		bool storeUsedSamples,
		int *usedSamples,
		bool useProvidedIndices,
		int sample_list_or_rand_seed_size,
		int *sample_list_or_rand_seed,
		int idx,
		int *rand_idx,
		const int datasetCount,
		const float *src,
		float *_dst)
{
	int i,j;

    // use provided sample indices or random data
    if (useProvidedIndices) // provided sample indices
    {
#pragma unroll
    	for (i=0; i<PARAMS_REQUIRED; ++i)
    	{
    		rand_idx[i] = sample_list_or_rand_seed[idx*PARAMS_REQUIRED + i];
    		if (storeUsedSamples)
    		{
    			usedSamples[i] = rand_idx[i];
    		}
    	}
    }
    else // select random data set indices
    {
		int seedIdx = 2*(idx % (sample_list_or_rand_seed_size/2));
		int seed1 = sample_list_or_rand_seed[seedIdx];
		int seed2 = sample_list_or_rand_seed[seedIdx+1];

#pragma unroll
    	for (i=0; i<PARAMS_REQUIRED; ++i)
    	{
    		// random number generator from George Marsaglia
    		seed1=36969*(seed1&65535)+(seed1>>16);
    		seed2=18000*(seed2&65535)+(seed2>>16);
    		rand_idx[i] = (int)(datasetCount*(((seed1<<16)+seed2)+2147483646.0f)/4294967294.0f);
    		if (storeUsedSamples)
    		{
    			usedSamples[i] = rand_idx[i];
    		}
    	}
    	sample_list_or_rand_seed[seedIdx]   = seed1;
    	sample_list_or_rand_seed[seedIdx+1] = seed2;
    }

    // compose the input data set for the transformation
#pragma unroll
    for(i=0; i < PARAMS_REQUIRED; i++) {
#pragma unroll
    	for (j=0; j < DATA_PT_SIZE; j++) {
    		_dst[DATA_PT_SIZE*i+j] = src[DATA_PT_SIZE*rand_idx[i]+j];
    	}
    }
}


template <int PARAMS_REQUIRED, int DATA_PT_SIZE>
__device__ bool containsDuplicates(
		float *src,
		int *rand_idx,
		DuplicateCheckType duplicateCheckState)
{
	if (duplicateCheckState == DuplicateCheckTypeNoCheck)
		return false;

	int i,j;

	if (duplicateCheckState == DuplicateCheckTypeIndexAndValueBased)
	{
		int k;
		bool all_the_same;

		// Check for duplicates
	    for (i=0; i<PARAMS_REQUIRED-1; ++i)
	    {
	    	for (j=i+1; j<PARAMS_REQUIRED; ++j)
	    	{
	    		// same index, then reject this iteration
	            if (rand_idx[i] == rand_idx[j])
	                return true;

				all_the_same = true;
				for (k=0; k<DATA_PT_SIZE; ++k)
				{
					if (src[DATA_PT_SIZE*i+k] != src[DATA_PT_SIZE*j+k])
					{
						all_the_same = false;
						break;
					}
				}
				if (all_the_same)
					return all_the_same;
	    	}
	    }
	}
	else if (duplicateCheckState == DuplicateCheckTypeIndexBased)
	{
	    // Check for duplicates only with index based check
	    for (i=0; i<PARAMS_REQUIRED-1; ++i)
	    {
	    	for (j=i+1; j<PARAMS_REQUIRED; ++j)
	    	{
	    		// same index, then reject this iteration
	            if (rand_idx[i] == rand_idx[j])
	                return true;
	    	}
	    }
	}

    return false;
}



