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

/*! \file */
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <assert.h>
#include <vector>

#include <sys/time.h>

/*! \mainpage FestGPU documentation main page
 *
 * \section introduction Introduction
 * <p>Robust estimation is used in a wide range of applications. One of the most popular
 * algorithms for robust estimation is the Random Sample Consensus (RANSAC, <a href="#ref1">[1]</a>)
 * achieving a high degree of accuracy even with a significant amount of outliers. A major drawback
 * is the fast increasing number of iterations caused by higher outlier ratios also involving
 * increasing computational costs.</p>
 * <p>Many problems are currently solved on the GPU. Depending on the problem the computation
 * can be up to some orders of magnitude faster than on the CPU, for instance,
 * segmentation <a href="#ref2">[2]</a> and 3D reconstruction <a href="#ref3">[3]</a>. Even
 * larger problems such as bundle adjustment can be solved on the GPU <a href="#ref4">[4]</a>.
 * </p>
 * <p><em>FestGPU</em>, a framework for fast robust estimation on the GPU is presented
 * that supports multiple subtypes of the RANSAC family <a href="#ref5">[5]</a>.
 * Compared to a CPU implementation, it reaches a speedup up to 135 times to a singlecore
 * CPU and up to 36 times to a quadcore CPU of the same hardware generation.
 * Together with a C++ and a Matlab interface the framework is publicly on this website.</p>
 * Two examples are included in the package to demonstrate the use of the framework:
 * <ol><li>Line estimation from 2D points</li>
 * <li>Fundamental matrix estimation from 2D point correspondences <a href="#ref6">[6]</a></li></ol>
 *
 * <p>If you have any troubles with the framework, feel free to <a href="mailto:jan_dot_roters_at_uni-muenster_dot_de">contact me</a>.</p>
 *
 * \section getting_started Getting started
 * <p>The framework is implemented in CUDA C. Thus, only NVIDIA graphics devices are supported.</p>
 *
 * \subsection dependencies Dependencies
 * <p>To build the framework there are only a few dependencies.</p>
 * <ol>
 * <li><em><a href="http://www.cmake.org/" onclick="window.open('http://www.cmake.org/'); return false;">CMake:</a></em><br>
 * The framework uses CMake to compile the source codes on different platforms. At least version 2.6 of CMake is required.
 * <li><em><a href="https://developer.nvidia.com/cuda-toolkit" onclick="window.open('https://developer.nvidia.com/cuda-toolkit'); return false;">
 * CUDA:</a></em><br>
 * The CUDA framework is required to compile the CUDA source code. It has been tested with CUDA 5.0 but
 * it should also work with higher versions.</li>
 * <li><em><a href="http://www.mathworks.de/products/matlab/" onclick="window.open('http://www.mathworks.de/products/matlab/'); return false;">
 * Matlab (optional):</a></em><br>
 * To compile the Matlab interface it is required that Matlab is installed on your computer. We've tested
 * the framework with a rather old version of Matlab, 7.11.0.584 (R2010b) but it should also work with
 * newer versions.</li>
 * </ol>
 *
 * \subsection compilation Compile the framework
 * <p>The framework uses <a href="http://www.cmake.org/" onclick="window.open('http://www.cmake.org/'); return false;">CMake</a>
 * to check for the availability of the dependencies and to compile the framework.</p>
 * <p>In the following it is described how to compile the framework.
 * Firstly, a makefile or project files for IDEs, e.g. Visual Studio, are created.
 * These files are created from the CMakeLists.txt included in the root directory of the framework.
 * Secondly, the project files can be opened or the makefile can be compiled by command line.
 * </p>
 * <p>Linux / Mac OS X: (command line)</p>
\verbatim
mkdir build
cd build
cmake ..
make
\endverbatim
 *
 * <p>Windows</p>
\verbatim
coming soon
\endverbatim
 *
 *
 * \subsection examples Execute the examples
 * <p>The examples are compiled to the bin directory inside the root directory of the framework.
 * Furthermore, the example data is copied to the exact same directory.</p>
 * <p>There are three example files included in the framework.</p>
 * <ol>
 * <li><em>example.cu:</em><br>
 * This file is compiled to 'FestGPU_example' in the bin directory and should be executed there.
 * It is important that the data is in the same directory.</li>
 * <li><em>example_fmatrix.cu:</em><br>
 * This file is compiled to 'FestGPU_example_FMatrix' in the bin directory. The first parameter should be the path
 * to a data file. Therefore, the data file has to be in a following format. Each four rows of the data file should
 * contain x1, y1, x2, y2 of one point correspondence.</li>
 * <li><em>example_line.cu:</em><br>
 * This file is compiled to 'FestGPU_example_Line' in the bin directory. The first parameter should be the path
 * to a data file. Therefore, the data file has to be in a following format. Each two rows of the data file should
 * contain x, y of one 2D point.</li>
 * </ol>
 *
 * \section estimator_implementation Implement your own estimator class
 * <p>To write a custom estimator class please copy the ExampleEstimator (ExampleEstimator.cu) in
 * the subdirectory src/estimators to a new file into the same directory.</p>
 * <p>Thereafter, you should rename the estimator class from ExampleEstimator some other name.
 * Furthermore, the two methods, computeHypothesis and computeLossValue have to be implemented.</p>
 * <p>Another simple example is given in LineEstimator showing the estimation of lines from 2D points.</p>
 *
 * \section acknowledgements Acknowledgements
 * This work was developed in the project AVIGLE funded by the State of North Rhine Westphalia (NRW), Germany,
 * and the European Union, European Regional Development Fund "Europe - Investing in your future". AVIGLE is
 * conducted in cooperation with several industrial and academic partners. We thank all project partners for
 * their work and contributions to the project. Furthermore, we thank Cenalo GmbH for their image acquisition.
 *
 * \section references References
 * <ol>
 * <li><a name="ref1"></a>Fischler, M.A., Bolles, R.C.:
 * <em>Random sample consensus: a paradigm for model fitting
 * with applications to image analysis and automated cartography.</em>
 * Communications of the ACM 24(6), 381-395 (1981)
 * </li>
 * <li><a name="ref2"></a>Montanes Laborda, M., Torres Moreno, E., Martinez del Rincon, J., Herrero Jaraba, J.:
 * <em>Real-time gpu color-based segmentation of football players.</em>
 * Journal of Real-Time Image Processing 7(4), 267-279 (2012)
 * </li>
 * <li><a name="ref3"></a>Roters, J., Jiang, X.:
 * <em>Incremental dense reconstruction from sparse 3D points with an integrated level-of-detail concept.</em>
 * In: Advances in Depth Image Analysis and Applications, vol. LNCS 7854, pp. 116-125.
 * Springer, Heidelberg (2013)
 * </li>
 * <li><a name="ref4"></a>Wu, C., Agarwal, S., Curless, B., Seitz, S.M.:
 * <em>Multicore bundle adjustment.</em>
 * In: Proceedings of IEEE Conference on Computer Vision and Pattern Recognition, pp. 3057-3064 (2011)
 * </li>
 * <li><a name="ref5"></a>Choi, S., Kim, T., Yu, W.:
 * <em>Performance evaluation of RANSAC family.</em>
 * In: Proceedings of the British Machine Vision Conference, pp. 1-12 (2009)
 * </li>
 * <li><a name="ref6"></a>Hartley, R.I., Zisserman, A.:
 * <em>Multiple View Geometry in Computer Vision</em>,
 * second edn. Cambridge University Press (2004)
 * </li>
 * </ol>
 *
 */


//#define CUDA_ERROR_CHECKS


#ifndef NTHREADS
#define NTHREADS 128
#endif


using namespace std;


#include "FestGPU_helpers.cu"
#include "FestGPU_user_include.cu"


/*!
 * \brief Kernel to estimate a model using LMS.
 */
template <template <int, int, int> class T, int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
__global__ void CUDA_LMS_ESTIMATION_KERNEL(
		T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> *estimator,
		const float *src,
		int datasetCount,
		bool storeUsedSamples,
		int *usedSamples,
		bool useProvidedIndices, // if true, providedIndices are used, otherwise startSeed is used
		DuplicateCheckType duplicateCheckState,
		int sample_list_or_rand_seed_size,
		int *sample_list_or_rand_seed,
		int iterationCount,
		float *distances,
		float *ret_measurement,
		float *ret_model)
{
	int i;
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx >= iterationCount) {
        return;
    }

    float _src[PARAMS_REQUIRED*DATA_PT_SIZE];
    float _dst[HYPOTHESIS_SIZE];
    int rand_idx[PARAMS_REQUIRED];

    // initialize measurement return value to the maximum possible value
    ret_measurement[idx] = 1e38;

    generateSamples<PARAMS_REQUIRED, DATA_PT_SIZE>(
    		storeUsedSamples,
    		usedSamples,
    		useProvidedIndices,
    		sample_list_or_rand_seed_size,
    		sample_list_or_rand_seed,
    		idx,
    		rand_idx,
    		datasetCount,
    		src,
    		_src);


    if (containsDuplicates<PARAMS_REQUIRED, DATA_PT_SIZE>(
    		_src,
    		rand_idx,
    		duplicateCheckState))
    	return;


	// do degeneration check
	if (estimator->isDegenerated(
			_src))
		return;

	// compute transformation for this iteration
	estimator->computeHypothesis(
			_src,
			_dst);


    // evaluate transformation with each (remaining) data set
    float *dists = &(distances[idx*datasetCount]);
    for (i=0; i < datasetCount; ++i)
    {
    	dists[i] = estimator->computeLossValue(_dst, &(src[DATA_PT_SIZE*i]));
    }

    // copy the median of error values into global memory (return value)
    ret_measurement[idx] = CUDA_MEDIAN_DEVICE(dists, datasetCount);


    // copy the model into global memory (return value)
#pragma unroll
    for (i=0; i < HYPOTHESIS_SIZE; ++i)
    {
    	ret_model[idx*HYPOTHESIS_SIZE + i] = _dst[i];
    }
}


/*!
 * \brief Kernel to estimate a model using RANSAC or other subtypes of the RANSAC family.
 */
template <template <int, int, int> class T, int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
__global__ void CUDA_ESTIMATION_KERNEL(
		T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> *estimator,
		const float *src,
		int datasetCount,
		bool storeUsedSamples,
		int *usedSamples,
		bool useProvidedIndices, // if true, providedIndices are used, otherwise startSeed is used
		DuplicateCheckType duplicateCheckState,
		int sample_list_or_rand_seed_size,
		int *sample_list_or_rand_seed,
		int iterationCount,
		float *ret_measurement,
		float *ret_model)
{
	int i;
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx >= iterationCount) {
        return;
    }

    float _src[PARAMS_REQUIRED*DATA_PT_SIZE];
    float _dst[HYPOTHESIS_SIZE];
    int rand_idx[PARAMS_REQUIRED];

    // initialize measurement return value to dataset count
    // since that is the worst value that could appear
    // this initialization is only for aborting the computation
    ret_measurement[idx] = datasetCount;

    generateSamples<PARAMS_REQUIRED, DATA_PT_SIZE>(
    		storeUsedSamples,
    		usedSamples,
    		useProvidedIndices,
    		sample_list_or_rand_seed_size,
    		sample_list_or_rand_seed,
    		idx,
    		rand_idx,
    		datasetCount,
    		src,
    		_src);


    if (containsDuplicates<PARAMS_REQUIRED, DATA_PT_SIZE>(
    		_src,
    		rand_idx,
    		duplicateCheckState))
    	return;


	// do degeneration check
	if (estimator->isDegenerated(
			_src))
		return;

	// compute transformation for this iteration
	estimator->computeHypothesis(
			_src,
			_dst);


	// evaluate transformation with each (remaining) data set
	float measurement_value = 0.0f;
	for (i=0; i < datasetCount; ++i)
	{
		// here is the point where to change the rating of the error value
		measurement_value += estimator->computeLossValue(
				_dst,
				&(src[DATA_PT_SIZE*i]));
	}

	// copy inlier_count of this iteration into global memory (return value)
	ret_measurement[idx] = measurement_value;


    // copy the model into global memory (return value)
#pragma unroll
    for (i=0; i < HYPOTHESIS_SIZE; ++i)
    {
    	ret_model[idx*HYPOTHESIS_SIZE + i] = _dst[i];
    }
}



/**
 * \brief Starts the estimation with a specified EstimationType
 *
 * Starts estimation with a specified EstimationType, e.g. RANSAC or LMS.
 * There are several parameters changing the estimation or the input / output
 * values.
 *
 * \param[in] estimationType Defines the type of the estimation, e.g. EstimationTypeRANSAC
 * \param[in] estimator The estimator delivering the required problem-specific methods. For further information take a look at the LineEstimator
 * \param[in] src Source data in custom order (take care in the CUDA methods computeHypothesis, computeLossValue, ...)
 * \param[in] storeUsedSamples If true, the samples that have been used to compute the hypothesis are stored in usedSamples
 * \param[out] usedSamples Stores the samples that have been used to compute the final hypothesis. This is only used when storeUsedSamples is true
 * \param[in] useProvidedIndices If true, the user can provide the indices of the datapoints to be preferred in the estimation
 * \param[in] providedIndices The indices of the datapoints in a specified order. For instance, the order is given by the order of the quality of the datapoints. This is only used when useProvidedIndices is true
 * \param[in] duplicateCheckState Determines if a duplicate check is applied. Furthermore, either a index based duplicate check can be done or a index and value based check.
 * \param[out] bestModel The hypothesis with the bestMeasurement value is returned
 * \param[out] bestMeasurement The best measurement that has been found in the estimation, e.g. the minimum number of outliers when estimating with RANSAC
 * \param[out] iterationCount Count of iterationCount the framework has done to finish the estimation
 * \param[in] useAdaptiveRansac If true, the adaptive RANSAC is used instead of the non-adaptive RANSAC. The expectedInlierRatio is not required
 * \param[in] expectedInlierRatio The expected inlier ratio of the dataset src. This is only used when useAdaptiveRansac is true (0 < expectedInlierRatio <= 1)
 * \param[in] confidence The confidence that should be achieved by the estimation to get a good result (0 < confidence < 1)
 * \return A ResultValueType, e.g. an error state
 */
template <template <int, int, int> class T, int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
ResultValueType CUDA_Estimation(
		EstimationType estimationType,
		T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> &estimator,
		const vector<float> &src,
		bool storeUsedSamples,
		vector<int> &usedSamples,
		bool useProvidedIndices, // if true, providedIndices are used, otherwise seeds have to be generated for each thread
		const vector<int> &providedIndices,
		DuplicateCheckType duplicateCheckState,
		vector<float> &bestModel,
		float *bestMeasurement,
		int *iterationCount,
		bool useAdaptiveRansac,
		float expectedInlierRatio, // this is for the non-adaptive case
		float confidence)
{
	int devCount = 0;
	cudaGetDeviceCount(&devCount);

	if (devCount < 1)
    {
        return ResultValueTypeNoCUDADeviceFound;
    }

    cudaDeviceProp deviceProperties;
	for (int i=0; i<devCount; ++i)
	{
        cudaGetDeviceProperties(&deviceProperties, i);
	}

	if ((estimationType == EstimationTypeLMS && useAdaptiveRansac)
			|| (!useAdaptiveRansac && expectedInlierRatio <= 0.0f)
			|| (!useAdaptiveRansac && expectedInlierRatio > 1.0f))
	{
		return ResultValueTypeWrongParameters;
	}

	long localIterationCount;
	if (useAdaptiveRansac)
	{
		localIterationCount = LONG_MAX;
	}
	else
	{
		localIterationCount = (long)ceil(
				log(1.0 - confidence) / log(1.0 - pow(expectedInlierRatio, PARAMS_REQUIRED))
				+ sqrt(1.0 - pow(expectedInlierRatio, 2))/pow(expectedInlierRatio, PARAMS_REQUIRED)); // standard deviation
		if (localIterationCount < 0)
			localIterationCount = LONG_MAX;
	}


    int threads = NTHREADS; //deviceProperties.maxThreadsPerBlock;
    int blocks;
    int maxBlocks;
    if (useAdaptiveRansac)
    {
    	maxBlocks = 2048;
    	blocks    = deviceProperties.multiProcessorCount;
    }
    else
    {
    	maxBlocks = localIterationCount / threads + ((localIterationCount % threads)?1:0);
    	blocks    = maxBlocks;
    }
    int threadsTimesBlocks = threads * blocks;
    int threadsTimesMaxBlocks = threads * maxBlocks;
    int datasetCount = src.size() / DATA_PT_SIZE;

    // allocate memory for the estimator
	T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> *gpu_estimator;
	cudaMalloc(&gpu_estimator, sizeof(T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE>));

    float *gpu_src;
    cudaMalloc(&gpu_src, sizeof(float)*src.size());

    // alloc memory for the provided samples or the random seeds
    int sample_indices_list_size;
	int *gpu_sample_indices_or_rand_list;
    if (useProvidedIndices) // provided samples
    	sample_indices_list_size = threadsTimesMaxBlocks * PARAMS_REQUIRED;
    else // random seeds
    	sample_indices_list_size = 2*deviceProperties.warpSize * deviceProperties.multiProcessorCount;
    int *sample_indices_list = new int[sample_indices_list_size];
	cudaMalloc(&gpu_sample_indices_or_rand_list, sample_indices_list_size*sizeof(int));


    // alloc memory for the samples that have been used to compute the estimated model
	int *used_samples_arr;
	int *gpu_used_samples;
    if (storeUsedSamples)
    {
		used_samples_arr = new int[PARAMS_REQUIRED];
		usedSamples.resize(PARAMS_REQUIRED);
    	cudaMalloc(&gpu_used_samples, sizeof(int)*PARAMS_REQUIRED);
    }


    // alloc memory for measurements (loss values) for each
    // iteration (which could be executed at the same time)
    float *gpu_measurements;
    cudaMalloc(&gpu_measurements, sizeof(float)*threadsTimesMaxBlocks);


    // alloc memory for LMS distance array for each dataset and for each possible thread
    // there has to be one float for distance evaluation and median computation
    float *gpu_distances;
    if (estimationType == EstimationTypeLMS)
    {
        cudaMalloc(&gpu_distances, sizeof(float)*datasetCount*threadsTimesMaxBlocks);
    }

    // alloc memory for the models computed in each iteration
    float *gpu_models;
    cudaMalloc(&gpu_models, sizeof(float)*HYPOTHESIS_SIZE*threadsTimesMaxBlocks);

    // alloc memory for the minimum / maximum computation
	float minimum;
	int minimum_idx;
	int *gpu_minimum_idx; // gpu_maximum not required, since gpu_measurement can be used!
	cudaMalloc(&gpu_minimum_idx, sizeof(int)*blocks);

#ifdef CUDA_ERROR_CHECKS
    // check for previous allocation errors
    CheckCUDAError("cudaMalloc");
#endif

    // generate random numbers (seeds for random number generation on the device)
	timeval seed_init;
	gettimeofday(&seed_init, NULL);
    srand ( seed_init.tv_usec );
    //srand ( 0 ); // only for debugging
    if (!useProvidedIndices)
    {
    	for(int i=0; i < sample_indices_list_size; i++) {
    		sample_indices_list[i] = (int)rand();
    	}

    	// copy random seeds to device
    	cudaMemcpy(gpu_sample_indices_or_rand_list, sample_indices_list,
    			sizeof(int)*sample_indices_list_size, cudaMemcpyHostToDevice);
    }

    // copy estimator to gpu memory
	cudaMemcpy(gpu_estimator, &estimator, sizeof(T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE>), cudaMemcpyHostToDevice);

    // copy src input data to GPU
	cudaMemcpy(gpu_src, &src[0], sizeof(float)*src.size(), cudaMemcpyHostToDevice);
#ifdef CUDA_ERROR_CHECKS
	CheckCUDAError("cudaMemcpy");
#endif


	long iterationsDoneSoFar = 0;

	float best_measurement_so_far = datasetCount;
	float *best_model_arr_so_far = new float[HYPOTHESIS_SIZE];
	bool best_measurement_gets_better = false;
	int best_measurement_idx = -1;


	while (iterationsDoneSoFar < localIterationCount)
	{
		//cout << "iterationCount: " << iterationsDoneSoFar << " / " << localIterationCount << endl;

		// copy provided sample indices to device
		if (useProvidedIndices)
		{
			int startSample = iterationsDoneSoFar * PARAMS_REQUIRED;
			for (int i=0; i < threadsTimesBlocks * PARAMS_REQUIRED; i++) {
				if (i+startSample < providedIndices.size())
				{
					sample_indices_list[i] = providedIndices[i + startSample];
				}
				else
				{
					// fill required data with random indices
					// this only happens if there are not enough samples provided
					sample_indices_list[i] = (int)(datasetCount * (rand()/(1.0 + RAND_MAX)));
				}
			}

			cudaMemcpy(gpu_sample_indices_or_rand_list, sample_indices_list,
					sizeof(int) * threadsTimesBlocks * PARAMS_REQUIRED, cudaMemcpyHostToDevice);
			#ifdef CUDA_ERROR_CHECKS
				CheckCUDAError("cudaMemcpy");
			#endif
		}

		if (estimationType == EstimationTypeRANSAC)
		{
			CUDA_ESTIMATION_KERNEL<<<blocks, threads>>>(
					gpu_estimator,
					gpu_src,
					datasetCount,
					storeUsedSamples,
					gpu_used_samples,
					useProvidedIndices,
					duplicateCheckState,
					sample_indices_list_size,
					gpu_sample_indices_or_rand_list,
					threadsTimesBlocks,
					gpu_measurements,
					gpu_models);
			cudaThreadSynchronize();
	#ifdef CUDA_ERROR_CHECKS
			CheckCUDAError("CUDA_ESTIMATION_KERNEL");
	#endif
		}
		else
		{
			CUDA_LMS_ESTIMATION_KERNEL<<<blocks, threads>>>(
					gpu_estimator,
					gpu_src,
					datasetCount,
					storeUsedSamples,
					gpu_used_samples,
					useProvidedIndices,
					duplicateCheckState,
					sample_indices_list_size,
					gpu_sample_indices_or_rand_list,
					threadsTimesBlocks,
					gpu_distances,
					gpu_measurements,
					gpu_models);
			cudaThreadSynchronize();
	#ifdef CUDA_ERROR_CHECKS
			CheckCUDAError("CUDA_LMS_ESTIMATION_KERNEL");
	#endif
		}

		// find the best model, i.e. with minimum rating
		CUDA_SEARCH_MAXIMUM_MINIMUM<float, RT_MINIMUM>(blocks, &minimum, &minimum_idx, gpu_measurements, gpu_minimum_idx);

		if (minimum < best_measurement_so_far)
		{
			best_measurement_so_far = minimum;
			best_measurement_idx = minimum_idx;
			best_measurement_gets_better = true;

			if (storeUsedSamples)
			{
				cudaMemcpy(used_samples_arr, gpu_used_samples, sizeof(int)*PARAMS_REQUIRED, cudaMemcpyDeviceToHost);

#ifdef CUDA_ERROR_CHECKS
				CheckCUDAError("cudaMemcpy");
#endif
				usedSamples.clear();
				usedSamples.resize(PARAMS_REQUIRED);
				for (int i=0; i<PARAMS_REQUIRED; ++i)
				{
					usedSamples[i] = used_samples_arr[i];
				}
			}
		}

		iterationsDoneSoFar += threadsTimesBlocks;

		// copy the best model
		if (best_measurement_gets_better)
		{
			best_measurement_gets_better = false;
			bestMeasurement[0] = best_measurement_so_far;

			cudaMemcpy(best_model_arr_so_far, &gpu_models[best_measurement_idx*HYPOTHESIS_SIZE], sizeof(float)*HYPOTHESIS_SIZE, cudaMemcpyDeviceToHost);
#ifdef CUDA_ERROR_CHECKS
			CheckCUDAError("cudaMemcpy");
#endif

			if (useAdaptiveRansac)
			{
				float curInlierRatio = 1 - bestMeasurement[0]/datasetCount;
				localIterationCount = (long)ceil(
						log(1.0 - confidence) / log(1.0 - pow(curInlierRatio, PARAMS_REQUIRED))
						+ sqrt(1.0 - pow(curInlierRatio, 2))/pow(curInlierRatio, PARAMS_REQUIRED)); // standard deviation
				if (localIterationCount < 0)
					localIterationCount = LONG_MAX;
			}
		}

		if (useAdaptiveRansac)
		{
			blocks *= 2;
			if (blocks > maxBlocks)
				blocks = maxBlocks;
			threadsTimesBlocks = threads * blocks;

			// check if blocks can be reduced due to remaining iteration count
			long remainingCount = localIterationCount - iterationsDoneSoFar;
			if (remainingCount > 0 && remainingCount < threadsTimesBlocks)
			{
		    	blocks = remainingCount / threads + ((remainingCount % threads)?1:0);
				threadsTimesBlocks = threads * blocks;
			}
		}
	}

	*iterationCount = iterationsDoneSoFar;

	// copy best model to output vector
	bestModel.clear();
	bestModel.resize(HYPOTHESIS_SIZE);
	for (int i=0; i<HYPOTHESIS_SIZE; ++i)
	{
		bestModel[i] = best_model_arr_so_far[i];
	}
	delete[] best_model_arr_so_far;


	cudaFree(gpu_estimator);

	cudaFree(gpu_minimum_idx);

    cudaFree(gpu_src);

    cudaFree(gpu_sample_indices_or_rand_list);
    delete[] sample_indices_list;

    cudaFree(gpu_measurements);

    if (estimationType == EstimationTypeLMS)
    {
    	cudaFree(gpu_distances);
    }

    cudaFree(gpu_models);

    if (storeUsedSamples)
    {
    	cudaFree(gpu_used_samples);
    	delete[] used_samples_arr;
    }

    return ResultValueTypeSuccessful;
}



// -------------------------------------------------------
// --- FROM HERE ON ONLY THE INLIER COMPUTATION STARTS ---
// -------------------------------------------------------

/**
 * \brief Kernel for extracting inliers of a given model, threshold and a dataset
 */
template <template <int, int, int> class T, int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
__global__ void CUDA_INLIER_COMPUTATION_KERNEL(
		T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> *estimator,
		const float *src,
		int datasetCount,
		float *model,
		float threshold,
		int *inliers)
{
	int j;
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // if index is larger than the number of iterationCount required leave
    if(idx >= datasetCount) {
        return;
    }

    // evaluate transformation with each data set
    float test_set[DATA_PT_SIZE];
#pragma unroll
	for (j=0; j < DATA_PT_SIZE; ++j)
	{
		test_set[j] = src[DATA_PT_SIZE*idx+j];
	}
	inliers[idx] = (estimator->computeLossValue(model, test_set) <= threshold ? 1 : 0);
}



/**
 * \brief This method is for separate inlier computation. With a given model, threshold and the source data the inlier indices are computed.
 *
 * \param[in] src The datasource
 * \param[in] inlier_threshold Threshold to decide if a datum is an inlier or outlier
 * \param[in] model Model to compute the goodness values from the datasource
 * \param[in] threshold Threshold that separates between inliers and outliers related to the loss value
 * \param[out] inlier Inlier array with 1 = inlier and 0 = outlier
 * \return A ResultValueType, e.g. an error state
 */
template <template <int, int, int> class T, int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
ResultValueType CUDA_Get_Inliers(
		T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> &estimator,
		const vector<float> &src,
		vector<float> &model,
		float threshold,
		vector<int> &inlier)
{
	int devCount = 0;
	cudaGetDeviceCount(&devCount);

    if (devCount < 1)
    {
        return ResultValueTypeNoCUDADeviceFound;
    }

    cudaDeviceProp deviceProperties;
	for (int i=0; i<devCount; ++i)
	{
        cudaGetDeviceProperties(&deviceProperties, i);
	}

    int threads = NTHREADS; //deviceProperties.maxThreadsPerBlock;
    int datasetCount = src.size() / DATA_PT_SIZE;
    int blocks = datasetCount/threads + ((datasetCount % threads)?1:0);

    // allocate memory for the estimator
	T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> *gpu_estimator;
	cudaMalloc(&gpu_estimator, sizeof(T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE>));

    float *gpu_src;
    cudaMalloc(&gpu_src, sizeof(float)*src.size());

    // alloc memory for the model to compute the inliers for
    float *gpu_model;
    cudaMalloc(&gpu_model, sizeof(float)*HYPOTHESIS_SIZE);

    // alloc memory for return inlier values
    int *gpu_inliers;
    cudaMalloc(&gpu_inliers, sizeof(int)*datasetCount);

#ifdef CUDA_ERROR_CHECKS
    // check for previous allocation errors
    CheckCUDAError("cudaMalloc");
#endif

    // copy estimator data to gpu memory
	cudaMemcpy(gpu_estimator, &estimator, sizeof(T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE>), cudaMemcpyHostToDevice);

    // copy datasource to gpu memory
    cudaMemcpy(gpu_src, &src[0], sizeof(float)*src.size(), cudaMemcpyHostToDevice);

    // copy model to gpu memory
    cudaMemcpy(gpu_model, &model[0], sizeof(float)*HYPOTHESIS_SIZE, cudaMemcpyHostToDevice);

#ifdef CUDA_ERROR_CHECKS
    // check for previous memcpy errors
	CheckCUDAError("cudaMemcpy");
#endif

	CUDA_INLIER_COMPUTATION_KERNEL<<<blocks, threads>>>(
			gpu_estimator,
			gpu_src,
			datasetCount,
			gpu_model,
			threshold,
			gpu_inliers);
	cudaThreadSynchronize();

#ifdef CUDA_ERROR_CHECKS
	// check for previous kernel errors
	CheckCUDAError("CUDA_INLIER_COMPUTATION_KERNEL");
#endif


	// get the best model and inlier count
	if (inlier.size() < datasetCount)
		inlier.resize(datasetCount);
	cudaMemcpy(&inlier[0], gpu_inliers, sizeof(int)*datasetCount, cudaMemcpyDeviceToHost);

#ifdef CUDA_ERROR_CHECKS
	// check for previous memcpy errors
	CheckCUDAError("cudaMemcpy");
#endif

	cudaFree(gpu_estimator);
	cudaFree(gpu_src);
    cudaFree(gpu_model);
    cudaFree(gpu_inliers);

    return ResultValueTypeSuccessful;
}



// -------------------------------------------------------
// --------------------- Wrapper -------------------------
// -------------------------------------------------------

// ==== RANSAC WRAPPER ====

/**
 * \brief Starts the estimation with non-adaptive RANSAC. The exact estimation type can be changed by the implementations of the problem-specific functions.
 *
 * \param[in] estimator The estimator delivering the required problem-specific methods. For further information take a look at the LineEstimator
 * \param[in] src Source data in custom order (take care in the CUDA methods computeHypothesis, computeLossValue, ...)
 * \param[out] bestModel The hypothesis with the bestMeasurement value is returned
 * \param[in] expectedInlierRatio The expected inlier ratio of the dataset src. This is only used when useAdaptiveRansac is true (0 < expectedInlierRatio <= 1)
 * \return A ResultValueType, e.g. an error state
 */
template <template <int, int, int> class T, int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
ResultValueType FestGPU(
		T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> &estimator,
		const vector<float> &src,
		vector<float> &bestModel,
		float expectedInlierRatio
		)
{
	vector<int> usedSamples;
	vector<int> providedIndices;
	float bestMeasurement;
	int iterationCount;

	return CUDA_Estimation(
			EstimationTypeRANSAC,
			estimator,
			src,
			false,
			usedSamples,
			false,
			providedIndices,
			DuplicateCheckTypeIndexAndValueBased,
			bestModel,
			&bestMeasurement,
			&iterationCount,
			false,
			expectedInlierRatio,
			0.99f
			);
}

/**
 * \brief Starts the estimation with non-adaptive RANSAC. The exact estimation type can be changed by the implementations of the problem-specific functions.
 *
 * \param[in] estimator The estimator delivering the required problem-specific methods. For further information take a look at the LineEstimator
 * \param[in] src Source data in custom order (take care in the CUDA methods computeHypothesis, computeLossValue, ...)
 * \param[out] bestModel The hypothesis with the bestMeasurement value is returned
 * \param[in] expectedInlierRatio The expected inlier ratio of the dataset src. This is only used when useAdaptiveRansac is true (0 < expectedInlierRatio <= 1)
 * \param[in] duplicateCheckState Determines if a duplicate check is applied. Furthermore, either a index based duplicate check can be done or a index and value based check.
 * \param[out] bestMeasurement The best measurement that has been found in the estimation, e.g. the minimum number of outliers when estimating with RANSAC
 * \param[out] iterationCount Count of iterationCount the framework has done to finish the estimation
 * \param[in] confidence (Optional) The confidence that should be achieved by the estimation to get a good result (0 < confidence < 1)
 * \return A ResultValueType, e.g. an error state
 */
template <template <int, int, int> class T, int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
ResultValueType FestGPU(
		T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> &estimator,
		const vector<float> &src,
		vector<float> &bestModel,
		float expectedInlierRatio, // this is for the non-adaptive case
		DuplicateCheckType duplicateCheckState,
		float &bestMeasurement,
		int &iterationCount,
		float confidence = 0.99f
		)
{
	vector<int> usedSamples;
	vector<int> providedIndices;

	return CUDA_Estimation(
			EstimationTypeRANSAC,
			estimator,
			src,
			false,
			usedSamples,
			false,
			providedIndices,
			duplicateCheckState,
			bestModel,
			&bestMeasurement,
			&iterationCount,
			false,
			expectedInlierRatio,
			confidence
			);
}

/**
 * \brief Starts the estimation with non-adaptive RANSAC. The exact estimation type can be changed by the implementations of the problem-specific functions.
 *
 * \param[in] estimator The estimator delivering the required problem-specific methods. For further information take a look at the LineEstimator
 * \param[in] src Source data in custom order (take care in the CUDA methods computeHypothesis, computeLossValue, ...)
 * \param[out] bestModel The hypothesis with the bestMeasurement value is returned
 * \param[in] expectedInlierRatio The expected inlier ratio of the dataset src. This is only used when useAdaptiveRansac is true (0 < expectedInlierRatio <= 1)
 * \param[in] duplicateCheckState Determines if a duplicate check is applied. Furthermore, either a index based duplicate check can be done or a index and value based check.
 * \param[out] bestMeasurement The best measurement that has been found in the estimation, e.g. the minimum number of outliers when estimating with RANSAC
 * \param[out] iterationCount Count of iterationCount the framework has done to finish the estimation
 * \param[in] providedIndices The indices of the datapoints in a specified order. For instance, the order is given by the order of the quality of the datapoints. This is only used when useProvidedIndices is true
 * \param[in] confidence (Optional) The confidence that should be achieved by the estimation to get a good result (0 < confidence < 1)
 * \return A ResultValueType, e.g. an error state
 */
template <template <int, int, int> class T, int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
ResultValueType FestGPU(
		T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> &estimator,
		const vector<float> &src,
		vector<float> &bestModel,
		float expectedInlierRatio, // this is for the non-adaptive case
		DuplicateCheckType duplicateCheckState,
		float &bestMeasurement,
		int &iterationCount,
		vector<int> &providedIndices,
		float confidence = 0.99f
		)
{
	vector<int> usedSamples;

	return CUDA_Estimation(
			EstimationTypeRANSAC,
			estimator,
			src,
			false,
			usedSamples,
			true,
			providedIndices,
			bestModel,
			&bestMeasurement,
			&iterationCount,
			false,
			expectedInlierRatio,
			confidence
			);
}

/**
 * \brief Starts the estimation with non-adaptive RANSAC. The exact estimation type can be changed by the implementations of the problem-specific functions.
 *
 * \param[in] estimator The estimator delivering the required problem-specific methods. For further information take a look at the LineEstimator
 * \param[in] src Source data in custom order (take care in the CUDA methods computeHypothesis, computeLossValue, ...)
 * \param[out] bestModel The hypothesis with the bestMeasurement value is returned
 * \param[in] expectedInlierRatio The expected inlier ratio of the dataset src. This is only used when useAdaptiveRansac is true (0 < expectedInlierRatio <= 1)
 * \param[in] duplicateCheckState Determines if a duplicate check is applied. Furthermore, either a index based duplicate check can be done or a index and value based check.
 * \param[out] bestMeasurement The best measurement that has been found in the estimation, e.g. the minimum number of outliers when estimating with RANSAC
 * \param[out] usedSamples Stores the samples that have been used to compute the final hypothesis. This is only used when storeUsedSamples is true
 * \param[out] iterationCount Count of iterationCount the framework has done to finish the estimation
 * \param[in] confidence (Optional) The confidence that should be achieved by the estimation to get a good result (0 < confidence < 1)
 * \return A ResultValueType, e.g. an error state
 */
template <template <int, int, int> class T, int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
ResultValueType FestGPU(
		T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> &estimator,
		const vector<float> &src,
		vector<float> &bestModel,
		float expectedInlierRatio, // this is for the non-adaptive case
		DuplicateCheckType duplicateCheckState,
		float &bestMeasurement,
		vector<int> &usedSamples,
		int &iterationCount,
		float confidence = 0.99f
		)
{
	vector<int> providedIndices;

	return CUDA_Estimation(
			EstimationTypeRANSAC,
			estimator,
			src,
			true,
			usedSamples,
			true,
			providedIndices,
			bestModel,
			&bestMeasurement,
			&iterationCount,
			false,
			expectedInlierRatio,
			confidence
			);
}

/**
 * \brief Starts the estimation with non-adaptive RANSAC. The exact estimation type can be changed by the implementations of the problem-specific functions.
 *
 * \param[in] estimator The estimator delivering the required problem-specific methods. For further information take a look at the LineEstimator
 * \param[in] src Source data in custom order (take care in the CUDA methods computeHypothesis, computeLossValue, ...)
 * \param[out] bestModel The hypothesis with the bestMeasurement value is returned
 * \param[in] expectedInlierRatio The expected inlier ratio of the dataset src. This is only used when useAdaptiveRansac is true (0 < expectedInlierRatio <= 1)
 * \param[in] duplicateCheckState Determines if a duplicate check is applied. Furthermore, either a index based duplicate check can be done or a index and value based check.
 * \param[out] bestMeasurement The best measurement that has been found in the estimation, e.g. the minimum number of outliers when estimating with RANSAC
 * \param[out] usedSamples Stores the samples that have been used to compute the final hypothesis. This is only used when storeUsedSamples is true
 * \param[out] iterationCount Count of iterationCount the framework has done to finish the estimation
 * \param[in] providedIndices The indices of the datapoints in a specified order. For instance, the order is given by the order of the quality of the datapoints. This is only used when useProvidedIndices is true
 * \param[in] confidence (Optional) The confidence that should be achieved by the estimation to get a good result (0 < confidence < 1)
 * \return A ResultValueType, e.g. an error state
 */
template <template <int, int, int> class T, int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
ResultValueType FestGPU(
		T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> &estimator,
		const vector<float> &src,
		vector<float> &bestModel,
		float expectedInlierRatio, // this is for the non-adaptive case
		DuplicateCheckType duplicateCheckState,
		float &bestMeasurement,
		vector<int> &usedSamples,
		int &iterationCount,
		vector<int> &providedIndices,
		float confidence = 0.99f
		)
{
	return CUDA_Estimation(
			EstimationTypeRANSAC,
			estimator,
			src,
			true,
			usedSamples,
			true,
			providedIndices,
			bestModel,
			&bestMeasurement,
			&iterationCount,
			false,
			expectedInlierRatio,
			confidence
			);
}



// ==== Adaptive RANSAC WRAPPER ====

/**
 * \brief Starts the estimation with adaptive RANSAC. The exact estimation type can be changed by the implementations of the problem-specific functions.
 *
 * \param[in] estimator The estimator delivering the required problem-specific methods. For further information take a look at the LineEstimator
 * \param[in] src Source data in custom order (take care in the CUDA methods computeHypothesis, computeLossValue, ...)
 * \param[out] bestModel The hypothesis with the bestMeasurement value is returned
 * \return A ResultValueType, e.g. an error state
 */
template <template <int, int, int> class T, int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
ResultValueType FestGPU_Adaptive(
		T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> &estimator,
		vector<float> const &src,
		vector<float> &bestModel
		)
{
	vector<int> usedSamples;
	vector<int> providedIndices;
	float bestMeasurement;
	int iterationCount;

	return CUDA_Estimation(
			EstimationTypeRANSAC,
			estimator,
			src,
			false,
			usedSamples,
			false,
			providedIndices,
			DuplicateCheckTypeIndexBased,
			bestModel,
			&bestMeasurement,
			&iterationCount,
			true,
			0,
			0.99f
			);
}

/**
 * \brief Starts the estimation with adaptive RANSAC. The exact estimation type can be changed by the implementations of the problem-specific functions.
 *
 * \param[in] estimator The estimator delivering the required problem-specific methods. For further information take a look at the LineEstimator
 * \param[in] src Source data in custom order (take care in the CUDA methods computeHypothesis, computeLossValue, ...)
 * \param[out] bestModel The hypothesis with the bestMeasurement value is returned
 * \param[in] duplicateCheckState Determines if a duplicate check is applied. Furthermore, either a index based duplicate check can be done or a index and value based check.
 * \param[out] bestMeasurement The best measurement that has been found in the estimation, e.g. the minimum number of outliers when estimating with RANSAC
 * \param[out] iterationCount Count of iterationCount the framework has done to finish the estimation
 * \param[in] confidence (Optional) The confidence that should be achieved by the estimation to get a good result (0 < confidence < 1)
 * \return A ResultValueType, e.g. an error state
 */
template <template <int, int, int> class T, int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
ResultValueType FestGPU_Adaptive(
		T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> &estimator,
		vector<float> const &src,
		vector<float> &bestModel,
		DuplicateCheckType duplicateCheckState,
		float &bestMeasurement,
		int &iterationCount,
		float confidence = 0.99f
		)
{
	vector<int> usedSamples;
	vector<int> providedIndices;

	return CUDA_Estimation(
			EstimationTypeRANSAC,
			estimator,
			src,
			false,
			usedSamples,
			false,
			providedIndices,
			duplicateCheckState,
			bestModel,
			&bestMeasurement,
			&iterationCount,
			true,
			0,
			confidence
			);
}

/**
 * \brief Starts the estimation with adaptive RANSAC. The exact estimation type can be changed by the implementations of the problem-specific functions.
 *
 * \param[in] estimator The estimator delivering the required problem-specific methods. For further information take a look at the LineEstimator
 * \param[in] src Source data in custom order (take care in the CUDA methods computeHypothesis, computeLossValue, ...)
 * \param[out] bestModel The hypothesis with the bestMeasurement value is returned
 * \param[in] duplicateCheckState Determines if a duplicate check is applied. Furthermore, either a index based duplicate check can be done or a index and value based check.
 * \param[out] bestMeasurement The best measurement that has been found in the estimation, e.g. the minimum number of outliers when estimating with RANSAC
 * \param[out] iterationCount Count of iterationCount the framework has done to finish the estimation
 * \param[in] providedIndices The indices of the datapoints in a specified order. For instance, the order is given by the order of the quality of the datapoints. This is only used when useProvidedIndices is true
 * \param[in] confidence (Optional) The confidence that should be achieved by the estimation to get a good result (0 < confidence < 1)
 * \return A ResultValueType, e.g. an error state
 */
template <template <int, int, int> class T, int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
ResultValueType FestGPU_Adaptive(
		T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> &estimator,
		vector<float> const &src,
		vector<float> &bestModel,
		DuplicateCheckType duplicateCheckState,
		float &bestMeasurement,
		int &iterationCount,
		vector<int> &providedIndices,
		float confidence = 0.99f
		)
{
	vector<int> usedSamples;

	return CUDA_Estimation(
			EstimationTypeRANSAC,
			estimator,
			src,
			false,
			usedSamples,
			false,
			providedIndices,
			duplicateCheckState,
			bestModel,
			&bestMeasurement,
			&iterationCount,
			true,
			0,
			confidence
			);
}

/**
 * \brief Starts the estimation with adaptive RANSAC. The exact estimation type can be changed by the implementations of the problem-specific functions.
 *
 * \param[in] estimator The estimator delivering the required problem-specific methods. For further information take a look at the LineEstimator
 * \param[in] src Source data in custom order (take care in the CUDA methods computeHypothesis, computeLossValue, ...)
 * \param[out] bestModel The hypothesis with the bestMeasurement value is returned
 * \param[in] duplicateCheckState Determines if a duplicate check is applied. Furthermore, either a index based duplicate check can be done or a index and value based check.
 * \param[out] bestMeasurement The best measurement that has been found in the estimation, e.g. the minimum number of outliers when estimating with RANSAC
 * \param[out] usedSamples Stores the samples that have been used to compute the final hypothesis. This is only used when storeUsedSamples is true
 * \param[out] iterationCount Count of iterationCount the framework has done to finish the estimation
 * \param[in] confidence (Optional) The confidence that should be achieved by the estimation to get a good result (0 < confidence < 1)
 * \return A ResultValueType, e.g. an error state
 */
template <template <int, int, int> class T, int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
ResultValueType FestGPU_Adaptive(
		T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> &estimator,
		vector<float> const &src,
		vector<float> &bestModel,
		DuplicateCheckType duplicateCheckState,
		float &bestMeasurement,
		vector<int> &usedSamples,
		int &iterationCount,
		float confidence = 0.99f
		)
{
	vector<int> providedIndices;

	return CUDA_Estimation(
			EstimationTypeRANSAC,
			estimator,
			src,
			false,
			usedSamples,
			false,
			providedIndices,
			duplicateCheckState,
			bestModel,
			&bestMeasurement,
			&iterationCount,
			true,
			0,
			confidence
			);
}

/**
 * \brief Starts the estimation with adaptive RANSAC. The exact estimation type can be changed by the implementations of the problem-specific functions.
 *
 * \param[in] estimator The estimator delivering the required problem-specific methods. For further information take a look at the LineEstimator
 * \param[in] src Source data in custom order (take care in the CUDA methods computeHypothesis, computeLossValue, ...)
 * \param[out] bestModel The hypothesis with the bestMeasurement value is returned
 * \param[in] duplicateCheckState Determines if a duplicate check is applied. Furthermore, either a index based duplicate check can be done or a index and value based check.
 * \param[out] bestMeasurement The best measurement that has been found in the estimation, e.g. the minimum number of outliers when estimating with RANSAC
 * \param[out] usedSamples Stores the samples that have been used to compute the final hypothesis. This is only used when storeUsedSamples is true
 * \param[out] iterationCount Count of iterationCount the framework has done to finish the estimation
 * \param[in] providedIndices The indices of the datapoints in a specified order. For instance, the order is given by the order of the quality of the datapoints. This is only used when useProvidedIndices is true
 * \param[in] confidence (Optional) The confidence that should be achieved by the estimation to get a good result (0 < confidence < 1)
 * \return A ResultValueType, e.g. an error state
 */
template <template <int, int, int> class T, int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
ResultValueType FestGPU_Adaptive(
		T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> &estimator,
		vector<float> const &src,
		vector<float> &bestModel,
		DuplicateCheckType duplicateCheckState,
		float &bestMeasurement,
		vector<int> &usedSamples,
		int &iterationCount,
		vector<int> &providedIndices,
		float confidence = 0.99f
		)
{
	return CUDA_Estimation(
			EstimationTypeRANSAC,
			estimator,
			src,
			false,
			usedSamples,
			false,
			providedIndices,
			duplicateCheckState,
			bestModel,
			&bestMeasurement,
			&iterationCount,
			true,
			0,
			confidence
			);
}



// ==== LMS WRAPPER ====

/**
 * \brief Starts the estimation with Least Median of Squares (LMS). The exact estimation type can be changed by the implementations of the problem-specific functions.
 *
 * \param[in] estimator The estimator delivering the required problem-specific methods. For further information take a look at the LineEstimator
 * \param[in] src Source data in custom order (take care in the CUDA methods computeHypothesis, computeLossValue, ...)
 * \param[out] bestModel The hypothesis with the bestMeasurement value is returned
 * \param[in] expectedInlierRatio The expected inlier ratio of the dataset src. This is only used when useAdaptiveRansac is true (0 < expectedInlierRatio <= 1)
 * \return A ResultValueType, e.g. an error state
 */
template <template <int, int, int> class T, int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
ResultValueType FestGPU_LMS(
		T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> &estimator,
		const vector<float> &src,
		vector<float> &bestModel,
		float expectedInlierRatio
		)
{
	vector<int> usedSamples;
	vector<int> providedIndices;
	float bestMeasurement;
	int iterationCount;

	return CUDA_Estimation(
			EstimationTypeLMS,
			estimator,
			src,
			false,
			usedSamples,
			false,
			providedIndices,
			DuplicateCheckTypeIndexBased,
			bestModel,
			&bestMeasurement,
			&iterationCount,
			false,
			expectedInlierRatio,
			0.99f
			);
}

/**
 * \brief Starts the estimation with Least Median of Squares (LMS). The exact estimation type can be changed by the implementations of the problem-specific functions.
 *
 * \param[in] estimator The estimator delivering the required problem-specific methods. For further information take a look at the LineEstimator
 * \param[in] src Source data in custom order (take care in the CUDA methods computeHypothesis, computeLossValue, ...)
 * \param[out] bestModel The hypothesis with the bestMeasurement value is returned
 * \param[in] expectedInlierRatio The expected inlier ratio of the dataset src. This is only used when useAdaptiveRansac is true (0 < expectedInlierRatio <= 1)
 * \param[in] duplicateCheckState Determines if a duplicate check is applied. Furthermore, either a index based duplicate check can be done or a index and value based check.
 * \param[out] bestMeasurement The best measurement that has been found in the estimation, e.g. the minimum number of outliers when estimating with RANSAC
 * \param[out] iterationCount Count of iterationCount the framework has done to finish the estimation
 * \param[in] confidence (Optional) The confidence that should be achieved by the estimation to get a good result (0 < confidence < 1)
 * \return A ResultValueType, e.g. an error state
 */
template <template <int, int, int> class T, int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
ResultValueType FestGPU_LMS(
		T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> &estimator,
		const vector<float> &src,
		vector<float> &bestModel,
		float expectedInlierRatio, // this is for the non-adaptive case
		DuplicateCheckType duplicateCheckState,
		float &bestMeasurement,
		int &iterationCount,
		float confidence = 0.99f
		)
{
	vector<int> usedSamples;
	vector<int> providedIndices;

	return CUDA_Estimation(
			EstimationTypeLMS,
			estimator,
			src,
			false,
			usedSamples,
			false,
			providedIndices,
			duplicateCheckState,
			bestModel,
			&bestMeasurement,
			&iterationCount,
			false,
			expectedInlierRatio,
			confidence
			);
}


// ==== Inlier Computation WRAPPER ====

/**
 * \brief This method is for separate inlier computation. With a given model, threshold and the source data the inlier indices are computed.
 *
 * \param[in] estimator The estimator delivering the required problem-specific methods. For further information take a look at the LineEstimator
 * \param[in] src The datasource
 * \param[in] model Model to compute the goodness values from the datasource
 * \param[in] threshold Threshold that separates between inliers and outliers related to the loss value
 * \param[out] inliers Inlier array with 1 = inlier and 0 = outlier
 * \return A ResultValueType, e.g. an error state
*/
template <template <int, int, int> class T, int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
ResultValueType FestGPU_ComputeInliers(
		T<PARAMS_REQUIRED, DATA_PT_SIZE, HYPOTHESIS_SIZE> &estimator,
		const vector<float> &src,
		vector<float> &model,
		float threshold,
		vector<int> &inliers
		)
{
	return CUDA_Get_Inliers(
			estimator,
			src,
			model,
			threshold,
			inliers);
}







