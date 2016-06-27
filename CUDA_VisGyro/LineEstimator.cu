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


/*!
 * \brief Estimator for estimating lines from a set of 2D points.
 *
 * This class is defined for RANSAC estimation since the loss value computation
 * returns either 0 or 1.
 */
template <int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
class LineEstimator
{
public: // constructor / destructor
	/*!
	 * \brief Default constructor
	 * \param[in] ransac_threshold The threshold for RANSAC estimation
	 */
	LineEstimator(float ransac_threshold)
	: _ransac_threshold(ransac_threshold)
	{ }

	/*!
	 * \brief Constructor with an options array.
	 * \param[in] options The options are defined by the user such as this class. For instance, it may contain the threshold for RANSAC estimations
	 */
	LineEstimator(vector<float> &options)
	{
		_ransac_threshold = options[0];
	}

	virtual ~LineEstimator() { }

private: // instance attributes
	float _ransac_threshold;


public: // methods for estimation

	/*!
	 * \brief Device function to compute a hypothesis from a subset of datapoints
	 * \param[in] inDataPts A subset of datapoints that can be used to compute a hypothesis. The size should be k times s where k is the number of required datapoints to compute a hypothesis and s is the size of each datapoint, e.g. k=2 for 2D line estimation
	 * \param[out] outHypothesis The hypothesis that has been computed. The size is equal to HYPOTHESIS_SIZE (h in the paper)
	 */
	__device__ void computeHypothesis(float *inDataPts, float *outHypothesis)
	{
		outHypothesis[0] = inDataPts[0];
		outHypothesis[1] = inDataPts[1];
		outHypothesis[2] = inDataPts[2];
		outHypothesis[3] = inDataPts[3];
	}

	/*!
	 * \brief Device function to compute the loss value of a hypothesis to a datapoint
	 * \param[in] inHypothesis The hypothesis that has been computed previously. The size is equal to HYPOTHESIS_SIZE (h in the paper)
	 * \param[in] inDataPt A subset of datapoints that can be used to compute a hypothesis. The size should be k times s where k is the number of required datapoints to compute a hypothesis and s is the size of each datapoint, e.g. k=2 for 2D line estimation
	 * \returns The loss value of the datapoint related to the hypothesis
	 */
	__device__ float computeLossValue(float *inHypothesis, const float *inDataPt)
	{
		float sub_x2_x1 = inHypothesis[2]-inHypothesis[0];
		float sub_y2_y1 = inHypothesis[3]-inHypothesis[1];
		float tmp = sub_x2_x1*(inHypothesis[1]-inDataPt[1]) - (inHypothesis[0]-inDataPt[0])*sub_y2_y1;

		float d = tmp*tmp / (sub_x2_x1*sub_x2_x1+sub_y2_y1*sub_y2_y1);
		return d > _ransac_threshold; // compare with threshold
	}

	/*!
	 * \brief Device function to check if the chosen subset of datapoints is degenerated
	 * \param[in] inDataPts A subset of datapoints that has to be checked if it is degenerated. The size should be k times s where k is the number of required datapoints to compute a hypothesis and s is the size of each datapoint, e.g. k=2 for 2D line estimation
	 * \returns True if the subset is degenerated
	 */
	__device__ bool isDegenerated(const float *inDataPts)
	{
		return false;
	}
};

