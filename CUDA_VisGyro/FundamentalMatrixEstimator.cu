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
 * \brief Estimator for estimating the fundamental matrix from a set of 2D points correspondences.
 *
 * This class is defined for RANSAC estimation since the loss value computation
 * returns either 0 or 1.
 */
template <int PARAMS_REQUIRED, int DATA_PT_SIZE, int HYPOTHESIS_SIZE>
class FundamentalMatrixEstimator
{
public: // constructor / destructor
	/*!
	 * \brief Default constructor
	 * \param[in] ransac_threshold The threshold for RANSAC estimation
	 */
	FundamentalMatrixEstimator(float ransac_threshold)
	: _ransac_threshold(ransac_threshold)
	{ }

	/*!
	 * \brief Constructor with an options array.
	 * \param[in] options The options are defined by the user such as this class. For instance, it may contain the threshold for RANSAC estimations
	 */
	FundamentalMatrixEstimator(vector<float> &options)
	{
		_ransac_threshold = options[0];
	}

	virtual ~FundamentalMatrixEstimator() { }

private: // instance attributes
	int _ransac_threshold;


public: // methods for estimation

	/*!
	 * \brief Device function to compute a hypothesis from a subset of datapoints
	 * \param[in] inDataPts A subset of datapoints that can be used to compute a hypothesis. The size should be k times s where k is the number of required datapoints to compute a hypothesis and s is the size of each datapoint, e.g. k=2 for 2D line estimation
	 * \param[out] outHypothesis The hypothesis that has been computed. The size is equal to HYPOTHESIS_SIZE (h in the paper)
	 */
	__device__ void computeHypothesis(float *inDataPts, float *outHypothesis)
	{
		int i;

		float src[DATA_PT_SIZE*PARAMS_REQUIRED];
	#pragma unroll
		for (i=0; i < DATA_PT_SIZE*PARAMS_REQUIRED; ++i)
		{
			src[i] = inDataPts[i];
		}

		// compute the centroid of the points
		// assumed that the points are NOT at infinity (z > EPS)
		float mean_x1 = 0.0f;
		float mean_y1 = 0.0f;
		float mean_x2 = 0.0f;
		float mean_y2 = 0.0f;
	#pragma unroll
		for (i=0; i < DATA_PT_SIZE*PARAMS_REQUIRED; i+=DATA_PT_SIZE)
		{
			mean_x1 += src[i];
			mean_y1 += src[i+1];
			mean_x2 += src[i+2];
			mean_y2 += src[i+3];
		}
		float ONE_FMATRIX_PARAMS_REQUIRED = 1.0f / PARAMS_REQUIRED;
		mean_x1 *= ONE_FMATRIX_PARAMS_REQUIRED;
		mean_y1 *= ONE_FMATRIX_PARAMS_REQUIRED;
		mean_x2 *= ONE_FMATRIX_PARAMS_REQUIRED;
		mean_y2 *= ONE_FMATRIX_PARAMS_REQUIRED;

		// shift points about -(mean_x, mean_y)
		// and compute the standard aviation
		float mean_pt_scale_term1 = 0.0f;
		float mean_pt_scale_term2 = 0.0f;
	#pragma unroll
		for (i=0; i < DATA_PT_SIZE*PARAMS_REQUIRED; i+=DATA_PT_SIZE)
		{
			src[i]   -= mean_x1;
			src[i+1] -= mean_y1;
			src[i+2] -= mean_x2;
			src[i+3] -= mean_y2;
			mean_pt_scale_term1 += sqrtf(src[i]*src[i] + src[i+1]*src[i+1]);
			mean_pt_scale_term2 += sqrtf(src[i+2]*src[i+2] + src[i+3]*src[i+3]);
		}

		// scale the points with mean_pt_scale sqrt(2) = 1.41421356f, mean_pt_scale_term1/PARAMS_REQUIRED
		mean_pt_scale_term1 = (PARAMS_REQUIRED*1.41421356f)/mean_pt_scale_term1;
		mean_pt_scale_term2 = (PARAMS_REQUIRED*1.41421356f)/mean_pt_scale_term2;
		for (i=0; i < DATA_PT_SIZE*PARAMS_REQUIRED; i+=DATA_PT_SIZE)
		{
			src[i]   *= mean_pt_scale_term1;
			src[i+1] *= mean_pt_scale_term1;
			src[i+2] *= mean_pt_scale_term2;
			src[i+3] *= mean_pt_scale_term2;
		}

	//    A = [x2(1,:)'.*x1(1,:)'   x2(1,:)'.*x1(2,:)'  x2(1,:)' ...
	//         x2(2,:)'.*x1(1,:)'   x2(2,:)'.*x1(2,:)'  x2(2,:)' ...
	//         x1(1,:)'             x1(2,:)'            ones(npts,1) ];

		float A[9*9]; // required 9 rows and 8 columns, but for dsvd 9x9
	#pragma unroll
		for (i=0; i<PARAMS_REQUIRED; ++i)
		{
			// src[xxx + 0] = x of 1st image
			// src[xxx + 3] = x of 2nd image
			A[i + 0*PARAMS_REQUIRED] = src[DATA_PT_SIZE*i + 2] * src[DATA_PT_SIZE*i + 0];
			A[i + 1*PARAMS_REQUIRED] = src[DATA_PT_SIZE*i + 2] * src[DATA_PT_SIZE*i + 1];
			A[i + 2*PARAMS_REQUIRED] = src[DATA_PT_SIZE*i + 2]; // * 1.0f
			A[i + 3*PARAMS_REQUIRED] = src[DATA_PT_SIZE*i + 3] * src[DATA_PT_SIZE*i + 0];
			A[i + 4*PARAMS_REQUIRED] = src[DATA_PT_SIZE*i + 3] * src[DATA_PT_SIZE*i + 1];
			A[i + 5*PARAMS_REQUIRED] = src[DATA_PT_SIZE*i + 3]; // * 1.0f
			A[i + 6*PARAMS_REQUIRED] = src[DATA_PT_SIZE*i + 0]; // 1.0f * ...
			A[i + 7*PARAMS_REQUIRED] = src[DATA_PT_SIZE*i + 1]; // 1.0f * ...
			A[i + 8*PARAMS_REQUIRED] = 1.0f; // * 1.0f
		}

		float Vt[PARAMS_REQUIRED*PARAMS_REQUIRED];
		float sigma[PARAMS_REQUIRED];

		// stores U in A, sigma is a 8 comp vector storing the singular
		// values and Vt is the translated right side matrix
		dsvd(A, 9,PARAMS_REQUIRED, sigma, Vt);

		float At[9*9];
		transposeMatrix(A, 9, 8, At);
		fill_cols_of_U(At, 9, 8);

		// F contains the elements of the last column of U
		float Ftmp[9];
		for (i=0; i<9; ++i)
		{
			Ftmp[i] = At[72+i];
		}


	    //[U,D,V] = svd(F,0);
		float D[3];
		float Vt2[9];
		dsvd(Ftmp, 3, 3, D, Vt2);

		// set smallest singular value to zero!
		D[((D[1]<D[2] && D[1]<D[0]) + 2*(D[2]<D[0] && D[2]<D[1]))] = 0.0f;


		// create diagonal matrix
		float DMat[9];
		DMat[0] = D[0]; // 0,0
		DMat[1] = 0.0f; // 0,1
		DMat[2] = 0.0f; // 0,2
		DMat[3] = 0.0f; // 1,0
		DMat[4] = D[1]; // 1,1
		DMat[5] = 0.0f; // 1,2
		DMat[6] = 0.0f; // 2,0
		DMat[7] = 0.0f; // 2,1
		DMat[8] = D[2]; // 2,2


		float F[9];
		float Ftmp2[9];
	    //F = U*diag([D(1,1) D(2,2) 0])*V';
		float V2[9];
		transposeMatrix(Vt2, 3,3, V2);
		matrixMatrixMult(Ftmp, 3,3, DMat, 3, Ftmp2);
		matrixMatrixMult(Ftmp2, 3,3, V2, 3, F);


		float T1[9];
		T1[0] = mean_pt_scale_term1;
		T1[1] = 0.0f;
		T1[2] = -mean_pt_scale_term1*mean_x1;
		T1[3] = 0.0f;
		T1[4] = mean_pt_scale_term1;
		T1[5] = -mean_pt_scale_term1*mean_y1;
		T1[6] = 0.0f;
		T1[7] = 0.0f;
		T1[8] = 1.0f;

		float T2[9];
		T2[0] = mean_pt_scale_term2;
		T2[1] = 0.0f;
		T2[2] = -mean_pt_scale_term2*mean_x2;
		T2[3] = 0.0f;
		T2[4] = mean_pt_scale_term2;
		T2[5] = -mean_pt_scale_term2*mean_y2;
		T2[6] = 0.0f;
		T2[7] = 0.0f;
		T2[8] = 1.0f;


		// Denormalize
	    //F = T2'*F*T1;
		float Tt2[9];
		matrixMatrixMult(F, 3, 3, T1, 3, Ftmp);
		transposeMatrix(T2, 3, 3, Tt2);
		matrixMatrixMult(Tt2, 3, 3, Ftmp, 3, F);

	#pragma unroll
		for (i=0; i<HYPOTHESIS_SIZE; ++i)
		{
			outHypothesis[i] = F[i];
		}
	}

	/*!
	 * \brief Device function to compute the loss value of a hypothesis to a datapoint
	 * \param[in] inHypothesis The hypothesis that has been computed previously. The size is equal to HYPOTHESIS_SIZE (h in the paper)
	 * \param[in] inDataPt A subset of datapoints that can be used to compute a hypothesis. The size should be k times s where k is the number of required datapoints to compute a hypothesis and s is the size of each datapoint, e.g. k=2 for 2D line estimation
	 * \returns The loss value of the datapoint related to the hypothesis
	 */
	__device__ float computeLossValue(float *inHypothesis, const float *inDataPt)
	{
		//x1 = x(1:3,:);  // x1 = test_set[0+x]
		//x2 = x(4:6,:);  // x2 = test_set[3+x]
		float Fx1[3];
		float Ftx2[3];

		// Fx1 = F*x1;
		// Ftx2 = F'*x2;
		Fx1[0]  = inHypothesis[0] * inDataPt[0] + inHypothesis[1] * inDataPt[1] + inHypothesis[2];
		Fx1[1]  = inHypothesis[3] * inDataPt[0] + inHypothesis[4] * inDataPt[1] + inHypothesis[5];
		Fx1[2]  = inHypothesis[6] * inDataPt[0] + inHypothesis[7] * inDataPt[1] + inHypothesis[8];
		Ftx2[0] = inHypothesis[0] * inDataPt[2] + inHypothesis[3] * inDataPt[3] + inHypothesis[6];
		Ftx2[1] = inHypothesis[1] * inDataPt[2] + inHypothesis[4] * inDataPt[3] + inHypothesis[7];
		Ftx2[2] = inHypothesis[2] * inDataPt[2] + inHypothesis[5] * inDataPt[3] + inHypothesis[8];

		// compute inner product of Fx1 and Ftx2 to get
		// x2tFx1(n) = x2'*F*x1;
		float x2tFx1 = inDataPt[0]*Ftx2[0]+inDataPt[1]*Ftx2[1]+ Ftx2[2];

		// Evaluate distances
		//d = x2tFx1.^2 ./ ...
		//    (Fx1(1,:).^2 + Fx1(2,:).^2 + Ftx2(1,:).^2 + Ftx2(2,:).^2);
		float d = x2tFx1*x2tFx1 /
				(Fx1[0]*Fx1[0] + Fx1[1]*Fx1[1] + Ftx2[0]*Ftx2[0] + Ftx2[1]*Ftx2[1]);
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



private: // helper functions

	__device__ int fill_cols_of_U(float *U, int m, int n)
	{
		float s = 0;
		int i = n; // last col (transposed: row)
		int stride = m; // stride col size (# of rows)
		int iter, j, k;
		float sd;

		while( s == 0 )
		{
			// if we got a zero singular value, then in order to get the corresponding left singular vector
			// we generate a random vector, project it to the previously computed left singular vectors,
			// subtract the projection and normalize the difference.
			float val0 = 1.0f/m;
			for( k = 0; k < m; k++ )
			{
				//float val = (rng.next() & 256) ? val0 : -val0;
				//float val = val0;
				U[i*stride + k] = val0;
			}
			int stridei, stridej;
			float t, asum;
			for( iter = 0; iter < 2; iter++ )
			{
				stridei = i*stride;
				for( j = 0; j < i; j++ )
				{
					stridej = j*stride;
					sd = 0;
					for( k = 0; k < m; k++ )
						sd += U[stridei + k]*U[stridej + k];
					asum = 0.0f;
					for( k = 0; k < m; k++ )
					{
						t = U[stridei + k] - sd*U[stridej + k];
						U[stridei + k] = t;
						asum += fabsf(t);
					}
					asum = asum ? 1.0f/asum : 0.0f;
					for( k = 0; k < m; k++ )
						U[stridei + k] *= asum;
				}
			}
			sd = 0;
			for( k = 0; k < m; k++ )
			{
				t = U[i*stride + k];
				sd += t*t;
			}
			s = sqrtf(sd);
		}

		s = 1/s;
		for( k = 0; k < m; k++ )
			U[i*stride + k] *= s;

		return 0;
	}


	__device__ void matrixMatrixMult(float *lMat, int lRows, int lCols,
			float *rMat, int rCols, float *result)
	{
		//int rRows = lCols;
		for (int rC=0; rC<rCols; ++rC)
		{
			for (int lR=0; lR<lRows; ++lR)
			{
				result[lR*rCols+rC] = 0.0f;
				for (int lC_rR=0; lC_rR<lCols; ++lC_rR)
				{
					result[lR*rCols+rC] += lMat[lR*lCols+lC_rR]*rMat[lC_rR*rCols + rC];
				}
			}
		}
	}

	__device__ void transposeMatrix(float *mat, int rows, int cols, float *res)
	{
		for (int r=0; r<rows; ++r)
		{
			for (int c=0; c<cols; ++c)
			{
				res[c*rows + r] = mat[r*cols + c];
			}
		}
	}

	__device__ void transposeMatrixSquare(float *mat, int rows, int cols)
	{
		float tmp;
		for (int r=0; r<rows; ++r)
		{
			for (int c=0; c<cols; ++c)
			{
				tmp = mat[r*cols + c];
				mat[r*cols + c] = mat[c*cols + r];
				mat[c*cols + r] = tmp;
			}
		}
	}



	// Port of SVD code from:
	// http://www.public.iastate.edu/~dicook/JSS/paper/code/svd.c
	// with modifications by me to use single * pointers instead of float pointers

	// Call int dsvd(float *a, int m, int n, float *w, float *v)
	// Returns 1 on sucess, anything else is a fail.

	// LIMITATIONS:
	// Largest matrix size has to be known in advance. The original code used malloc to
	// declare a temporary variable. Future version of CUDA will probably support
	// dynamic mmory.

	// Thos SVD code requires rows >= columns.
	#define SVD_ROW_COUNT 9 // rows
	#define SVD_COL_COUNT 9 // cols

	__device__ float SIGN(float a, float b)
	{
		if(b > 0) {
			return fabs(a);
		}

		return -fabs(a);
	}

	__device__ float PYTHAG(float a, float b)
	{
		float at = fabs(a), bt = fabs(b), ct, result;

		if (at > bt)       { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
		else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
		else result = 0.0;
		return(result);
	}

	__device__ float MAX(float a, float b)
	{
		if (a > b)
			return a;
		return b;
	}

	// Returns 1 on success, fail otherwise
	__device__ int dsvd(float *a, int m, int n, float *w, float *v)
	{
		int flag, i, its, j, jj, k, l, nm;
		float c, f, h, s, x, y, z;
		float anorm = 0.0, g = 0.0, scale = 0.0;
		float rv1[SVD_COL_COUNT];

	/* Householder reduction to bidiagonal form */
		for (i = 0; i < n; i++)
		{
			/* left-hand reduction */
			l = i + 1;
			rv1[i] = scale * g;
			g = s = scale = 0.0;
			if (i < m)
			{
				for (k = i; k < m; k++)
					scale += fabs(a[k*n+i]);
				if (scale)
				{
					for (k = i; k < m; k++)
					{
						a[k*n+i] = (a[k*n+i]/scale);
						s += (a[k*n+i] * a[k*n+i]);
					}
					f = a[i*n+i];
					g = -SIGN(sqrt(s), f);
					h = f * g - s;
					a[i*n+i] = f - g;
					if (i != n - 1)
					{
						for (j = l; j < n; j++)
						{
							for (s = 0.0, k = i; k < m; k++)
								s += (a[k*n+i] * a[k*n+j]);
							f = s / h;
							for (k = i; k < m; k++)
								a[k*n+j] += (f * a[k*n+i]);
						}
					}
					for (k = i; k < m; k++)
						a[k*n+i] = a[k*n+i]*scale;
				}
			}
			w[i] = scale * g;

			/* right-hand reduction */
			g = s = scale = 0.0;
			if (i < m && i != n - 1)
			{
				for (k = l; k < n; k++)
					scale += fabs(a[i*n+k]);
				if (scale)
				{
					for (k = l; k < n; k++)
					{
						a[i*n+k] = a[i*n+k]/scale;
						s += a[i*n+k] * a[i*n+k];
					}
					f = a[i*n+l];
					g = -SIGN(sqrt(s), f);
					h = f * g - s;
					a[i*n+l] = f - g;
					for (k = l; k < n; k++)
						rv1[k] = a[i*n+k] / h;
					if (i != m - 1)
					{
						for (j = l; j < m; j++)
						{
							for (s = 0.0, k = l; k < n; k++)
								s += a[j*n+k] * a[i*n+k];
							for (k = l; k < n; k++)
								a[j*n+k] += s * rv1[k];
						}
					}
					for (k = l; k < n; k++)
						a[i*n+k] = a[i*n+k]*scale;
				}
			}
			anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
		}

		/* accumulate the right-hand transformation */
		for (i = n - 1; i >= 0; i--)
		{
			if (i < n - 1)
			{
				if (g)
				{
					for (j = l; j < n; j++)
						v[j*n+i] = a[i*n+j] / (a[i*n+l] * g);
						/* float division to avoid underflow */
					for (j = l; j < n; j++)
					{
						for (s = 0.0, k = l; k < n; k++)
							s += a[i*n+k] * v[k*n+j];
						for (k = l; k < n; k++)
							v[k*n+j] += s * v[k*n+i];
					}
				}
				for (j = l; j < n; j++)
					v[i*n+j] = v[j*n+i] = 0.0;
			}
			v[i*n+i] = 1.0;
			g = rv1[i];
			l = i;
		}

		/* accumulate the left-hand transformation */
		for (i = n - 1; i >= 0; i--)
		{
			l = i + 1;
			g = w[i];
			if (i < n - 1)
				for (j = l; j < n; j++)
					a[i*n+j] = 0.0;
			if (g)
			{
				g = 1.0 / g;
				if (i != n - 1)
				{
					for (j = l; j < n; j++)
					{
						for (s = 0.0, k = l; k < m; k++)
							s += a[k*n+i] * a[k*n+j];
						f = (s / a[i*n+i]) * g;
						for (k = i; k < m; k++)
							a[k*n+j] += f * a[k*n+i];
					}
				}
				for (j = i; j < m; j++)
					a[j*n+i] = a[j*n+i]*g;
			}
			else
			{
				for (j = i; j < m; j++)
					a[j*n+i] = 0.0;
			}
			++a[i*n+i];
		}

		/* diagonalize the bidiagonal form */
		for (k = n - 1; k >= 0; k--)
		{                             /* loop over singular values */
			for (its = 0; its < 30; its++)
			{                         /* loop over allowed iterations */
				flag = 1;
				for (l = k; l >= 0; l--)
				{                     /* test for splitting */
					nm = l - 1;
					if (fabs(rv1[l]) + anorm == anorm)
					{
						flag = 0;
						break;
					}
					if (fabs(w[nm]) + anorm == anorm)
						break;
				}
				if (flag)
				{
					c = 0.0;
					s = 1.0;
					for (i = l; i <= k; i++)
					{
						f = s * rv1[i];
						if (fabs(f) + anorm != anorm)
						{
							g = w[i];
							h = PYTHAG(f, g);
							w[i] = h;
							h = 1.0 / h;
							c = g * h;
							s = (- f * h);
							for (j = 0; j < m; j++)
							{
								y = a[j*n+nm];
								z = a[j*n+i];
								a[j*n+nm] = y * c + z * s;
								a[j*n+i] = z * c - y * s;
							}
						}
					}
				}
				z = w[k];
				if (l == k)
				{                  /* convergence */
					if (z < 0.0)
					{              /* make singular value nonnegative */
						w[k] = -z;
						for (j = 0; j < n; j++)
							v[j*n+k] = (-v[j*n+k]);
					}
					break;
				}
				if (its >= 30) {
					//free((void*) rv1);
					//fprintf(stderr, "No convergence after 30,000! iterations \n");
					return(0);
				}

				/* shift from bottom 2 x 2 minor */
				x = w[l];
				nm = k - 1;
				y = w[nm];
				g = rv1[nm];
				h = rv1[k];
				f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
				g = PYTHAG(f, 1.0);
				f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

				/* next QR transformation */
				c = s = 1.0;
				for (j = l; j <= nm; j++)
				{
					i = j + 1;
					g = rv1[i];
					y = w[i];
					h = s * g;
					g = c * g;
					z = PYTHAG(f, h);
					rv1[j] = z;
					c = f / z;
					s = h / z;
					f = x * c + g * s;
					g = g * c - x * s;
					h = y * s;
					y = y * c;
					for (jj = 0; jj < n; jj++)
					{
						x = v[jj*n+j];
						z = v[jj*n+i];
						v[jj*n+j] = x * c + z * s;
						v[jj*n+i] = z * c - x * s;
					}
					z = PYTHAG(f, h);
					w[j] = z;
					if (z)
					{
						z = 1.0 / z;
						c = f * z;
						s = h * z;
					}
					f = (c * g) + (s * y);
					x = (c * y) - (s * g);
					for (jj = 0; jj < m; jj++)
					{
						y = a[jj*n+j];
						z = a[jj*n+i];
						a[jj*n+j] = y * c + z * s;
						a[jj*n+i] = z * c - y * s;
					}
				}
				rv1[l] = 0.0;
				rv1[k] = f;
				w[k] = x;
			}
		}
		return(1);
	}

};




