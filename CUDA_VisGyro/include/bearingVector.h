/*
Copyright 2016. All rights reserved.
DSP Lab
Sharif University of Technology, Iran

This file is part of VisGyro.
Authors: Danial Kamran

VisGyro is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or any later version.

VisGyro is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
*/

#include "../matcher.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "matrix.h"
#ifndef BEARINGVECTOR_H_
#define BEARINGVECTOR_H_

class bearingVector {
	//one bearing vector computing from K^-1 * (u,v,1)
	std::vector<Matrix> BVsC;
	std::vector<Matrix> BVsP;
	std::vector<Matrix> BVsC_normal;
	std::vector<Matrix> BVsP_normal;
	Matrix center_c;
	Matrix center_p;
	Matrix H,R;
public:
	bearingVector(const std::vector<Matcher::p_match> &p_matched,const std::vector<int32_t> &active,Matrix KInv);
	bearingVector(const std::vector<Matcher::p_match> &p_matched,Matrix KInv);
	void computeCenter_p();
	void computeCenter_c();
	void minuseCenter_c();
	void minuseCenter_p();
	virtual ~bearingVector();
	void computeH();
	Matrix computeR();
	float get_yaw_scar();
	std::vector<float> get_rpy();
	std::vector<float> get_rpy(Matrix R);
	std::vector<int32_t> getInliers(Matrix R, double threshold);
};

#endif /* BEARINGVECTOR_H_ */
