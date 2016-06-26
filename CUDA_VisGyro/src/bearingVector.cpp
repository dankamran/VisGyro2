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

#include "bearingVector.h"
#include "math.h"
#include <stdio.h>
#include <fstream>
bearingVector::bearingVector(const std::vector<Matcher::p_match> &p_matched,const std::vector<int32_t> &active,Matrix KInv) {
	//calculate bearing vector for each 2d-2d correspondent
	// number of active p_matched
    int32_t N = active.size();

    //calculate bearing vector for each matched point
    for (int32_t i=0; i<N; i++) {
    	Matrix bvc(3,1);
    	Matrix bvp(3,1);
    	Matcher::p_match m = p_matched[active[i]];
    	bvc.val[0][0]=m.u1c;
    	bvc.val[1][0]=m.v1c;
    	bvc.val[2][0]=1;

    	bvp.val[0][0]=m.u1p;
    	bvp.val[1][0]=m.v1p;
    	bvp.val[2][0]=1;

    	bvp = KInv * bvp;
    	bvc = KInv * bvc;

    	float valuec = sqrt(pow(bvc.val[0][0],2)+pow(bvc.val[1][0],2)+pow(bvc.val[2][0],2));
    	bvc = bvc / valuec;

		float valuep = sqrt(pow(bvp.val[0][0],2)+pow(bvp.val[1][0],2)+pow(bvp.val[2][0],2));
		bvp = bvp / valuep;

    	BVsC.push_back(bvc);
    	BVsP.push_back(bvp);
    }



}


//constructor for all correspondences
bearingVector::bearingVector(const std::vector<Matcher::p_match> &p_matched,Matrix KInv) {
	//calculate bearing vector for each 2d-2d correspondent
	// number of active p_matched
    int32_t N = p_matched.size();

    //calculate bearing vector for each matched point
    for (int32_t i=0; i<N; i++) {
    	Matrix bvc(3,1);
    	Matrix bvp(3,1);
    	Matcher::p_match m = p_matched[i];
    	//mexPrintf("u1c:%f,v1c:%f,u1p:%f,v1p:%f",m.u1c,m.v1c,m.u1p,m.v1p);
    	bvc.val[0][0]=m.u1c;
    	bvc.val[1][0]=m.v1c;
    	bvc.val[2][0]=1;

    	bvp.val[0][0]=m.u1p;
    	bvp.val[1][0]=m.v1p;
    	bvp.val[2][0]=1;

    	bvp = KInv * bvp;
    	bvc = KInv * bvc;

    	float valuec = sqrt(pow(bvc.val[0][0],2)+pow(bvc.val[1][0],2)+pow(bvc.val[2][0],2));
    	bvc = bvc / valuec;

		float valuep = sqrt(pow(bvp.val[0][0],2)+pow(bvp.val[1][0],2)+pow(bvp.val[2][0],2));
		bvp = bvp / valuep;

    	BVsC.push_back(bvc);
    	BVsP.push_back(bvp);
    }



}

void bearingVector::computeCenter_c()
{
	Matrix center(3,1);
	center.val[0][0]=0;
	center.val[1][0]=0;
	center.val[2][0]=0;
	float size=BVsC.size();

	for(int32_t i=0; i<BVsC.size();i++)
	{
		center = center + BVsC[i];

	}
	center = center / size;
	center_c = center;

	return;
}

void bearingVector::computeCenter_p()
{
	Matrix center(3,1);
	center.val[0][0]=0;
	center.val[1][0]=0;
	center.val[2][0]=0;
	float size=BVsP.size();

	for(int32_t i=0; i<BVsP.size();i++)
	{
		center = center + BVsP[i];

	}
	center = center / size;
	center_p = center;
	return;
}

void bearingVector::minuseCenter_p()
{
	for(int32_t i=0; i<BVsP.size();i++)
	{
		BVsP_normal.push_back(BVsP[i]-center_p);

	}
	return;
}

void bearingVector::minuseCenter_c()
{
	for(int32_t i=0; i<BVsC.size();i++)
	{
		BVsC_normal.push_back(BVsC[i]-center_c);

	}
	return;
}

void bearingVector::computeH()
{
	Matrix H_final(3,3);
	for(int i=0;i<3;i++)
	{
		H_final.val[i][0]=0;
		H_final.val[i][1]=0;
		H_final.val[i][2]=0;
	}
	for(int32_t i=0; i<BVsC_normal.size();i++)
	{
		Matrix h_temp = BVsP_normal[i] * ~BVsC_normal[i];
		H_final = H_final + h_temp;
	}
	H = H_final;
	//mexPrintf("H: %f %f %f;%f %f %f;%f %f %f\n",H.val[0][0],H.val[0][1],H.val[0][2],H.val[1][0],H.val[1][1],H.val[1][2],H.val[2][0],H.val[2][1],H.val[2][2]);



	return;
}

Matrix bearingVector::computeR()
{
	Matrix U,W,V;
	H.svd(U,W,V);
	R =  V * ~ U;
	//if(R.det()<0)
	{
		//printf("error\n");
	//	R.val[0][2]= -1*R.val[0][2];
	//	R.val[1][2]= -1*R.val[1][2];
	//	R.val[2][2]= -1*R.val[2][2];
	}
	return R;
}





std::vector<int32_t> bearingVector::getInliers(Matrix Rtest,double threshold)
{
	std::vector<int32_t> inliers;
	for(int32_t i=0; i< BVsC_normal.size();i++)
	{
		Matrix error = (BVsC_normal[i]*1000) - (Rtest * BVsP_normal[i]*1000);
		double sum_error = abs(error.val[0][0]) + abs(error.val[0][1])+ abs(error.val[0][2])+ abs(error.val[1][0])+ abs(error.val[1][1])+ abs(error.val[1][2])+ abs(error.val[2][0])+ abs(error.val[2][1])+ abs(error.val[2][2]);
		if(sum_error<threshold)
		{
			//mexPrintf("error: %f %f %f;%f %f %f;%f %f %f\n",error.val[0][0],error.val[0][1],error.val[0][2],error.val[1][0],error.val[1][1],error.val[1][2],error.val[2][0],error.val[2][1],error.val[2][2]);
			inliers.push_back(i);
		}
	}
	return inliers;

}


std::vector <float>bearingVector::get_rpy()
{
	std::vector <float> rpy(3);
	double r= atan2((double)R.val[2][1],(double)R.val[2][2]);
	double temp=pow((double)R.val[2][1],(double)2) + pow((double)R.val[2][2],(double)2);
	double p=atan2((double)(-1*R.val[2][0]),(double)temp);
	double y=atan2((double)R.val[1][0],(double)R.val[0][0]);
	rpy[0]=r;
	rpy[1]=p;
	rpy[2]=y;
	return rpy;
}

std::vector <float>bearingVector::get_rpy(Matrix R)
{
	std::vector <float> rpy(3);
	double r= atan2((double)R.val[2][1],(double)R.val[2][2]);
	double temp=pow((double)R.val[2][1],(double)2) + pow((double)R.val[2][2],(double)2);
	double p=atan2((double)(-1*R.val[2][0]),(double)temp);
	double y=atan2((double)R.val[1][0],(double)R.val[0][0]);
	rpy[0]=r;
	rpy[1]=p;
	rpy[2]=y;
	return rpy;
}
float bearingVector::get_yaw_scar()
{
    if(BVsC.size()!=1)
        std::cout<<"error not one bv"<<std::endl;
    ///using scaramuzza 2009 formula for estimating yaw by just one point:
    float x1=BVsP[0].val[0][0];
    float y1=BVsP[0].val[1][0];
    float z1=BVsP[0].val[2][0];

    float x2=BVsC[0].val[0][0];
    float y2=BVsC[0].val[1][0];
    float z2=BVsC[0].val[2][0];

    double t1 = y2*z1 - z2*y1;
    double t2 = x2*z1 + z2*x1;
    double yaw = -2*atan(t1/t2);
    return yaw;
}
bearingVector::~bearingVector() {
	// TODO Auto-generated destructor stub
}

