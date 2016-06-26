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
#ifndef IMAGE_PROVIDER_H
#define IMAGE_PROVIDER_H
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "../matcher.h"
#include "matrix.h"
#include"bearingVector.h"
#include"libviso_F.h"
#include"libviso_geometry.h"
#include <fstream>
#include "../DataTypes.h"
#include <stdio.h>
using namespace std;
using namespace cv;
class image_provider
{
    int32_t argc;
    char** argv;
    int32_t frame;
    string first_path;
    string extention;
    int32_t frames_num;
    Mat newImg,grey;
    Matcher* matcher;
    std::vector<Matcher::p_match>  p_matched;  // feature point32_t matches
    vector <Point2Df> src, dst;
    vector <int> match_index;
    vector <float> match_score;
    Matrix F_gpu,F_serial,K,R_gpu,R_serial,F_serial_voting;
    FILE* frpy_gpu,*frpy_serial,*frpy_Arun,*frpy_voting,*frpy_arun_voting,*frpy_arun_voting2,*frpy_scar_voting;
    std::fstream callib_file;
    vector <char> inlier_mask_serial;
    vector <char> inlier_mask_gpu;
    float yaw_voting,arun_voting,arun_voting2,scar_voting,yaw_gpu;
    float drift;
    vector<int32_t> inliers_serial;
    Matrix Tp,Tc;

    public:
        string dataset_name;
        image_provider(int,char**);
        virtual ~image_provider();
        void init();
        string get_newimage_name();
        int32_t get_num_frames();
        void load_new_image();
        void match_new_image();
        void run_gpu_F(int32_t ransac_iter,float ransac_th);
        void run_serial_F(int32_t ransac_iter,float ransac_th);
        void run_serial_yaw_voting(int32_t ransac_iter,float ransac_th,int32_t res,float th);
        void run_serial_arun_voting(int32_t ransac_iter,float ransac_th,int32_t res,float var,int frame);
        void run_serial_arun_voting2(int32_t ransac_iter,float ransac_th,float depth);
        void run_serial_scar_voting(int32_t ransac_iter,float ransac_th);
        void save_R_gpu();
        void save_R_serial();
        void save_arun_voting();
        void save_R_gpu_zero();
        void save_R_serial_zero();
        void save_arun_voting_zero();

        void save_yaw_voting();
        void save_arun_voting2();
        void save_scar_voting();
        void display_results();
        void finish();
        void set_drift(float);
        bool normalizeFeaturePoints(vector<Matcher::p_match> &p_matched,Matrix &Tp,Matrix &Tc);

    protected:
    private:
    void calcFundamental_inliers (const vector <Point2Df> &src, const vector <Point2Df> &dst, const vector <char> &inlier_mask,int32_t inliers,float* best_F);
    void calcFundamental_inliers (const vector <Point2Df> &src, const vector <Point2Df> &dst, const vector <char> &inlier_mask,int32_t inliers,Matrix &F);
};
  struct bucketing {
    int32_t max_features;  // maximal number of features per bucket
    double  bucket_width;  // width of bucket
    double  bucket_height; // height of bucket
    bucketing () {
      max_features  = 2;
      bucket_width  = 50;
      bucket_height = 50;
    }
  };
#endif // IMAGE_PROVIDER_H
