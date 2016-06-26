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

#include "image_provider.h"
#include"../CUDA_RANSAC_Homography.h"
#include<iostream>
using namespace std;
image_provider::image_provider(int32_t c,char**v)
{
    argv=v;
    argc=c;
    frame=0;
    matcher   = new Matcher();
    //ctor
}

image_provider::~image_provider()
{
    //dtor
}

void image_provider::init()
{
	if(argc < 8)
	{
		cout << "./exe FirstPath Extention NumFrames DisparityThr Detector Descriptor MatchingThr "<<endl;
		exit(0);
	}
    int32_t num_features;
	if(argc == 9)
		num_features = atoi(argv[8]);
	else
		num_features=1000;
    first_path=argv[1];
    extention=argv[2];
    frames_num=atoi(argv[3]);
    int32_t disparity_th=atoi(argv[4]);
    dataset_name=argv[5];
    string extractor_typestr=argv[6];
    float match_th=atof(argv[7]);
    Matrix kk(3,3);
    K=kk;
    ifstream callib_file;
    callib_file.open("callibration.txt",std::fstream::in);
    int32_t m=0,n=0;
    for(int32_t i=0;i<9;i++)
    {
        double element;
        callib_file>>element;
        K.val[m][n]=element;
        n++;
        if(n==3)
        {
            n=0;
            m++;
        }
    }

    frpy_Arun=fopen("rpy_arun","w+");
    frpy_gpu=fopen("rpy_gpu","w+");
    frpy_serial=fopen("rpy_serial","w+");
    frpy_voting=fopen("rpy_voting","w+");
    frpy_arun_voting=fopen("rpy_arun_voting","w+");
    frpy_arun_voting2=fopen("rpy_arun_voting2","w+");
    frpy_scar_voting=fopen("rpy_scar_voting","w+");
}

string image_provider::get_newimage_name()
{
    std::stringstream ss1,ss2;
    ss1 << frame;
    string n1 = ss1.str();
    string temp_firstname=first_path;
    temp_firstname.erase(temp_firstname.length()-n1.length(),n1.length());
    string name1=temp_firstname+n1;
    return name1+extention;

}

int32_t image_provider::get_num_frames()
{
    return frames_num;
}

void image_provider::load_new_image()
{
    newImg = imread(get_newimage_name(), CV_LOAD_IMAGE_GRAYSCALE );
    assert(newImg.data);
    grey=newImg ;
    frame++;
}

void image_provider::match_new_image()
{
    unsigned char *I = grey.data;
    int32_t  dims_[] = {grey.cols,grey.rows,grey.cols};
    if(frame==1)//first image frame=0
    {
        matcher->pushBack(I,dims_,0);
        newImg.release();
        grey.release();
        return;
    }
    matcher->pushBack(I,dims_,0);
    matcher->matchFeatures(0);
    bucketing b;
    matcher->bucketFeatures(b.max_features,b.bucket_width,b.bucket_height);
    p_matched = matcher->getMatches();



    //vector<Matcher::p_match> p_matched_normalized = p_matched;
    //if (!normalizeFeaturePoints(p_matched_normalized,Tp,Tc))
        //cout<<"errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrror"<<endl;
    //p_matched=p_matched_normalized;
    Tp=Matrix::eye(3);
    Tc=Matrix::eye(3);
    src.clear();
    dst.clear();
    match_score.clear();
    for(int32_t i=0; i < p_matched.size(); i++)
    {
        Point2Df pt1, pt2;
        pt1.x = p_matched[i].u1p;
        pt1.y = p_matched[i].v1p;

        pt2.x = p_matched[i].u1c;
        pt2.y = p_matched[i].v1c;

        src.push_back(pt1);
        dst.push_back(pt2);
        match_score.push_back(1);//equal match score-> no effect
    }
    cout<<endl<<"new match: points"<<src.size()<<endl;
}
void image_provider::run_gpu_F(int32_t ransac_iter,float ransac_th)
{
    int32_t best_inliers;
    float best_F[9];
    inlier_mask_gpu.clear();
    CUDA_RANSAC_Fundamental(src, dst, match_score, ransac_th, ransac_iter, &best_inliers, best_F, &inlier_mask_gpu);
    calcFundamental_inliers (src, dst, inlier_mask_gpu, best_inliers, F_gpu);
    F_gpu =  ~Tc*F_gpu*Tp;

    cout<<"inliers gpu"<<best_inliers<<endl;

}

void image_provider::run_serial_F(int32_t ransac_iter,float ransac_th)
{
    ///Calculate F using libviso in serial
    int32_t best_inliers;
    libviso_F libf(ransac_iter,ransac_th);
    libf.get_F(p_matched,F_serial);
    cout<<"F serial:"<<F_serial.val[0][0]<<" "<<F_serial.val[0][1]<<" "<<F_serial.val[0][2]<<endl<<F_serial.val[1][0]<<" "<<F_serial.val[1][1]<<" "<<F_serial.val[1][2]<<endl<<F_serial.val[2][0]<<" "<<F_serial.val[2][1]<<" "<<F_serial.val[2][2]<<endl;
    inliers_serial=libf.getInlier(p_matched,F_serial);
    F_serial =  ~Tc*F_serial*Tp;
    //calcFundamental_inliers (src, dst, inlier_mask_serial, best_inliers, F_serial); to change implementaiton

}

void image_provider::run_serial_yaw_voting(int32_t ransac_iter,float ransac_th,int32_t res,float th)
{
    int32_t best_inliers;
    libviso_F libf(ransac_iter,ransac_th,res,th,0,0);//error
    yaw_voting=libf.get_yaw_voting(p_matched,K);
    return;

}

void image_provider::run_serial_arun_voting(int32_t ransac_iter,float ransac_th,int32_t res,float var,int frame)
{
    int32_t best_inliers;
    libviso_F libf(ransac_iter,ransac_th,res,0,var,frame);
    arun_voting=libf.get_arun_voting(p_matched,K);
    return;

}

void image_provider::run_serial_arun_voting2(int32_t ransac_iter,float ransac_th,float depth)
{
    int32_t best_inliers;
    libviso_F libf(ransac_iter,ransac_th,depth);
    arun_voting2=libf.get_arun_voting2(p_matched,K);
    return;

}
void image_provider::run_serial_scar_voting(int32_t ransac_iter,float ransac_th)
{
    libviso_F libf(ransac_iter,ransac_th);
    scar_voting=libf.get_scar_voting(p_matched,K);
    return;

}
void image_provider::save_R_gpu()
{
    //F = ~Tc*F*Tp;
    libviso_geometry lg;
    Matrix R_gpu;
    lg.get_R(F_gpu,K,p_matched,R_gpu);
    vector<float> rpy_F_gpu=lg.get_rpy(R_gpu);
    fprintf(frpy_gpu,"%f %f %f\n",rpy_F_gpu[1]+drift,rpy_F_gpu[0],rpy_F_gpu[2]);
    cout<<"yaw gpu: "<<rpy_F_gpu[1]<<endl;
    yaw_gpu=rpy_F_gpu[1]+drift;
}

void image_provider::save_R_gpu_zero()
{

    fprintf(frpy_gpu,"%f %f %f\n",0,0,0);
}

void image_provider::save_yaw_voting()
{
    //F = ~Tc*F*Tp;
    //libviso_geometry lg;
    //Matrix R_gpu;
    //lg.get_R(F_gpu,K,p_matched,R_gpu);
    //vector<float> rpy_F_gpu=lg.get_rpy(R_gpu);
    fprintf(frpy_voting,"%f %f %f\n",yaw_voting,0,0);
    cout<<"yaw voting: "<<yaw_voting<<endl;

}

void image_provider::save_arun_voting()
{
    //F = ~Tc*F*Tp;
    //libviso_geometry lg;
    //Matrix R_gpu;
    //lg.get_R(F_gpu,K,p_matched,R_gpu);
    //vector<float> rpy_F_gpu=lg.get_rpy(R_gpu);
    fprintf(frpy_arun_voting,"%f %f %f\n",arun_voting,0,0);
    cout<<"arun voting: "<<arun_voting<<endl;

}

void image_provider::save_arun_voting_zero()
{
    fprintf(frpy_arun_voting,"%f %f %f\n",0,0,0);

}

void image_provider::save_arun_voting2()
{
    //F = ~Tc*F*Tp;
    //libviso_geometry lg;
    //Matrix R_gpu;
    //lg.get_R(F_gpu,K,p_matched,R_gpu);
    //vector<float> rpy_F_gpu=lg.get_rpy(R_gpu);
    fprintf(frpy_arun_voting2,"%f %f %f\n",arun_voting2,0,0);
    cout<<"arun voting2: "<<arun_voting2<<endl;

}


void image_provider::save_scar_voting()
{
    //F = ~Tc*F*Tp;
    //libviso_geometry lg;
    //Matrix R_gpu;
    //lg.get_R(F_gpu,K,p_matched,R_gpu);
    //vector<float> rpy_F_gpu=lg.get_rpy(R_gpu);
    fprintf(frpy_scar_voting,"%f %f %f\n",scar_voting,0,0);
    cout<<"scar voting: "<<scar_voting<<endl;

}

void image_provider::save_R_serial()
{
    libviso_geometry lg;
    Matrix R_serial;
    lg.get_R(F_serial,K,p_matched,R_serial);
    vector<float> rpy_F_serial = lg.get_rpy(R_serial);
    fprintf(frpy_serial,"%f %f %f\n",rpy_F_serial[1],rpy_F_serial[0],rpy_F_serial[2]);
    cout<<"yaw serial: "<<rpy_F_serial[1]<<endl;
}

void image_provider::save_R_serial_zero()
{
    fprintf(frpy_serial,"%f %f %f\n",0,0,0);

}


void image_provider::display_results()
{
    printf("\n");
    printf("Fundamental matrix\n");
    for(int32_t i=0; i < 3; i++)
    {
        for( int32_t j=0;j<3;j++)
            printf("%g ", F_gpu.val[i][j]);
        printf("\n");
    }
    ///Display results
    {
        int32_t h = grey.rows;
        int32_t w = grey.cols;

        Mat result(h, w, CV_8UC3);

        for(int32_t y=0; y < grey.rows; y++) {
            for(int32_t x=0; x < grey.cols; x++) {
                result.at<Vec3b>(y,x)[0] = grey.at<uchar>(y,x);
                result.at<Vec3b>(y,x)[1] = grey.at<uchar>(y,x);
                result.at<Vec3b>(y,x)[2] = grey.at<uchar>(y,x);
            }
        }
        for(int i=0;i<src.size();i++)
            line(result, Point(src[i].x, src[i].y), Point(dst[i].x, dst[i].y), CV_RGB(0,0,255));
        for(int32_t i=0; i < inlier_mask_gpu.size(); i++) {
            if(inlier_mask_gpu[i]) {
                line(result, Point(src[i].x, src[i].y), Point(dst[i].x, dst[i].y), CV_RGB(255,0,0));
            }
        }

        imshow("result",result);
        waitKey(0);
        result.release();
    }
    ///Display results
}
void image_provider::finish()
{
    fclose(frpy_Arun);
    fclose(frpy_gpu);
    fclose(frpy_serial);
    fclose(frpy_voting);
    fclose(frpy_arun_voting);
    fclose(frpy_arun_voting2);
    fclose(frpy_scar_voting);
    grey.release();
    newImg.release();
    p_matched.clear();
    src.clear();
    dst.clear();
}



///////////////////////////////////////////////

///calculate Fundamental for all inliers:
void image_provider::calcFundamental_inliers (const vector <Point2Df> &src, const vector <Point2Df> &dst, const vector <char> &inlier_mask,int32_t inliers,float* best_F) {
  // create constraint32_t matrix A
  Matrix F(3,3);
  Matrix A(inliers,9);
  int32_t matrix_row=0;
  for(int32_t i=0;i<src.size();i++)
  {
    if(inlier_mask[i]==1)
    {
        float srcx=src[i].x;

        float srcy=src[i].y;

        float dstx=dst[i].x;

        float dsty=dst[i].y;

        A.val[matrix_row][0] = dstx*srcx;

        A.val[matrix_row][1] = dstx*srcy;

        A.val[matrix_row][2] = dstx;

        A.val[matrix_row][3] = dsty*srcx;

        A.val[matrix_row][4] = dsty*srcy;

        A.val[matrix_row][5] = dsty;

        A.val[matrix_row][6] = srcx;

        A.val[matrix_row][7] = srcy;

        A.val[matrix_row][8] = 1;

        matrix_row++;
    }

  }
   // compute singular value decomposition of A

  Matrix U,W,V;
  A.svd(U,W,V);

  // extract fundamental matrix from the column of V corresponding to the smallest singular value
  F = Matrix::reshape(V.getMat(0,8,8,8),3,3);

  // enforce rank 2
  F.svd(U,W,V);
  W.val[2][0] = 0;
  F = U*Matrix::diag(W)*~V;

  for(int32_t i=0;i<3;i++)
    for(int32_t j=0;j<3;j++)
        best_F[j*3+i]=F.val[i][j];
  return;
}

void image_provider::calcFundamental_inliers (const vector <Point2Df> &src, const vector <Point2Df> &dst, const vector <char> &inlier_mask,int32_t inliers,Matrix &F) {

  // create constraint32_t matrix A
  Matrix A(inliers,9);
  int32_t matrix_row=0;
  for(int32_t i=0;i<src.size();i++)
  {

    if(inlier_mask[i]==1)
    {

        float srcx=src[i].x;

        float srcy=src[i].y;

        float dstx=dst[i].x;

        float dsty=dst[i].y;

        A.val[matrix_row][0] = dstx*srcx;

        A.val[matrix_row][1] = dstx*srcy;

        A.val[matrix_row][2] = dstx;

        A.val[matrix_row][3] = dsty*srcx;

        A.val[matrix_row][4] = dsty*srcy;

        A.val[matrix_row][5] = dsty;

        A.val[matrix_row][6] = srcx;

        A.val[matrix_row][7] = srcy;

        A.val[matrix_row][8] = 1;

        matrix_row++;
    }

  }
   // compute singular value decomposition of A

  Matrix U,W,V;
  A.svd(U,W,V);

  // extract fundamental matrix from the column of V corresponding to the smallest singular value
  F = Matrix::reshape(V.getMat(0,8,8,8),3,3);

  // enforce rank 2
  F.svd(U,W,V);
  W.val[2][0] = 0;
  F = U*Matrix::diag(W)*~V;
  return;
}


void image_provider::set_drift(float d)
{
    drift=d;
}
bool image_provider::normalizeFeaturePoints(vector<Matcher::p_match> &p_matched,Matrix &Tp,Matrix &Tc) {

  // shift origins to centroids
  double cpu=0,cpv=0,ccu=0,ccv=0;
  for (vector<Matcher::p_match>::iterator it = p_matched.begin(); it!=p_matched.end(); it++) {
    cpu += it->u1p;
    cpv += it->v1p;
    ccu += it->u1c;
    ccv += it->v1c;
  }
  cpu /= (double)p_matched.size();
  cpv /= (double)p_matched.size();
  ccu /= (double)p_matched.size();
  ccv /= (double)p_matched.size();
  for (vector<Matcher::p_match>::iterator it = p_matched.begin(); it!=p_matched.end(); it++) {
    it->u1p -= cpu;
    it->v1p -= cpv;
    it->u1c -= ccu;
    it->v1c -= ccv;
  }

  // scale features such that mean distance from origin is sqrt(2)
  double sp=0,sc=0;
  for (vector<Matcher::p_match>::iterator it = p_matched.begin(); it!=p_matched.end(); it++) {
    sp += sqrt(it->u1p*it->u1p+it->v1p*it->v1p);
    sc += sqrt(it->u1c*it->u1c+it->v1c*it->v1c);
  }
  if (fabs(sp)<1e-10 || fabs(sc)<1e-10)
    return false;
  sp = sqrt(2.0)*(double)p_matched.size()/sp;
  sc = sqrt(2.0)*(double)p_matched.size()/sc;
  for (vector<Matcher::p_match>::iterator it = p_matched.begin(); it!=p_matched.end(); it++) {
    it->u1p *= sp;
    it->v1p *= sp;
    it->u1c *= sc;
    it->v1c *= sc;
  }

  // compute corresponding transformation matrices
  double Tp_data[9] = {sp,0,-sp*cpu,0,sp,-sp*cpv,0,0,1};
  double Tc_data[9] = {sc,0,-sc*ccu,0,sc,-sc*ccv,0,0,1};
  Tp = Matrix(3,3,Tp_data);
  Tc = Matrix(3,3,Tc_data);

  // return true on success
  return true;
}
