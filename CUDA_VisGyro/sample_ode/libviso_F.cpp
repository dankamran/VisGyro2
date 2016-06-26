#include "libviso_F.h"
#include "libviso_geometry.h"
#include "bearingVector.h"
#include <fstream>
#include <sstream>
#include <cmath>
using namespace std;
libviso_F::libviso_F(int32_t iter,float th)
{
    //ctor
    param.ransac_iters=iter;
    param.inlier_threshold=th;
    frame=0;
}

libviso_F::libviso_F(int32_t iter,float th,float depth)
{
    //ctor
    param.ransac_iters=iter;
    param.inlier_threshold=th;
    param.depth_th=depth;
    frame=0;
}

libviso_F::libviso_F(int32_t iter,float th,int32_t res,float th2,float var,int fff)
{
    //ctor
    param.ransac_iters=iter;
    param.inlier_threshold=th;
    param.res=res;
    param.th=th2;
    param.var=var;
    frame=fff;
}

libviso_F::~libviso_F()
{
    //dtor
}


void libviso_F::get_F(const vector<Matcher::p_match> &p_matched,Matrix &F)
{
    vector<int32_t> inliers;
    int32_t N = p_matched.size();
    for (int32_t k=0;k<param.ransac_iters;k++) {

        // draw random sample set
        vector<int32_t> active = getRandomSample(N,8);

        // estimate fundamental matrix and get inliers
        fundamentalMatrix(p_matched,active,F);
        vector<int32_t> inliers_curr = getInlier(p_matched,F);

        // update model if we are better
        if (inliers_curr.size()>inliers.size())
          inliers = inliers_curr;
    }
    fundamentalMatrix(p_matched,inliers,F);
    cout<<"serial inliers:"<<inliers.size()<<"ransac th: "<<param.inlier_threshold<<"iter"<<param.ransac_iters<<endl;



    return;
}

float libviso_F::get_yaw_voting(const vector<Matcher::p_match> &p_matched,Matrix &K)
{
    Matrix F;
    vector<int32_t> inliers;
    int32_t N = p_matched.size();
    vector<float> yaws;
    for (int32_t k=0;k<param.ransac_iters;k++)
    {
        vector<int32_t> active = getRandomSample(N,8);
        fundamentalMatrix(p_matched,active,F);
        ///vector<int32_t> inliers_curr = getInlier(p_matched,F);
        libviso_geometry lg;
        Matrix R;
        lg.get_R(F,K,p_matched,R);
        vector<float> rpy=lg.get_rpy(R);
        yaws.push_back(rpy[1]);
        //cout<<"new yaw: "<<rpy[1]<<endl;
    }

    float yaw = get_best_yaw(yaws,param.res,param.th);
    cout<<"voting yaw:"<<yaw<<endl;
    return yaw;
}

float libviso_F::get_arun_voting(const vector<Matcher::p_match> &p_matched,Matrix &K)
{
    vector<int32_t> inliers;
    int32_t N = p_matched.size();
    vector<float> yaws;
    Matrix KInv=K.inv(K);
    for (int32_t k=0;k<param.ransac_iters;k++)
    {
        vector<int32_t> active = getRandomSample(N,3);
        bearingVector bv(p_matched,active,KInv);
        bv.computeCenter_c();
        bv.computeCenter_p();
        bv.minuseCenter_c();
        bv.minuseCenter_p();
        bv.computeH();
        Matrix R=bv.computeR();
        if(R.det()<0) continue;
        ///fundamentalMatrix(p_matched,active,F);
        ///vector<int32_t> inliers_curr = getInlier(p_matched,F);
        libviso_geometry lg;
        vector<float> rpy=lg.get_rpy(R);
        if(rpy[1]>param.th+param.var)continue;
        if(rpy[1]<param.th-param.var)continue;

        yaws.push_back(rpy[1]);
        //cout<<"new yaw: "<<rpy[1]<<endl;
    }
   // float yaw=get_best_yaw_median(yaws,param.res,param.th);
   float yaw=get_best_yaw(yaws,param.res,param.th,param.var);
    cout<<"arun voting yaw:"<<yaw<<endl;
    return yaw;
}




float libviso_F::get_arun_voting2(const vector<Matcher::p_match> &p_matched,Matrix &K)
{
    vector<int32_t> inliers;
    int32_t N = p_matched.size();
    vector<float> yaws;
    Matrix KInv=K.inv(K);
    for (int32_t k=0;k<param.ransac_iters;k++)
    {
        vector<int32_t> active = getRandomSample(N,3);
        if(check_depth(p_matched,active,param.depth_th)==false)
            continue;
        bearingVector bv(p_matched,active,KInv);
        bv.computeCenter_c();
        bv.computeCenter_p();
        bv.minuseCenter_c();
        bv.minuseCenter_p();
        bv.computeH();
        Matrix R=bv.computeR();
        ///fundamentalMatrix(p_matched,active,F);
        ///vector<int32_t> inliers_curr = getInlier(p_matched,F);
        libviso_geometry lg;
        vector<float> rpy=lg.get_rpy(R);
        yaws.push_back(rpy[1]);
        //cout<<"new yaw: "<<rpy[1]<<endl;
    }
    vector<int32_t> histogram=getHistogram(yaws,10000,0.15);
    int32_t max_idx = getMaxHistogram(histogram);
    max_idx=max_idx-300;
    float yaw = (float)max_idx/100;
    cout<<"arun voting yaw:"<<yaw<<endl;
    return yaw;
}

float libviso_F::get_scar_voting(const std::vector<Matcher::p_match> &p_matched,Matrix &K)
{
    vector<int32_t> inliers;
    int32_t N = p_matched.size();
    vector<float> yaws;
    Matrix KInv=K.inv(K);
    for (int32_t k=0;k<p_matched.size();k++)
    {
        vector<int32_t> active;
        active.push_back(k);
        bearingVector bv(p_matched,active,KInv);
        double yaw=bv.get_yaw_scar();
        yaws.push_back(yaw);
        cout<<"new yaw: "<<yaw<<endl;
    }
    float yaw=get_best_yaw_median(yaws,param.res,param.th);
    return yaw;

}

float libviso_F::get_best_yaw(std::vector<float> yaws,int32_t res,float th)
{
    vector<int32_t> histogram=getHistogram(yaws,res,th);
    int32_t max_idx = getMaxHistogram(histogram);
    max_idx=max_idx- (th)*res;
    float yaw = (float)max_idx/res;
    //cout<<"scar voting yaw:"<<yaw<<endl;
    return yaw;

}

float libviso_F::get_best_yaw(std::vector<float> yaws,int32_t res,float th,float var)
{
    vector<int32_t> histogram=getHistogram(yaws,res,th,var);
    print_histogram(histogram,frame,res,th,var);
    int32_t max_idx = getMaxHistogram(histogram);
    max_idx=max_idx+ (th-var)*res;
    float yaw = (float)max_idx/res;
    //cout<<"scar voting yaw:"<<yaw<<endl;
    return yaw;

}
void libviso_F::print_histogram(std::vector<int32_t> histogram,int frame,int32_t res, float th,float var)
{
    ofstream outfile;
    std::stringstream ss1;
    ss1 <<"hist/"<< frame;
    string n1 = ss1.str();
    outfile.open(n1.c_str());
    for(int i=0;i<histogram.size();i++)
    {
        int idx=i;
        idx=idx+ (th-var)*res;
        float yaw = (float)idx/res;
        outfile<<yaw<<" "<<histogram[i]<<endl;

    }
    outfile.close();
}


float libviso_F::get_best_yaw_median(std::vector<float> yaws,int32_t res,float th)
{
    std::sort(yaws.begin(),  yaws.end(), std::greater<float>());
    float median;
    if ( (yaws.size() - 1 )  % 2 == 0)
    {
        median = yaws[(yaws.size() - 1 )/2] ;
    }
    else
    {
        median = (yaws[yaws.size() / 2] + yaws[yaws.size() / 2 -1])/2 ;
    }


    return median;

}

bool libviso_F::check_depth(const vector<Matcher::p_match> &p_matched,vector<int32_t> &active,float threshold)
{
    Matcher::p_match m1=p_matched[active[0]];
    Matcher::p_match m2=p_matched[active[1]];
    Matcher::p_match m3=p_matched[active[2]];
    double a=sqrt(pow(m1.u1c-m2.u1c,2) + pow(m1.v1c-m2.v1c,2));
    double a2=sqrt(pow(m1.u1p-m2.u1p,2) + pow(m1.v1p-m2.v1p,2));

    double b=sqrt(pow(m1.u1c-m3.u1c,2) + pow(m1.v1c-m3.v1c,2));
    double b2=sqrt(pow(m1.u1p-m3.u1p,2) + pow(m1.v1p-m3.v1p,2));

    double c=sqrt(pow(m3.u1c-m2.u1c,2) + pow(m3.v1c-m2.v1c,2));
    double c2=sqrt(pow(m3.u1p-m2.u1p,2) + pow(m3.v1p-m2.v1p,2));

    double d1=std::abs(a*b2 - a2*b);

    double d2=std::abs(a*c2 - a2*c);

    double d3=std::abs(c*b2 - c2*b);

    if(d1>threshold)
        return false;
    if(d2>threshold)
        return false;
    if(d3>threshold)
        return false;
    return true;
}
std::vector<int32_t> libviso_F::getHistogram(std::vector<float> data,int32_t res,float th)
{
    ///max: 0.15 rad or 8.5 deg min: -0.15 resulotion: *10000
    vector<int32_t> histogram;
    histogram.resize(res*th*2+1,0);
    for(int32_t i=0;i<data.size();i++)
    {
        float di = data[i];
        if(di>th)
            continue;
            //di=0.15;
        if(di<-1*th) continue;
            //di=-0.15;
        di=di*res;
        int32_t di_idx = (int32_t)di;
        di_idx = di_idx+th*res;
        histogram[di_idx]++;
    }
    return histogram;
}

std::vector<int32_t> libviso_F::getHistogram(std::vector<float> data,int32_t res,float th,float var)
{
    ///max: 0.15 rad or 8.5 deg min: -0.15 resulotion: *10000
    vector<int32_t> histogram;
    histogram.resize(res*var*2+1,0);
    for(int32_t i=0;i<data.size();i++)
    {
        float di = data[i];
        if(di>var+th)
            continue;
            //di=0.15;
        if(di<th-var) continue;
            //di=-0.15;
        di=di*res;
        int32_t di_idx = (int32_t)di;
        di_idx = di_idx - (th-var)*res;
        histogram[di_idx]++;
    }
    return histogram;
}

int32_t libviso_F::getMaxHistogram(std::vector<int32_t> histogram)
{
    int32_t max_number=-100;
    int32_t max_idx = 0;
    for(int32_t i=0;i<histogram.size();i++)
    {
        if(histogram[i]>max_number)
        {
            max_number=histogram[i];
            max_idx=i;
        }
    }
    return max_idx;
}

void libviso_F::fundamentalMatrix(const vector<Matcher::p_match> &p_matched,const vector<int32_t> &active,Matrix &F) {

  // number of active p_matched
  int32_t N = active.size();

  // create constraint32_t matrix A
  Matrix A(N,9);
  for (int32_t i=0; i<N; i++) {
    Matcher::p_match m = p_matched[active[i]];
    A.val[i][0] = m.u1c*m.u1p;
    A.val[i][1] = m.u1c*m.v1p;
    A.val[i][2] = m.u1c;
    A.val[i][3] = m.v1c*m.u1p;
    A.val[i][4] = m.v1c*m.v1p;
    A.val[i][5] = m.v1c;
    A.val[i][6] = m.u1p;
    A.val[i][7] = m.v1p;
    A.val[i][8] = 1;
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
}


vector<int32_t> libviso_F::getInlier (const vector<Matcher::p_match> &p_matched,Matrix &F) {

  // extract fundamental matrix
  double f00 = F.val[0][0]; double f01 = F.val[0][1]; double f02 = F.val[0][2];
  double f10 = F.val[1][0]; double f11 = F.val[1][1]; double f12 = F.val[1][2];
  double f20 = F.val[2][0]; double f21 = F.val[2][1]; double f22 = F.val[2][2];

  // loop variables
  double u1,v1,u2,v2;
  double x2tFx1;
  double Fx1u,Fx1v,Fx1w;
  double Ftx2u,Ftx2v;

  // vector with inliers
  vector<int32_t> inliers;

  // for all matches do
  for (int32_t i=0; i<(int32_t)p_matched.size(); i++) {

    // extract matches
    u1 = p_matched[i].u1p;
    v1 = p_matched[i].v1p;
    u2 = p_matched[i].u1c;
    v2 = p_matched[i].v1c;

    // F*x1
    Fx1u = f00*u1+f01*v1+f02;
    Fx1v = f10*u1+f11*v1+f12;
    Fx1w = f20*u1+f21*v1+f22;

    // F'*x2
    Ftx2u = f00*u2+f10*v2+f20;
    Ftx2v = f01*u2+f11*v2+f21;

    // x2'*F*x1
    x2tFx1 = u2*Fx1u+v2*Fx1v+Fx1w;

    // sampson distance
    double d = x2tFx1*x2tFx1 / (Fx1u*Fx1u+Fx1v*Fx1v+Ftx2u*Ftx2u+Ftx2v*Ftx2v);

    // check threshold
    if (fabs(d)<param.inlier_threshold)
      inliers.push_back(i);
  }

  // return set of all inliers
  return inliers;
}

vector<int32_t> libviso_F::getRandomSample(int32_t N,int32_t num) {

  // init sample and totalset
  vector<int32_t> sample;
  vector<int32_t> totalset;

  // create vector containing all indices
  for (int32_t i=0; i<N; i++)
    totalset.push_back(i);

  // add num indices to current sample
  sample.clear();
  for (int32_t i=0; i<num; i++) {
    int32_t j = rand()%totalset.size();
    sample.push_back(totalset[j]);
    totalset.erase(totalset.begin()+j);
  }

  // return sample
  return sample;
}
