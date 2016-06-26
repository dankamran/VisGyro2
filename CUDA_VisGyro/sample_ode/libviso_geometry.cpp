#include "libviso_geometry.h"
using namespace std;
libviso_geometry::libviso_geometry()
{
    //ctor
}

libviso_geometry::~libviso_geometry()
{
    //dtor
}
void libviso_geometry::get_R(Matrix &F,Matrix&K,const std::vector<Matcher::p_match> &p_matched,Matrix &R )
{
        Matrix E;
        E = ~K*F*K;

        // re-enforce rank 2 constraint on essential matrix
        Matrix U,W,V;
        E.svd(U,W,V);
        W.val[2][0] = 0;
        E = U*Matrix::diag(W)*~V;

        // compute 3d points X and R|t up to scale
        Matrix X,t;
        EtoRt(E,K,p_matched,X,R,t);
}
int32_t libviso_geometry::triangulateChieral (const vector<Matcher::p_match> &p_matched,Matrix &K,Matrix &R,Matrix &t,Matrix &X) {

  // init 3d point matrix
  X = Matrix(4,p_matched.size());

  // projection matrices
  Matrix P1(3,4);
  Matrix P2(3,4);
  P1.setMat(K,0,0);
  P2.setMat(R,0,0);
  P2.setMat(t,0,3);
  P2 = K*P2;

  // triangulation via orthogonal regression
  Matrix J(4,4);
  Matrix U,S,V;
  for (int32_t i=0; i<(int)p_matched.size(); i++) {
    for (int32_t j=0; j<4; j++) {
      J.val[0][j] = P1.val[2][j]*p_matched[i].u1p - P1.val[0][j];
      J.val[1][j] = P1.val[2][j]*p_matched[i].v1p - P1.val[1][j];
      J.val[2][j] = P2.val[2][j]*p_matched[i].u1c - P2.val[0][j];
      J.val[3][j] = P2.val[2][j]*p_matched[i].v1c - P2.val[1][j];
    }
    J.svd(U,S,V);
    X.setMat(V.getMat(0,3,3,3),0,i);
  }

  // compute inliers
  Matrix  AX1 = P1*X;
  Matrix  BX1 = P2*X;
  int32_t num = 0;
  for (int32_t i=0; i<X.n; i++)
    if (AX1.val[2][i]*X.val[3][i]>0 && BX1.val[2][i]*X.val[3][i]>0)
      num++;

  // return number of inliers
  return num;
}
void libviso_geometry::EtoRt(Matrix &E,Matrix &K,const vector<Matcher::p_match> &p_matched,Matrix &X,Matrix &R,Matrix &t) {

  // hartley matrices
  double W_data[9] = {0,-1,0,+1,0,0,0,0,1};
  double Z_data[9] = {0,+1,0,-1,0,0,0,0,0};
  Matrix W(3,3,W_data);
  Matrix Z(3,3,Z_data);

  // extract T,R1,R2 (8 solutions)
  Matrix U,S,V;
  E.svd(U,S,V);
  Matrix T  = U*Z*~U;
  Matrix Ra = U*W*(~V);
  Matrix Rb = U*(~W)*(~V);

  // convert T to t
  t = Matrix(3,1);
  t.val[0][0] = T.val[2][1];
  t.val[1][0] = T.val[0][2];
  t.val[2][0] = T.val[1][0];

  // assure determinant to be positive
  if (Ra.det()<0) Ra = -Ra;
  if (Rb.det()<0) Rb = -Rb;

  // create vector containing all 4 solutions
  vector<Matrix> R_vec;
  vector<Matrix> t_vec;
  R_vec.push_back(Ra); t_vec.push_back( t);
  R_vec.push_back(Ra); t_vec.push_back(-t);
  R_vec.push_back(Rb); t_vec.push_back( t);
  R_vec.push_back(Rb); t_vec.push_back(-t);

  // try all 4 solutions
  Matrix X_curr;
  int32_t max_inliers = 0;
  for (int32_t i=0; i<4; i++) {
    int32_t num_inliers = triangulateChieral(p_matched,K,R_vec[i],t_vec[i],X_curr);
    if (num_inliers>max_inliers) {
      max_inliers = num_inliers;
      X = X_curr;
      R = R_vec[i];
      t = t_vec[i];
    }
  }
}

std::vector <float>libviso_geometry::get_rpy(Matrix R)
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
