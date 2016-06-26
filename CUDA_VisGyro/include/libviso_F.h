//From Libviso Library, Andreas Geiger: www.cvlibs.net

#ifndef LIBVISO_F_H
#define LIBVISO_F_H
#include"matrix.h"
#include "../matcher.h"
struct parameter
{
    int32_t ransac_iters;
    float inlier_threshold;
    float depth_th;
    int32_t res;
    float th;
    float var;
};
class libviso_F
{
    parameter param;
    int frame;

    public:
        libviso_F(int32_t ,float);
        libviso_F(int32_t ,float,float);
        libviso_F(int32_t ,float,int32_t,float,float,int);
        virtual ~libviso_F();
        void fundamentalMatrix (const std::vector<Matcher::p_match> &p_matched,const std::vector<int32_t> &active,Matrix &F);
        std::vector<int32_t> getInlier (const std::vector<Matcher::p_match> &p_matched,Matrix &F);
        void get_F(const std::vector<Matcher::p_match> &p_matched,Matrix &F);
        float get_yaw_voting(const std::vector<Matcher::p_match> &p_matched,Matrix &K);
        float get_arun_voting(const std::vector<Matcher::p_match> &p_matched,Matrix &K);
        float get_arun_voting2(const std::vector<Matcher::p_match> &p_matched,Matrix &K);
        float get_scar_voting(const std::vector<Matcher::p_match> &p_matched,Matrix &K);
        std::vector<int32_t> getRandomSample(int32_t N,int32_t num);
        std::vector<int32_t> getHistogram(std::vector<float>,int32_t res,float th);
        std::vector<int32_t> getHistogram(std::vector<float>,int32_t res,float th,float var);
        int32_t getMaxHistogram(std::vector<int32_t>);
        float get_best_yaw(std::vector<float>,int32_t res,float th);
        float get_best_yaw(std::vector<float>,int32_t res,float th,float var);
        float get_best_yaw_median(std::vector<float>,int32_t res,float th);
        bool check_depth(const  std::vector<Matcher::p_match> &p_matched, std::vector<int32_t> &active,float threshold);
        void print_histogram(std::vector<int32_t> histogram,int frame,int32_t res, float th,float var);

    protected:
    private:
};

#endif // LIBVISO_F_H
