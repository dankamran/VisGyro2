#ifndef LIBVISO_GEOMETRY_H
#define LIBVISO_GEOMETRY_H
#include"matrix.h"
#include<vector>
#include "../matcher.h"
class libviso_geometry
{
    public:
        libviso_geometry();
        virtual ~libviso_geometry();
        void get_R(Matrix &F,Matrix &K,const std::vector<Matcher::p_match> &p_matched,Matrix &R );
        void EtoRt(Matrix &E,Matrix &K,const std::vector<Matcher::p_match> &p_matched,Matrix &X,Matrix &R,Matrix &t);
        int32_t triangulateChieral (const std::vector<Matcher::p_match> &p_matched,Matrix &K,Matrix &R,Matrix &t,Matrix &X);
        std::vector <float>get_rpy(Matrix R);

    protected:
    private:
};

#endif // LIBVISO_GEOMETRY_H
