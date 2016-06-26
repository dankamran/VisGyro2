

#include <sys/time.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "CUDA_RANSAC_Homography.h"
//#include "CUDA_RANSAC_Fundamental.h"
#include "CUDA_BruteForceMatching.h"

#include "bearingVector.h"
#include "matcher.h"
#include <fstream>
#include "matrix.h"
#include "libviso_geometry.h"
#include "libviso_F.h"
#include "image_provider.h"


// Calc the theoretical number of iterations using some conservative parameters
const double CONFIDENCE = 0.99;
const double INLIER_RATIO = 0.40; // Assuming lots of noise in the data!
const double INLIER_THRESHOLD = 3.0; // pixel distance

///Parameters from Libviso:
int   libviso_ransac_iters     = 4000;
float libviso_inlier_threshold = 0.1;//0.00001;



double TimeDiff(timeval t1, timeval t2)
{
    double t;
    t = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    t += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms

    return t;
}



int main(int argc, char *argv[])
{
    ifstream calib;
    calib.open("calib.txt");
    float gpu_ransac_th,serial_ransac_th,voting_ransac_th,arun_var,arun_th,drift;
    int32_t gpu_ransac_iter,serial_ransac_iter,voting_ransac_iter,arun_res;
    calib>>gpu_ransac_iter;
    calib>>gpu_ransac_th;
    calib>>serial_ransac_iter;
    calib>>serial_ransac_th;
    calib>>voting_ransac_iter;
    calib>>voting_ransac_th;
    calib>>arun_res;
    calib>>arun_var;
    calib>>drift;

    calib.close();

    image_provider imp(argc,argv);
    imp.set_drift(drift);
    timeval start_time, t1, t2;

    imp.init();

    double gpu_t=0, serial_t=0, match_t=0,arun_t=0;
    for(int frame=0;frame<imp.get_num_frames();frame++)
    {
        std::cout<<"Reading frame: "<<frame<<std::endl;

        gettimeofday(&t1, NULL);
        imp.load_new_image();
        imp.match_new_image();
        gettimeofday(&t2, NULL);
        match_t += TimeDiff(t1,t2);

        printf("Match images: %g ms\n", TimeDiff(t1,t2));

        if (frame==0) continue;
        gettimeofday(&t1, NULL);
        imp.run_gpu_F(gpu_ransac_iter,gpu_ransac_th);
        gettimeofday(&t2, NULL);
        printf("GPU F: %g ms\n", TimeDiff(t1,t2));
        gpu_t += TimeDiff(t1,t2);

        gettimeofday(&t1, NULL);
        //imp.run_serial_F(serial_ransac_iter,serial_ransac_th);
        gettimeofday(&t2, NULL);
        printf("Serial F: %g ms\n", TimeDiff(t1,t2));
        serial_t += TimeDiff(t1,t2);

        gettimeofday(&t1, NULL);
        imp.save_R_gpu();
        gettimeofday(&t2, NULL);
        printf("R GPU: %g ms\n", TimeDiff(t1,t2));

        gettimeofday(&t1, NULL);
        //imp.save_R_serial();
        gettimeofday(&t2, NULL);
        printf("R serial: %g ms\n", TimeDiff(t1,t2));
        ////////////////////////////////////
        //imp.run_serial_yaw_voting(voting_ransac_iter,voting_ransac_th,arun_res,arun_th);
        //imp.save_yaw_voting();
        gettimeofday(&t1, NULL);
        //imp.run_serial_arun_voting(voting_ransac_iter,voting_ransac_th,arun_res,arun_var,frame);
        gettimeofday(&t2, NULL);
        arun_t += TimeDiff(t1,t2);

        imp.save_arun_voting();




        ///bypass
        //frame++;
       // imp.save_R_gpu_zero();
       // imp.save_R_serial_zero();
      //  imp.save_arun_voting_zero();
      //  imp.load_new_image();

        //imp.display_results();

       // imp.run_serial_scar_voting(voting_ransac_iter,voting_ransac_th);
       // imp.save_scar_voting();

        //imp.run_serial_arun_voting2(voting_ransac_iter,voting_ransac_th,voting_depth);
        //imp.save_arun_voting2();

    }
    printf("Match AVG:%g\n",match_t/(imp.get_num_frames()-1));
    printf("GPU AVG:%g\n",gpu_t/(imp.get_num_frames()-1));
    printf("serial AVG:%g\n",serial_t/(imp.get_num_frames()-1));
    printf("Arun AVG:%g\n",arun_t/(imp.get_num_frames()-1));

    ofstream outfile;
    std::stringstream ss1;
    ss1 <<"time_report/"<< imp.dataset_name<<"_g"<<gpu_ransac_iter<<"_s"<<serial_ransac_iter<<"_a"<<voting_ransac_iter;
    string n1 = ss1.str();
    outfile.open(n1.c_str());
    outfile<<match_t/(imp.get_num_frames()-1)<<endl;
    outfile<<gpu_t/(imp.get_num_frames()-1)<<endl;
    outfile<<serial_t/(imp.get_num_frames()-1)<<endl;
    outfile<<arun_t/(imp.get_num_frames()-1)<<endl;
    outfile<<"match"<<endl<<"gpu"<<endl<<"serial"<<endl<<"arun"<<endl;

    return 0;
}
