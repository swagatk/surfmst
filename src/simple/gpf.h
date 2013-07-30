/** Generic Particle Filter
 * Start Date: May 10, 2013, Friday.
 * Last Update: May 10, 2013, Friday.
 * Author: Swagat Kumar (swagat.kumar@gmail.com)
 * ------------------------------ */

#ifndef _PF_H
#define _PF_H

#include <vector>
#include <algorithm>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <cmath>
#include <unistd.h>
#include <sys/types.h>

namespace PF
{
  typedef unsigned int uint;

//sampling strategies
enum sampling_strategy{
    WRSWR,
    MULTINOM,
    SYSTEMATIC,
};

  class pf
  {
    uint itn_num;                    // iteration number
    uint Ns;                         // No. of particles
    uint Nx;                         // size of the state vector
    uint Nz;                         // size of measurement vector

    enum  sampling_strategy ss;     

    std::vector<double> w;                       // Weights
    std::vector< std::vector<double> > xk;       // States at instant k
    std::vector<double> xfk;                     // particle filter output
    std::vector<std::vector<double> > zk;        // Measurement vector (not needed now)
    double Neff;
    std::vector<uint> zbin; // Stores binary data

    //GSL random number generator variable
    gsl_rng *r;

    public:
    pf();                                    // Default constructor
    pf(const pf &o);                         // Copy Constructor
    pf(uint ns, uint nx, uint nz, sampling_strategy s);           // Constructor
    ~pf();                                   // Destructor

    void assignIterationNumber(uint k);
    void initialize(uint k, double mean, double proc_noise_sd);    // Initialize the particle filter
    double sample_odometry_motion_model();  
    void initialize(uint k, const std::vector<double> &w,
                    const std::vector<std::vector<double> > &x); //overloaded initialize function
    void initialize(uint k, const std::vector<double> &w,
                    const std::vector<std::vector<double> > &x,
                    const std::vector<uint> &zb);

    void particleFilterUpdate(
        void (*pmodel)(std::vector<double> &x, const std::vector<double> &xprev, void* data), // Process Model 
        void (*omodel)(std::vector<double> &z, const std::vector<double> &x, void* data), // Observation Model 
        double(*likelihood)(const std::vector<double> &z, const std::vector<double> &zhat, void *data),
         const std::vector<double> &z,   //Observation Model
         uint resample_size = 0);  //default option: no resample

    void pfupdate1(
            void (*pmodel)(std::vector<double> &x, const std::vector<double> &xprev, void* data), // Process Model
            uint (*omodel)(double p, gsl_rng *r), // Observation Model
            double(*likelihood)(uint z, uint zhat, gsl_rng *r), //likelihood function
            const std::vector<uint> &z,   //Observation Model
            uint resample_size = 0);  //default option: no resample

    void particleFilterUpdate2(
            void (*pmodel)(std::vector<double> &x, const std::vector<double> &xprev, void* data), // Process Model
            void (*omodel)(std::vector<double> &z, const std::vector<double> &x, gsl_rng *r), // Observation Model
            double(*likelihood)(const std::vector<double> &z, const std::vector<double> &zhat, gsl_rng *r), //likelihood function
            const std::vector<std::vector<double> >&zd,   //Observation data
            uint resample_size = 0);  //default option: no resample




    void filterOutput(std::vector<double> &xf);       // Output of Particle Filter

    //sampling routines
    void resample();
    void wrswr();
    void multinomial_sampling();
    void systematic_resampling();
    void systematic_resampling2();
    void print_states();
    void display_array(double **xv, uint row, uint col);
    double getParticleState(std::vector<double> &x, std::vector<double> &z, uint pfidx);
    void setParticleState(uint pfidx, std::vector<double> &x, double wt=0);
    void setParticleState(uint pfidx, std::vector<double> &x, double wt, uint zbin);
    double getEffectivePopulation();
    uint getItnNum();
  };
}

#endif
