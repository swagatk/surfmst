#include <iostream>
#include <fstream>
#include "gpf.h"

using namespace std;
using namespace PF;

//#define FILEIO

//#define ZBIN

#define PF_OPT 1   // 0 - Standard PF where the measurement vector z = center of window with default initialization
                   // 1 - Initial particle states and weights are provided.


#define ARLOC       // Learn an AR model for the window location
//#define ARHT        // Learn an AR model for window height (Not implemented yet) - 12/06/13


const double delta = 0.1;
const double var_x = 50.0;
const double var_y = 20.0;
const double var_xdot = 1.0;
const double var_z = 1.0;
const uint Ns = 1000;
const uint Nx = 4;
const uint Nz = 2;

//Parameters related to the Particle Filter
const double resample_percentage = 0.5;

//Paramters related to the AR predictor
const uint arlen = 5;   // length of AR predictor (time-steps)
const uint arDim = 3; // Dimension of state vector for AR predictor


//----------------------------
// Process Equation
// xn : process noise
void process(std::vector<double> &xk, const std::vector<double> &xkm1, void* data)
{
  gsl_rng *r = (gsl_rng*)data;

  xk[0] = xkm1[0] + delta * xkm1[1] + gsl_ran_gaussian(r, sqrt(var_x));
  xk[1] = xkm1[1] +  gsl_ran_gaussian(r, sqrt(var_xdot));
  xk[2] = xkm1[2] + delta * xkm1[3] + gsl_ran_gaussian(r, sqrt(var_y));
  xk[3] = xkm1[3] + gsl_ran_gaussian(r, sqrt(var_xdot));
}
//-------------------------
// Observation Equation
// vn: measurement noise
void observation(std::vector<double> &zk, const std::vector<double> &xk, gsl_rng *r)
{
  // (x,y) position
  zk[0] = xk[0] +  gsl_ran_gaussian(r, sqrt(var_x));
  zk[1] = xk[2] +  gsl_ran_gaussian(r, sqrt(var_y));

}
//-----------------------------------------------------
// Likelihood is a t-distribution with nu = 10
double likelihood(const std::vector<double> &z, const std::vector<double> &zhat, gsl_rng *r)
{

  double prod = 1.0, e;
  for(uint i = 0; i < z.size(); ++i)
  {
    e = z[i] - zhat[i];
   //prod = prod* gsl_ran_gaussian_pdf(e, var_z);
   prod = prod * 1.5 * pow((1+e*e/10), -5.5);
  }

  return prod;
}
//----------------------------------------
double likelihood3(const std::vector<double> &z, const std::vector<double> &zhat, gsl_rng *r)
{

  double prod = 1.0, e;
  for(uint i = 0; i < z.size(); ++i)
  {
    e = z[i] - zhat[i];
   //prod = prod* gsl_ran_gaussian_pdf(e, var_z);
   prod = prod * 1.5 * pow((1+e*e), -1.0);
  }

  return prod;
}
//-----------------------------------
uint observation2(double p, gsl_rng *r)
{
    uint zk = (uint) gsl_ran_bernoulli(r, p);
    return zk;
}
//----------------------------------------------
double likelihood2(uint z, uint zhat, gsl_rng *r)
{
    uint e = abs(z-zhat);
    double p = gsl_ran_bernoulli_pdf(e, 0.1); // return high probability if z = zhat
    return(p);
}
//---------------------------------------------
// AR Predictor for window location
// xhat = \sum { W_i * x_i }, i = 1, 2, ... N


void arPredictor4Location(std::vector<double> &xhat, const std::vector<std::vector<double> >&x,
                          const std::vector<std::vector<double> >&w)
{
    if( x.size() != w.size() )
    {
        cerr << "size mismatch: w and x must have same size" << endl;
        cerr << "Error in Line " << __LINE__ << "in File " << __FILE__ << endl;
        exit(-1);
    }
    else
    {
        if(x[0].size() != xhat.size())
        {
            cout << "xhat must have same size as that of x[i] and w[i]" << endl;
            cout << "Error in line number " << __LINE__ << "in File " << __FILE__ << endl;
            exit(-1);
        }
        else
        {
            for(uint j = 0; j < xhat.size(); ++j)
            {
                xhat[j] = 0.0;
                for(uint i = 0; i < x.size(); ++i) // size of AR predictor AR(p)
                {
                    xhat[j] += w[i][j] * x[i][j];
                }
            }
        }
    }
}

//-------------------------------------------
// Update the AR model parameters using Gradient Descent Algorithm
// Delta_w = -eta * dE / dw
// NOte that w and x have same size
// eta is the learning rate
//------------------------------------------------
void arPredictorUpdateGD(std::vector<std::vector<double> > &w, const std::vector<double> &e,
                       const std::vector<std::vector<double> > &x, double eta)
{
    for(uint i = 0; i < w.size(); ++i)
    {
        for(uint j = 0; j < w[i].size(); ++j)
        {
            w[i][j] += eta * e[j] * x[i][j];
        }
    }
}
//----------------------------------------------------------------
