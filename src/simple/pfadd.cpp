
#include <iostream>
#include <fstream>
#include "gpf.h"

using namespace std;
using namespace PF;

//#define FILEIO


const double delta = 0.1;
const double var_x = 50.0;
const double var_y = 20.0;
const double var_xdot = 1.0;
const double var_z = 1.0;
const uint Ns = 1000;
const uint Nx = 4;
const uint Nz = 2;


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
void observation(std::vector<double> &zk, const std::vector<double> &xk, void* data)
{
 gsl_rng *r = (gsl_rng*) data;

  // (x,y) position
  zk[0] = xk[0] +  gsl_ran_gaussian(r, sqrt(var_x));
  zk[1] = xk[2] +  gsl_ran_gaussian(r, sqrt(var_y));

}
//-----------------------------------------------------
// Likelihood is a t-distribution with nu = 10
double likelihood(const std::vector<double> &z, const std::vector<double> &zhat, void* data)
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
//-----------------------------------
