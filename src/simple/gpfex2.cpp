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
const uint T = 1033;


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
int main()
{
  // Initialize Random number Generator
  gsl_rng *rg;
  long seed = time(NULL)*getpid();
  rg = gsl_rng_alloc(gsl_rng_rand48);
  gsl_rng_set(rg,seed);

  //Re-sample criterion
  float resample_percentage = 0.025;
  uint Nt = ceil(resample_percentage * Ns);


  std::vector<double> x(Nx);    // state
  std::vector<double> z(Nz); 
  std::vector<double> xf(Nx);   // Filtered state
  std::vector<double> wt(Ns);   // weights



  // Create a pointCloud 
  PF::pf  pointCloud(Ns, Nx, Nz, WRSWR);

  ifstream fin("data3.txt");
  ofstream fout("result.txt");
  ofstream fout2("particle.txt");

  uint nonSampItnCnt = 0.0;
  double varx, vary;
  for(uint k = 0; k < T; ++k)
  {
    //cout << "k = " << k << endl;

    fin >> z[0] >> z[1];

    if(k == 0) // initialize PF
    {
      pointCloud.initialize(k, 0, var_x);

      // Actual values 
      x[0] = gsl_ran_gaussian(rg, var_x); 
      x[1] = gsl_ran_gaussian(rg, var_xdot);
      x[2] = gsl_ran_gaussian(rg, var_y); 
      x[3] = gsl_ran_gaussian(rg, var_xdot);

      z[0] = z[0]; 
      z[1] = z[1]; 

    }
    else
    {
      //Actual values
      process(x, x, (void*)rg);             // p(xk | xkm1)

     // cout << "z: ";
      for(uint i = 0; i < Nz; ++i)
      {
        z[i] = z[i] + gsl_ran_gaussian(rg, var_z);
        //cout << z[i] << "\t";
      }
      //cout << endl;

      // Note that for us, only measurement is available
      // Estimate the states using particle Filter
      pointCloud.particleFilterUpdate(process, observation, likelihood, z, 0); // Don't resample  here

      float neff = pointCloud.getEffectivePopulation();
      pointCloud.filterOutput(xf);

      if(ceil(neff) < Nt)
      {
        pointCloud.resample();
        cout << "k = " << k << "\tNeff = " << neff << "\t Resampling ..." << endl;
      }
      else
      {
        nonSampItnCnt++;
        cout << "k = " << k << "\tNeff = " << neff << endl;
      }

    }//else-loop

    fout << z[0] << "\t" << z[1] << "\t" << xf[0] << "\t" << xf[2] << endl;

    std::vector<double> xp(Nx);
    std::vector<double> zp(Nz);
    std::vector<double> wp(Ns);
    for(uint p = 0; p < Ns; ++p)
    {
      wp[p] = pointCloud.getParticleState(xp, zp, p);

      fout2 << xp[0] << "\t" << xp[2] << "\t" << wp[p] << endl;
    }
    fout2 << endl << endl;

    //getchar();

  }// iteration-loop

  cout << "Non sampling Iteration Count = \t" << nonSampItnCnt << endl;

  fin.close();
  fout.close();
  fout2.close();


  gsl_rng_free(rg);


  return 0;
}

