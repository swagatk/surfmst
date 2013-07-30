#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include<sys/time.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>

using namespace cv;
using namespace std;


// Parameters/Thresholds affecting performance and must be tuned

#define RANSAC_THREHOLD              1      // ransac threshold for homography ranges from 1-10
#define MAX_MS_ITER                  5     // Maximum iterations allowed if MS does not converge
#define SCALE_RANGE                  30     // Max percentage change that may occur wrt original image
#define MAX_INLIER_DIST              10     // Distance bw matching points in source and target after Homography
#define REPROJ_ALLOWED               5      // Maximum number of points to be reprojected per keypoint in source image
#define TOP_X                        100    // Percentage of points to reproject from target
#define NOC                          5      // Number of K-Means Clusters

// Optional Parameters

#define APPLY_SCALING                0


// If sacling has to be applied
#define PDF_OF_WHOLE                 0      // whether to make PDF out of whole set of descriptors or only the matching ones
#define SHOW_FINAL_ROI               0      // Set it ZERO to see iterating target windows also, 1 to see only Final ROI
#define GT_SHOW                      0      // To show the Ground Truth also
#define IMPROVING_PDF                1      // Method of Reprojection implemented
#define EXPANDED_BOX                 0     // Take a epxanded region around the new target to avoid loss of image due to shift and scaling
#define CHANGE_KERNEL                1      // Change kernel to Gaussian than Epanechnikov (for some functions)

//#define DEBUG                        0
void categorizePoints(vector<Point2f>& points1, vector<Point2f>& points2,vector<Point>& polyPoints1, vector<KeyPoint> &keypoints,
                      Mat &descriptors, vector<KeyPoint> &keypoints1, Mat &descriptors1);
void bcategorizePoints(vector<Point2f>& points1, vector<Point2f>& points2, vector<Point>& polyPoints1, vector<KeyPoint>& keypoints, Mat& descriptors,
                      vector<KeyPoint>& keypoints1, Mat& descriptors1);
void categorizePoints(vector<Point2f>& points1, vector<Point>& polyPoints1, vector<KeyPoint>& keypoints, Mat& descriptors,
                vector<KeyPoint>& tpkeypoints, Mat& tpdescriptors);



void draw_box(Mat img1, CvRect rect );
void colorTheClusters(Mat& img, Rect roi, vector<Point2f>& kp, Mat& labels);

void Epanechnikov( vector<Point2f>& points, vector<float>& pWeight1, CvRect roi );
float EpanechKernel(float x);
float EpanechKernel2(float x);

void my_mouse_callback( int event, int x, int y, int flags, void* param );

void weightScalingAspect ( vector<Point2f>& matchPoints1, vector<Point2f>& matchPoints2, vector<float>& pWeight1,
                          float *overallScale);
void weightScalingAspect ( vector<Point2f>& matchPoints1, vector<Point2f>& matchPoints2, float *overallScale);
float kernelGaussian2 ( float x );
float kernelGaussian ( float x );
void shiftPoints( vector<Point2f>& points, CvRect roi);
float findBC(vector<float>& hist1, vector<float>& hist2);
void clusterStats(Mat& clusters, Mat& descriptors, Mat& labels, vector<int>& memberCount, vector<float>& statNeeded);

void gaussianWeight(vector<Point2f>& points, vector<float>& pWeight1, CvRect roi);
void weightedPDF ( vector<Point2f>& IPoints, CvRect roi, Mat& labels, int noc, vector<float>& pdf );
void searchBin ( Mat& desc2, Mat& clusters, Mat& labels2);
void findWeights ( vector<float>& prevPDF, vector<float>& predPDF, Mat& labels1, Mat& labels2,
                  vector<float>& weights, vector<int>& queryIdxs, Mat& desc1, Mat& desc2);
void findNewCenter ( vector<Point2f>& IPts2, vector<float> weights, CvRect roi, Point2f& z );

int getMatcherFilterType( const string& str );
void simpleMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
                           const Mat& descriptors1, const Mat& descriptors2,
                           vector<DMatch>& matches12 );
void crossCheckMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
                               const Mat& descriptors1, const Mat& descriptors2,
                               vector<DMatch>& filteredMatches12, int knn );
void boundaryCheckRect(Rect& box);
void findDI(Ptr<DescriptorMatcher>& descriptorMatcher, Mat& desc1, Mat& desc2, vector<KeyPoint>& kp2,
            double& DI, vector<Point2f>& Mpoints3);


#endif
