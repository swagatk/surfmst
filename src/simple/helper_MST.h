#ifndef HELPER_H
#define HELPER_H

/**
    class.h
    Purpose: All Helper utilites found here
    @author Mayank Jain ( mailTo: mayank10.j@tcs.com )
    @version 0.1 16/10/2012
*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

#if CV_MINOR_VERSION > 3
    #include "opencv2/nonfree/nonfree.hpp"
#endif


#include <iostream>
#include <stdio.h>
#include <fstream>
#include <iomanip>

using namespace cv;
using namespace std;



#define DRAW_RICH_KEYPOINTS_MODE     0
#define DRAW_OUTLIERS_MODE           0
#define BE_QUITE                     0
#define CLUSTER_COUNT                64

#define DROP_PERCENTAGE 20.0

#define REJECT_FRACTION 0.0



//enum { NONE_FILTER = 0, CROSS_CHECK_FILTER = 1 };


// Function for indexed quicksort
void quicksort(vector<double>& main, vector<int>& index) ;

// Function for appending matrix
void appendMatrix( cv::Mat &originalMat,const Mat &matToBeAppended );

// Function for appending matrix horizontally
void appendMatrixHorz( cv::Mat &originalMat,const cv::Mat& matToBeAppended );

// print vector
void printVector(vector<double>& vec) ;


void printVector(vector<int>& vec) ;


void printVector(vector<float>& vec) ;


int max_element_index ( vector<double>& data );


int min_element_index ( vector<double>& data );




// Function for normalizing <double> histogram

void normalizeHistogram(vector<double>& hist) ;

//static int getMatcherFilterType( const string& str )
//{
//    if( str == "NoneFilter" )
//        return NONE_FILTER;
//    if( str == "CrossCheckFilter" )
//        return CROSS_CHECK_FILTER;
//    CV_Error(CV_StsBadArg, "Invalid filter name");
//    return -1;
//}

//static void simpleMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
//                           const Mat& descriptors1, const Mat& descriptors2,
//                           vector<DMatch>& matches12 )
//{
//    vector<DMatch> matches;
//    descriptorMatcher->match( descriptors1, descriptors2, matches12 );
//}

//static void crossCheckMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
//                               const Mat& descriptors1, const Mat& descriptors2,
//                               vector<DMatch>& filteredMatches12, int knn=1 )
//{
//    filteredMatches12.clear();
//    vector<vector<DMatch> > matches12, matches21;
//    descriptorMatcher->knnMatch( descriptors1, descriptors2, matches12, knn );
//    descriptorMatcher->knnMatch( descriptors2, descriptors1, matches21, knn );
//    for( size_t m = 0; m < matches12.size(); m++ )
//    {
//        bool findCrossCheck = false;
//        for( size_t fk = 0; fk < matches12[m].size(); fk++ )
//        {
//            DMatch forward = matches12[m][fk];

//            for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ )
//            {
//                DMatch backward = matches21[forward.trainIdx][bk];
//                if( backward.trainIdx == forward.queryIdx )
//                {
//                    filteredMatches12.push_back(forward);
//                    findCrossCheck = true;
//                    break;
//                }
//            }
//            if( findCrossCheck ) break;
//        }
//    }
//}










#endif
