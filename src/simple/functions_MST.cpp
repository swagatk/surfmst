/**
File Name: fuctions_MST.cpp
Title: SURF-based Mean-shift Tracker
Author: Sourav Garg (garg.sourav@tcs.com)
--------------- */

#include "functions_MST.h"
#include "helper_MST.h"

std::ofstream eachIter, eachImage, trajectory, wtVsCord, zVsBC, pointsCount;
std::ifstream gtZ;
extern Rect box,boxOrg,GTbox;
extern bool drawing_box ;
extern bool rect_drawn ;
extern float prevScale ;
extern Mat clustersOrg, labelsOrg;
extern vector<Point2f> pointsORG;
extern vector<int> IndexOrg1, IndexOrg2;
extern cv::VideoWriter writeOut, circleAnalysis;
enum { NONE_FILTER = 0, CROSS_CHECK_FILTER = 1 };

/*

  Draws a box on the input Image

*/
void draw_box(Mat img1, CvRect rect ){
    cv::rectangle( img1, cvPoint(rect.x, rect.y), cvPoint( rect.x+rect.width,rect.y+rect.height ),
                   Scalar(0,0,255),2 );
}

void colorTheClusters(Mat& img, Rect roi, vector<Point2f>& kp, Mat& labels)
{
    double r[NOC] = {0,255,0,255,0},g[NOC] = {255,0,255,0,0}, b[NOC] = { 0, 255, 255,0,255} ;
    // 1. Green
    // 2. Magenta
    // 3. Yellow
    // 4. Blue
    // 5. Red

    Scalar color;
    //      for (int i = 0; i< NOC; i++)
    //      {
    //          r[i] = 255 * ( rand() / (double)RAND_MAX );
    //          g[i] = 255 * ( rand() / (double)RAND_MAX );
    //          b[i] = 255 * ( rand() / (double)RAND_MAX );
    //      }

    //          for (int i = 0; i< NOC; i++)
    //          {
    //              r[i] = 200 * i/NOC;
    //              g[i] = 100 * i/NOC;
    //              b[i] = 50 * i/NOC;
    //          }


    for (uint k=0; k<kp.size(); k++)
    {
        color = Scalar(r[labels.at<int>(k)],g[labels.at<int>(k)],b[labels.at<int>(k)]);
        cv::circle(img, kp[k], 2, color, 2);
    }
}

// Gaussian Kernel Function
float kernelGaussian ( float x )
{
    float k;
    k = ( exp(-x*x/2) ) / ( 44/7 );
    return k;
}


// Gaussian g(x) = - k'(x)
float kernelGaussian2 ( float x )
{
    float k;
    k = x * ( exp(-x*x/2) ) / ( 44/7 );
    return k;
}

// Epanechnikov Kernel
float EpanechKernel(float x)
{
    float k;

    if (x*x < 1)
        k = 0.75 * (1 - x*x);
    else
        k = 0;

    return k;
}

// g(x) = -k'(x)
float EpanechKernel2(float x)
{
    //float k;
    //    if (x*x < 1)
    //        k = 0.75 * (1 - x*x);
    //    else
    //        k = 0;

    return 0.75;
}

// Gaussian weights based on the distance from ROI's center
void gaussianWeight(vector<Point2f>& points, vector<float>& pWeight1, CvRect roi)
{
    float h;
    Point2f center;

    if ( roi.height > roi.width )
        h = roi.height/2;
    else
        h = roi.width/2;

    center.x = roi.x + roi.width/2;
    center.y = roi.y + roi.height/2;

    for (unsigned int i = 0; i < points.size(); i++)
    {
        pWeight1[i] = kernelGaussian(( (norm(center - points[i])) / h ));
    }
}


/*
  Calculating Epanechnikov weights based on distance of those points from the ROI's center
*/

void Epanechnikov( vector<Point2f>& points, vector<float>& pWeight1, CvRect roi )
{
    float s, h;
    Point2f center;

    if ( roi.height > roi.width )
        h = roi.height*0.5;
    else
        h = roi.width*0.5;

    center.x = roi.x + roi.width/2;
    center.y = roi.y + roi.height/2;

    for (unsigned int i = 0; i< points.size(); i++)
    {
        s = norm( center - points[i] ) / h;
        if (s*s < 1)
            pWeight1[i] = ( 0.75 * ( 1 - ( s * s ) ));
        else
            pWeight1[i] = 0;

    }
}


/*

 Implement mouse callback

*/

void my_mouse_callback( int event, int x, int y, int flags, void* param )
{
    switch( event )
    {
    case CV_EVENT_MOUSEMOVE:
        if( drawing_box ){
            box.width = x-box.x;
            box.height = y-box.y;
        }
        break;

    case CV_EVENT_LBUTTONDOWN:
        drawing_box = true;
        box = cvRect( x, y, 0, 0 );
        break;

    case CV_EVENT_LBUTTONUP:
        drawing_box = false;
        rect_drawn=true;
        if( box.width < 0 ){
            box.x += box.width;
            box.width *= -1;
        }
        if( box.height < 0 ){
            box.y += box.height;
            box.height *= -1;
        }
        //draw_box(  box );
        break;
    }
}

// Shifts given vector of Points to the location of the ROI of original image
void shiftPoints( vector<Point2f>& points, CvRect roi)
{
    Point2f shift(roi.x, roi.y);
#if DEBUG
    cout << "Shifting Points from ROI to whole Image..." << endl;
#endif
    for ( unsigned int i=0; i < points.size(); i++)
        points[i] += shift;
}


// find BC for histograms
float findBC(vector<float>& hist1, vector<float>& hist2)
{
    float BC=0;

    for (unsigned int i=0; i<hist1.size(); i++)
    {
        BC += sqrt(hist1[i]*hist2[i]);
    }
    return BC;
}

/*

    Apply Scaling using the corresponding matching pairs in two images

*/

void weightScalingAspect ( vector<Point2f>& matchPoints1, vector<Point2f>& matchPoints2, vector<float>& pWeight1,
                           float *overallScale)
{
    int count=0, ind=0, num = matchPoints1.size();
    vector<float> scale, weight;
    float totalWeight=0;
    //  std::ofstream scaleData;
    //  scaleData.open("scaleData.txt",ios::out);
    if (matchPoints1.size() < 2 )
        cout << "MatchPoints less than 2 while calculating Scaling" << endl;

    // Computing scale factor and corresponding weight of scaling for all possible pairs of matched points in both x and y dirn
    for (int i = 0; i < num; i++)
        for (int j=i+1; j< num; j++)
        {
            if ((matchPoints1[i].x - matchPoints1[j].x) != 0 && (matchPoints1[i].y - matchPoints1[j].y) != 0)
            {
                scale.push_back(norm(matchPoints2[i] - matchPoints2[j])/norm(matchPoints1[i] - matchPoints1[j]));

                if ( scale[count]< (1+float(SCALE_RANGE)/100) && scale[count] > (1-float(SCALE_RANGE)/100) )
                {
                    weight.push_back( pWeight1[i] + pWeight1[j]);
                    totalWeight += weight[count];
                    //  scaleData << count<<"\t"<<scale[count]<<"\t"<<weight[count]<<"\t"<<matchPoints1[i]<<"\t"<<matchPoints1[j]<<"\t"<<matchPoints2[i]<<"\t"<<matchPoints2[j]<<endl;
                    count++;
                }
                else
                    scale.pop_back();
            }
            ind++;
        }

    *overallScale=0;

    // Computing overall scaling for the new ROI
    for (int i=0; i<count; i++ )
        *overallScale += weight[i] * scale[i];

    if (totalWeight == 0)
    {
        cout << "scaling weights are zero" << endl;
        // Last Scaling Value used in case of lesser matching points for finding scaling
        *overallScale = prevScale;
    }
    else
        *overallScale /= totalWeight;

    // Store Last Scaling Value
    prevScale = *overallScale;

    // scaleData.close();
}
// Find the weight of descriptors based on their belongingness to histogram bins
void weights4Descriptors (Mat &tpdescriptors, Mat& labels, int noc, vector<float>& pdf )
{
    float C = 0;

    // finding pdf
    for (int i = 0; i < tpdescriptors.rows; i++)
    {
        if (labels.at<int>(i) != -1)
        {
            if(!CHANGE_KERNEL)
                pdf[labels.at<int>(i)] += 1 ;
            else
                pdf[labels.at<int>(i)] += 1 ;
        }
    }

    // finding normalization constant and multiplying with pdf
    for (int j = 0; j < tpdescriptors.rows; j++)
        if (labels.at<int>(j) != -1)
        {
            if(!CHANGE_KERNEL)
                C += 1 ;
            else
                C += 1;

        }
    for (int i = 0; i < noc; i++)
    {
        if (C == 0)
            cout << "All weights zero while finding Histogram " << endl;
        else
            pdf[i] /= C;
    }
}

// Applying kernel as weights to the Interest Points for pdf

// weightedPDF ( Input Image cordinates of Ipts, ROI for its center and h, numberOfClusters, Output weighted PDF)
void weightedPDF ( vector<Point2f>& IPoints, CvRect roi, Mat& labels, int noc, vector<float>& pdf )
{
    float C = 0, h; // C is the normalization constant
    Point2f center;
    center.x = roi.x + roi.width/2;
    center.y = roi.y + roi.height/2;

    //    cout << "center = \t" << center << endl;
    // h is the bandwidth of this kernel
    if ( roi.height > roi.width )
        h = roi.height;
    else
        h = roi.width;

    // finding weighted pdf
    for (unsigned int i = 0; i < IPoints.size(); i++)
    {
        if (labels.at<int>(i) != -1)
        {
            if(!CHANGE_KERNEL)
                pdf[labels.at<int>(i)] += 1 * kernelGaussian(( (norm(center - IPoints[i])) / h ));
            else
                pdf[labels.at<int>(i)] += 1 * EpanechKernel(( (norm(center - IPoints[i])) / h ));

        }
    }

    // finding normalization constant and multiplying with pdf
    for (unsigned int j = 0; j < IPoints.size(); j++)
        if (labels.at<int>(j) != -1)
        {
            if(!CHANGE_KERNEL)
                C += 1 * kernelGaussian( (norm(center - IPoints[j])) / h);
            else
                C += 1 * EpanechKernel((norm(center - IPoints[j])) / h);

        }
    for (int i = 0; i < noc; i++)
    {
        if (C == 0)
            cout << "All weights zero while finding Histogram " << endl;
        else
            pdf[i] /= C;
    }
}


//Searching best cluster(<noc>-Bin) for a set of descriptors that comes from new ROI iteratively

// searchBin ( Input matrix of descriptors, Input clusters centers to be matched to, Output Matrix of labels for descriptors)
void searchBin ( Mat& desc2, Mat& clusters, Mat& labels2)
{
    float distFromCluster, minDis=999, clusterThresh ;
    eachIter << "searchBin- MindistFromCluster for " << desc2.rows << " points" << endl;
    if (desc2.rows <= NOC)
        cout << "Warning: Points less than noc" << endl;
    for (int i = 0; i < desc2.rows; i++)
    {
        int index = -1;
        minDis = 999;
        for (int j = 0; j < clusters.rows; j++)
        {
            clusterThresh = 999;

            distFromCluster = norm(desc2.row(i) - clusters.row(j));

            if (distFromCluster < minDis && distFromCluster < clusterThresh)
            {
                minDis = distFromCluster;
                index = j;
            }
        }
        labels2.push_back(index);
        eachIter << minDis << endl;
    }
}


// Find weights for each Ipoint, in order to calculate new position for ROI

// findWeights ( weighted PDF of original, weighted PDF from iteration, cluster labels 1, cluster labels 2, Output weights )
void findWeights ( vector<float>& sourcePDF, vector<float>& targetPDF, Mat& labels1, Mat& labels2,
                   vector<float>& weights, vector<int>& queryIdxs, Mat& desc1, Mat& desc2)
{
    eachIter << "Weights for " << labels2.rows << " points" << endl;

    for (int i = 0; i < labels2.rows; i++)
    {
        if (labels2.at<int>(i) == -1)
            weights[i] = 0;
        else
        {
            if (targetPDF[labels2.at<int>(i)] == 0 )
            {
                targetPDF[labels2.at<int>(i)] = 1;
            }
            weights[i] = sqrt( sourcePDF[labels1.at<int>(queryIdxs[i])] / targetPDF[labels2.at<int>(i)] );
        }
        eachIter << weights[i] << endl;
    }
}


// Finding new location of ROI for next iteration

void findNewCenter ( vector<Point2f>& IPts2, vector<Point2f>& Ipts1, vector<float> weights, CvRect roi, Point2f& z )
{
    float C = 0, h; // C is the normalization constant
    Point2f center;
    center.x = float(roi.x) + float(roi.width/2);
    center.y = roi.y + roi.height/2;

    // h is the bandwidth of the kernel
    if ( roi.height > roi.width )
        h = roi.height;
    else
        h = roi.width;

    // finding new location
    for (unsigned int i = 0; i < IPts2.size(); i++)
    {
        if (!CHANGE_KERNEL)
            z += (weights[i] * 100 * kernelGaussian2( (norm(center - IPts2[i])) / h ) )* (IPts2[i]);
        else
            z += (weights[i] * 100 * EpanechKernel2((norm(center - IPts2[i])) / h ) )* (IPts2[i]);


        wtVsCord << ( kernelGaussian2( (norm(center - IPts2[i])) / h ) ) << "\t" << weights[i] << "\t" << IPts2[i].x << "\t" << IPts2[i].y << endl;

        //        eachIter << z << weights[i] << kernelGaussian2((norm(center - IPts2[i])) / h) << IPts2[i] << endl;
    }

    // finding normalization constant and multiplying with pdf
    for (unsigned int i = 0; i < IPts2.size(); i++)
    {
        if (!CHANGE_KERNEL)
            C += weights[i] * 100 * ( kernelGaussian2( (norm(center - IPts2[i])) / h) );
        else
            C += weights[i] * 100 * ( EpanechKernel2((norm(center - IPts2[i])) / h) );

        //        eachIter << C << endl;
    }
    if (C==0)
        cout << "All weights zero while finding new Z" << endl;
    else
        z *= (1/C);

    eachIter << "New center for the iteration \t" << z << endl;

}



// SURF Matching Functions from sample code

int getMatcherFilterType( const string& str )
{
    if( str == "NoneFilter" )
        return 0;
    if( str == "CrossCheckFilter" )
        return 1;
    CV_Error(CV_StsBadArg, "Invalid filter name");
    return -1;
}

void simpleMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
                     const Mat& descriptors1, const Mat& descriptors2,
                     vector<DMatch>& matches12 )
{
    vector<DMatch> matches;
    descriptorMatcher->match( descriptors1, descriptors2, matches12 );
}

void crossCheckMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
                         const Mat& descriptors1, const Mat& descriptors2,
                         vector<DMatch>& filteredMatches12, int knn=1 )
{
    filteredMatches12.clear();
    vector<vector<DMatch> > matches12, matches21;
    descriptorMatcher->knnMatch( descriptors1, descriptors2, matches12, knn );
    descriptorMatcher->knnMatch( descriptors2, descriptors1, matches21, knn );
    for( size_t m = 0; m < matches12.size(); m++ )
    {
        bool findCrossCheck = false;
        for( size_t fk = 0; fk < matches12[m].size(); fk++ )
        {
            DMatch forward = matches12[m][fk];

            for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ )
            {
                DMatch backward = matches21[forward.trainIdx][bk];
                if( backward.trainIdx == forward.queryIdx )
                {
                    filteredMatches12.push_back(forward);
                    findCrossCheck = true;
                    break;
                }
            }
            if( findCrossCheck ) break;
        }
    }
}

void boundaryCheckRect(Rect& box)
{
    if (box.x < 0)
        box.x = 0;
    else if ((box.x + box.width) > 639 )
        box.width += 639 - box.width - box.x;
    if (box.y < 0)
        box.y = 0;
    else if ((box.y + box.height) > 479 )
        box.height += 479 - box.height - box.y;
}
//----------------------------------
double similarityMeasure(
        Mat &bDescriptors, Mat &tpDescriptors,
        Ptr<DescriptorMatcher>& descriptorMatcher, int matcherFilter)
{
    vector<DMatch> filteredMatches;

    //switch( matcherFilter )
    // {
    //case CROSS_CHECK_FILTER :
    crossCheckMatching( descriptorMatcher, bDescriptors, tpDescriptors, filteredMatches, 1 );
    //    break;
    //default:
    //    simpleMatching( descriptorMatcher, bDescriptors, tpDescriptors, filteredMatches );
    // }

    vector<int> queryIdxs, trainIdxs;
    for( size_t i = 0; i < filteredMatches.size(); i++ )
    {
        queryIdxs.push_back(filteredMatches[i].queryIdx);
        trainIdxs.push_back(filteredMatches[i].trainIdx);
    }



    Mat mDesc1, mDesc2;
    for(size_t i=0; i<queryIdxs.size(); i++)
    {
        mDesc1.push_back(bDescriptors.row(queryIdxs[i]));
        mDesc2.push_back(tpDescriptors.row(trainIdxs[i]));

    }


    uint MP = 0;;
    double disSum = 0.0;
    for( int i = 0; i < mDesc1.rows; i++ )
    {
        double descDiff = pow(norm(mDesc1.row(i) - mDesc2.row(i)), 2);
        if(descDiff < 0.06)
        {
            MP++;
            disSum += descDiff;
        }
    }
    disSum = disSum / MP;

    double percMatch = ((double) MP / tpDescriptors.rows) * exp(-1.0 * disSum);


    mDesc1.release();
    mDesc2.release();
    queryIdxs.clear();
    trainIdxs.clear();
    filteredMatches.clear();

    return percMatch;
}

//======================================================================
// It is a static function. It is visible only in this file.
// It is not necessary to declare it in the header file "functions_MST.h"

static void doIteration( Mat& img1, Mat& img2, //source & destination image or ROI
                         vector<KeyPoint> &keypoints1,Mat &descriptors1,
                         vector<KeyPoint> &keypoints2,Mat &descriptors2,
                         vector<int>& queryIdxs, vector<int>& trainIdxs, // these are returned
                         Ptr<DescriptorMatcher>& descriptorMatcher, int matcherFilter,
                         vector<Point2f>& matchedPoints1, vector<Point2f>& matchedPoints2,
                         Mat& matchedDesc1, Mat& matchedDesc2)
{
    assert( !img1.empty() );
    assert( !img2.empty() );

    // Compute SURF descriptors for target image ROI
    cv::SURF mySURF;    mySURF.extended = 0;

    mySURF.detect(img2, keypoints2 );
    mySURF.compute(img2, keypoints2, descriptors2 );

    // cout << "d2 size DO_ITE \t" << descriptors2.rows << endl;
    // detector->detect( img2, keypoints2 );
    //descriptorExtractor->compute( img2, keypoints2, descriptors2 );

    vector<DMatch> filteredMatches;
    switch( matcherFilter )
    {
    case CROSS_CHECK_FILTER :
        crossCheckMatching( descriptorMatcher, descriptors1, descriptors2, filteredMatches, 1 );
        break;
    default :
        simpleMatching( descriptorMatcher, descriptors1, descriptors2, filteredMatches );
    }

    trainIdxs.clear();
    queryIdxs.clear();
    for( size_t i = 0; i < filteredMatches.size(); i++ )
    {
        queryIdxs.push_back(filteredMatches[i].queryIdx); // indices of source Descriptors
        trainIdxs.push_back(filteredMatches[i].trainIdx); // indices of Destination Descriptors
    }

    //cout << "Number of Matching points between source and destination = " << filteredMatches.size() << endl;

    Mat H12;
    if( RANSAC_THREHOLD >= 0 )
    {
        vector<Point2f> points1; KeyPoint::convert(keypoints1, points1, queryIdxs);
        vector<Point2f> points2; KeyPoint::convert(keypoints2, points2, trainIdxs);
        if (points2.size() < 4 )
        {
            cout << "matchedPoints1 less than 4, hence prev ROI is retained" << endl;
            return;
        }

        H12 = findHomography( Mat(points1), Mat(points2), CV_RANSAC, RANSAC_THREHOLD );
    }

    Mat drawImg;

    if( !H12.empty() ) // filter outliers
    {
        vector<char> matchesMask( filteredMatches.size(), 0 );
        vector<Point2f> points1; KeyPoint::convert(keypoints1, points1, queryIdxs);
        vector<Point2f> points2; KeyPoint::convert(keypoints2, points2, trainIdxs);

        Mat points1t; perspectiveTransform(Mat(points1), points1t, H12);

        //        //int pCount=-1;
        //        vector<Point2f> points2Shift(points2.size());
        //        points2Shift = points2;
        //        shiftPoints(points2Shift, box);

        Point2f boxCenter;
        boxCenter.x = box.x + box.width/2;
        boxCenter.y = box.y + box.height/2;

        for( size_t i1 = 0; i1 < points1.size(); i1++ )
        {
            if( ( norm(points2[i1] - points1t.at<Point2f>((int)i1,0)) <= MAX_INLIER_DIST ))
            {
                matchesMask[i1] = 1;

                matchedPoints1.push_back( points1[i1]);
                matchedPoints2.push_back( points2[i1]);

                matchedDesc1.push_back(descriptors1.row(queryIdxs[i1]));
                matchedDesc2.push_back(descriptors2.row(trainIdxs[i1]));

            }
        }

        // draw inliers
        drawMatches( img1, keypoints1, img2, keypoints2, filteredMatches, drawImg,
                     CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask

             #if DRAW_RICH_KEYPOINTS_MODE
                     , DrawMatchesFlags::DRAW_RICH_KEYPOINTS
             #endif
                     );
    }
    else
        drawMatches( img1, keypoints1, img2, keypoints2,
                     filteredMatches, drawImg );

    imshow( "winName", drawImg );
    // waitKey(0);
}

//=========================
void reprojectPoints(vector<Point2f>& matchedPoints1, Mat& matchedDesc1, Mat& matchedDesc2, Mat& img1ROI,
                     Mat& pointsTransed21, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2,
                     Mat& descriptors1, vector<int> &reprojections)
{
    // Sort the matchin
    vector<double> eucl(matchedPoints1.size(), 0);
    vector<int> sortedIndex(matchedPoints1.size(), 0);
    for (unsigned int i=0; i<matchedPoints1.size(); i++)
    {
        eucl[i] = norm(matchedDesc1.row(i)-matchedDesc2.row(i));
        sortedIndex[i] = i;
    }

    quicksort(eucl, sortedIndex);

    int transCount = round(TOP_X * matchedPoints1.size() * 0.01); // Number of points to be reprojected
    int reprojCount=0;

    Mat imgTemp1= img1ROI.clone();
    if (transCount != 0 && !pointsTransed21.empty() )
    {
        for (int i=0; i<transCount; i++)
        {
            //                        keypoints2[IndexOrg2[sortedIndex[i]]].pt = pointsTransed21.at<Point2f>(sortedIndex[i]);
            cv::circle(imgTemp1,pointsTransed21.at<Point2f>(sortedIndex[i]), 4, Scalar(0,0,255), 1 );
            cv::circle(imgTemp1,keypoints1[IndexOrg1[sortedIndex[i]]].pt, 6, Scalar(255,255,0), 1 );
            //                        cv::circle(imgTemp1,pointsTransed21_.at<Point2f>(sortedIndex[i]), 8, Scalar(255,0,255), 1 );

            // Pushing back, not replacing
            if ( int(keypoints2[IndexOrg2[sortedIndex[i]]].pt.x - boxOrg.width) > 10 ||
                 int(keypoints2[IndexOrg2[sortedIndex[i]]].pt.y - boxOrg.height) > 10)
            {
                // cout << "KeyPoint out of range" << endl;
                continue;
            }

            //                        int projCount = findProjectionsCount(keypoints1, keypoints2[IndexOrg2[sortedIndex[i]]].pt);

            if (reprojections[IndexOrg1[sortedIndex[i]]] <= REPROJ_ALLOWED)
            {
                keypoints1.push_back( keypoints1[IndexOrg1[sortedIndex[i]]]);
                descriptors1.push_back( matchedDesc2.row(sortedIndex[i]));
                reprojCount++;
                reprojections[IndexOrg1[sortedIndex[i]]] += 1;
                reprojections.push_back( reprojections[IndexOrg1[sortedIndex[i]]] );
            }
        }

        cv::imshow("PDF",imgTemp1);
    }
    else
        cout << "Trans Count is ZERO \n" << endl;
}

//=============================================================

// Mean Shift Algorithm

void meanShift (Mat &img1, Mat &img2,  // source and destination images
                Ptr<DescriptorMatcher> &descriptorMatcher, int matcherFilterType, // ??
                vector<KeyPoint>& keypoints1, Mat& descriptors1, // source features
                vector<KeyPoint>& keypoints2, Mat& descriptors2, // Destination features
                Mat& clusters1,   // clusters obtained from K-means (used for creating Histograms)
                vector<int>& reprojections, // do we need it?
                Point2f &cp,  // Centre of the new target window
                int& flag, int &count, int &MP,
                Mat &temp, vector<int> &queryIdxs, vector<int> &trainIdxs)
{

    Mat img2ROI, img1ROI, labels1_;
    Point2f lastCenter(box.x+box.width/2,box.y+box.height/2);
    //float scale = 1; //  ??

    img1ROI = img1(boxOrg);         // box = tracker window on Source Image
                                    // box, boxOrg are global variables.
    vector<Point2f> points1;
    KeyPoint::convert(keypoints1, points1);
    shiftPoints(points1, boxOrg); // Translating points wrt to origin of boxOrg not for entire Image
    //int pointsInside = keypoints1.size();

    //Compute cluster labels for each descriptor
    searchBin(descriptors1, clusters1, labels1_);

    // Making histogram/pdf by normalizing the data and applying weights based on positions
    vector<float> sourcePDF(NOC); // pdf of source
    weightedPDF( points1, boxOrg, labels1_, NOC, sourcePDF);

    // mean-shift Iterations for finding the object in the target image

    Point2f zBCmax(0,0); // center corresponding to the iteration with max BC
    // Bhattachrya coefficient, max val for an image, previous and current val for an iteration
    float BCmax=0, BCprev=0, BCnow=0;
    int stopCount=0; // MS iterations must converege for stopCount < ManualSetThreshold
    int converged = 0, semiConverged = 0;

    while ( !converged )
    {
        vector<Point2f> matchedPoints1, matchedPoints2;
        Mat matchedDesc1, matchedDesc2;
        queryIdxs.clear(); trainIdxs.clear();
        cv::Rect expandedBox;

        //cout << "iteration in while = \t" << ++iter << endl;

        if (EXPANDED_BOX)
        {
            expandedBox = Rect(box.x-25, box.y-25, box.width+50, box.height+50);
            boundaryCheckRect(expandedBox);
            img2ROI = img2(expandedBox);
        }
        else
        {
            img2ROI = img2(box);   //initial window in the target image
                                   // This will be changed by this function
        }

        vector<Point2f> points2;
        Mat  pointsTransed21;


        // It returns matched points and descriptors in the source and the target ROI
        // Does not modify the box variable.
        doIteration( img1ROI, img2ROI, keypoints1, descriptors1,
                     keypoints2, descriptors2, queryIdxs, trainIdxs,
                     descriptorMatcher, matcherFilterType,
                     matchedPoints1, matchedPoints2, matchedDesc1, matchedDesc2);

        //	cout << "descs2 size in MS \t " << descriptors2.rows << endl;

        KeyPoint::convert(keypoints2, points2);

        if (matchedPoints2.empty())
        {
            // trajectory << box.x+box.width/2 << "\t" << box.y+box.height/2 << "\t" << box.width << "\t" << box.height << endl;
            cout << "Matched Points = 0, Hence Exiting" << endl;
            return;
        }

        if (EXPANDED_BOX)
        {
            shiftPoints(points2, expandedBox);
            shiftPoints(matchedPoints2, expandedBox);
        }
        else
        {
            shiftPoints(points2, box);
            shiftPoints(matchedPoints2, box);
        }

        shiftPoints(matchedPoints1,boxOrg);

        //vector<float> targetPDF(NOC,0); // predicted PDF for target ROI
        vector<float> targetPDF(NOC,0); // predicted PDF for target ROI

        Mat labels2, labels2_; // depending on PDF_OF_WHOLE
        Point2f z(0,0);

        // allot descriptors to the clusters to make histogram
        searchBin( matchedDesc2, clusters1, labels2);

        if (PDF_OF_WHOLE)
            searchBin( descriptors2, clusters1, labels2_);

        // find the PDF for the above histogram as per weights
        if (PDF_OF_WHOLE)
            weightedPDF( points2, box, labels2_, NOC, targetPDF);
        else
            weightedPDF( matchedPoints2, box, labels2, NOC, targetPDF);

        // find weights for each IPoint as per the values of weighted PDFs
        vector<float> weights(labels2.rows,0);
        Mat imgTemp = img2.clone();
        findWeights( sourcePDF, targetPDF, labels1_, labels2, weights, queryIdxs, matchedDesc1, matchedDesc2);

        // find new ROI center as per above weights
        findNewCenter(matchedPoints2, matchedPoints1, weights, box, z); // New centre is 'z'

        lastCenter = Point2f (box.x+box.width/2, box.y+box.height/2);

        // if current BC is less than previous BC, then take mean of the prev and current centers
        // Bhattacharya Coefficient between source and target histograms.
        BCnow = findBC(targetPDF,sourcePDF);

        if (BCnow < BCprev)
            z = 0.5*(z+lastCenter);

        BCprev = BCnow;

        // check if ROI centers converge to same pixel
        if ( (norm(z - lastCenter) < 3))
        {
            semiConverged = 1;
            if (!SHOW_FINAL_ROI)
                rectangle(temp, box, Scalar(0,0,255),2);
        }
        else
        {
            // keep iterating
            stopCount++;

            if (stopCount < MAX_MS_ITER)
            {
                if (!SHOW_FINAL_ROI)
                    rectangle(temp, box, Scalar(0,255,0), 2);
            }
            else if (stopCount >= MAX_MS_ITER)
            {
                semiConverged = 1;
                flag = 0;            // Mean-shift fails to converge within MAX_MS_ITER steps.
                if (!SHOW_FINAL_ROI)
                    rectangle(temp, box, Scalar(0,0,255),2);
                z = zBCmax;
            }

            box.x = z.x - box.width/2;
            box.y = z.y - box.height/2;
            boundaryCheckRect(box);
        }

        // store values of max BC and corresponding center z
        if ( BCnow > BCmax)
        {
            //BCmax = BC;    // what is the value of BC here? It should be BCnow, I guess!
            BCmax = BCnow;
            zBCmax = z;
        }

        if (semiConverged)
        {
            converged = 1;
            //  tempF.close();

            box.x = z.x - box.width/2;
            box.y = z.y - box.height/2;

            imgTemp.release();

//            if (APPLY_SCALING)
//            {
//                vector<float> epenWeight(matchedPoints1.size(),0);
//                Epanechnikov(matchedPoints2, epenWeight, box );
//                //                gaussianWeight(matchedPoints1, epenWeight, box);
//                weightScalingAspect(matchedPoints1, matchedPoints2, epenWeight, &scale);

//                box.x += box.width/2;
//                box.y += box.height/2;
//                box.height = round(boxOrg.height *scale);
//                box.width = round(( float(boxOrg.width)/float(boxOrg.height) ) * box.height);
//                box.x -= box.width/2;
//                box.y -= box.height/2;

//            }
            // box should be within the boundary of the image frame
            boundaryCheckRect(box);

            // trajectory << z.x << "\t" << z.y << "\t" << box.width << "\t" << box.height << endl;

            //Centre of the final target window
            cp = z;

//            if(IMPROVING_PDF)
//            {
//                reprojectPoints(matchedPoints1, matchedDesc1, matchedDesc2, img1ROI,
//                                     pointsTransed21, keypoints1, keypoints2,
//                                     descriptors1, reprojections);
//            }

            for(uint j=0; j<matchedPoints2.size();j++)
                cv::circle(temp, matchedPoints2[j], 2, Scalar(255,0,255), 2);

            cv::circle(temp, z, 2, Scalar(0,255,255), 2);
            cv::rectangle(temp, box, Scalar(255,0,0),2);

            writeOut.write(temp);
        }

        MP=matchedPoints2.size();
        count = stopCount;
        // cv::imshow("Iter", temp);

        char c = (char)waitKey(10);
        if( c == '\x1b' ) // esc
        {
            cout << "Exiting from while iterator..." << endl;
            break;
        }
    }

    eachIter.close();

}

//=======================================================
// Contributed by Meenakshi
// Date: July 30, 2013



//============================================

static void shiftPoints_G2L( vector<Point2f>& points, CvRect roi)
{
    Point2f shift(roi.x, roi.y);
    for ( unsigned int i=0; i < points.size(); i++)
        points[i] -= shift;
}

//==============================================


void weightScalingAspect ( vector<Point2f>& matchPoints1, vector<Point2f>& matchPoints2, double *overallScale)
{
    int count=0,  num = matchPoints1.size();
    vector<double> scale;

    if (matchPoints1.size() > 1 )
    {
        for (int i = 0; i < num; i++)
            for (int j=i+1; j< num; j++)
            {

                if ((matchPoints1[i].x - matchPoints1[j].x) != 0 && (matchPoints1[i].y - matchPoints1[j].y) != 0)
                {
                    scale.push_back(norm(matchPoints2[i] - matchPoints2[j])/norm(matchPoints1[i] - matchPoints1[j]));

                    if ( scale[count]< (1+float(SCALE_RANGE)/100) && scale[count] > (1-float(SCALE_RANGE)/100) )
                    {
                        // scaleData << count<<"\t"<<scale[count]<<"\t"<<weight[count]<<"\t"<<matchPoints1[i]<<"\t"<<matchPoints1[j]<<"\t"<<matchPoints2[i]<<"\t"<<matchPoints2[j]<<endl;
                        count++;
                    }
                    else
                        scale.pop_back();
                }
            }


        *overallScale=0;

        // Computing overall scaling for the new ROI
        for (int i=0; i<count; i++ )
            *overallScale += scale[i];

        *overallScale /= count;

    }
    else
    {
        cout << "MatchPoints less than 3 while calculating Scaling" << endl;
        *overallScale=1;
    }
}

//===========================================
void DivideAndComputeScaling(Mat &img1, Mat &img2)
{

    Rect box1 = boxOrg;
    Rect box2 = box;

    double scale;
    Mat img1ROI,img2ROI,temp1,temp2;
    temp1= img1.clone();
    temp2 = img2.clone();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    Mat H12;
    vector<Point2f> Pt1,Pt2,p1,p11,p22,p3;
    vector<int> queryIdxs,  trainIdxs;
    Rect B1,B2,B3;
    cv::SURF mySURF;    mySURF.extended = 0;

    B1.x = box1.x; B1.y = box1.y; B1.width= box1.width; B1.height =  int(0.15*box1.height);
    B2.x = box1.x; B2.y = box1.y + int(0.15*box1.height); B2.width= box1.width;
    B2.height =  int(0.4*box1.height);
    B3.x = box1.x; B3.y = box1.y + int(0.55*box1.height); B3.width= box1.width;
    B3.height =  int(0.45*box1.height);

    img1ROI = img1(box1);
    img2ROI = img2(box2);

    mySURF.detect(img1ROI, keypoints1);
    mySURF.compute(img1ROI, keypoints1, descriptors1);
    mySURF.detect(img2ROI, keypoints2);
    mySURF.compute(img2ROI, keypoints2, descriptors2);

    vector<DMatch> filteredMatches;
    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( "FlannBased" );
    crossCheckMatching( descriptorMatcher, descriptors1, descriptors2, filteredMatches, 1 );

    for( size_t i = 0; i < filteredMatches.size(); i++ )
    {
        queryIdxs.push_back(filteredMatches[i].queryIdx);
        trainIdxs.push_back(filteredMatches[i].trainIdx);
    }

    vector<Point2f> points1; KeyPoint::convert(keypoints1, points1, queryIdxs);
    vector<Point2f> points2; KeyPoint::convert(keypoints2, points2, trainIdxs);

    H12 = findHomography( Mat(points1), Mat(points2), CV_RANSAC, RANSAC_THREHOLD );

    vector<char> matchesMask( filteredMatches.size(), 0 );

    Mat points1t; perspectiveTransform(Mat(points1), points1t, H12);


    for( size_t i1 = 0; i1 < points1.size(); i1++ )
    {
        if( ( norm(points2[i1] - points1t.at<Point2f>((int)i1,0)) <= 20 ))
        {
            matchesMask[i1] = 1;
            Pt1.push_back( points1[i1]);
            Pt2.push_back( points2[i1]);
        }
    }

    shiftPoints(Pt1, box1);
    shiftPoints(Pt2, box2);

    for(size_t i=0;i<Pt1.size();i++)
    {
        if(B1.contains(Pt1[i]))
        {
            p1.push_back(Pt2[i]);
            circle(temp1, Pt1[i], 2, Scalar(255,0,0),2);
            circle(temp2, Pt2[i], 2, Scalar(255,0,0),2);
        }
        else if(B2.contains(Pt1[i]))
        {
            p11.push_back(Pt1[i]);
            p22.push_back(Pt2[i]);
            circle(temp1, Pt1[i], 2, Scalar(0,255,0),2);
            circle(temp2, Pt2[i], 2, Scalar(0,255,0),2);
        }
        else
        {
            p3.push_back(Pt2[i]);
            circle(temp1, Pt1[i], 2, Scalar(0,0,255),2);
            circle(temp2, Pt2[i], 2, Scalar(0,0,255),2);
        }
    }


    appendMatrixHorz(temp1,temp2);
    Point2f tpt;
    for(size_t i=0;i<Pt1.size();i++)
    {
        tpt.x=Pt2[i].x+640;   tpt.y =Pt2[i].y;
        cv::line(temp1,Pt1[i],tpt,Scalar(255,0,0),1,8,0);
    }


    shiftPoints_G2L(p11,box1);
    shiftPoints_G2L(p22,box2);
    weightScalingAspect(p11,p22,&scale);


    rectangle(temp1, B1, Scalar(255,0,0),2);
    rectangle(temp1, B2, Scalar(0,255,0),2);
    rectangle(temp1, B3, Scalar(0,0,255),2);

    Point2f cp;
    cp.x = box2.x+ box2.width/2.0;
    cp.y = box2.y+ box2.height/2.0;

    float aspectRatio = box1.width/(float) box1.height;
    box2.height = int (scale*box2.height);
    box2.width= int(aspectRatio*box2.height);

    box2.x = int(cp.x - box2.width/2.0);
    box2.y = int(cp.y - box2.height/2.0);

    Rect box2N=box2;
    box2N.x = box2.x+640;

    box = box2;

    rectangle(temp1, box2N, Scalar(255,0,0),2);
    imshow("img",temp1);

}
//==========================
