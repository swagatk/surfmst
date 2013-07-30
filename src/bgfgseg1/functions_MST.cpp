#include "functions_MST.h"

extern std::ofstream eachIter, eachImage, trajectory, wtVsCord, zVsBC, pointsCount;
extern std::ifstream gtZ;
extern CvRect box,boxOrg,GTbox,bbox,boxP;
extern bool drawing_box ;
extern bool rect_drawn ;
extern bool drawing_poly ;
extern bool poly_drawn ;
extern float prevScale ;
extern Mat clustersOrg, labelsOrg;
extern vector<Point2f> pointsORG;
extern vector<int> IndexOrg1, IndexOrg2;
extern cv::VideoWriter writeOut, circleAnalysis;

extern vector<Point> polyPoints, IpolyPoints;
//extern bool drawing_poly, poly_drawn;

/*

  Draws a box on the input Image

*/

void categorizePoints(vector<Point2f>& points1, vector<Point2f>& points2, vector<Point>& polyPoints1, vector<KeyPoint>& keypoints, Mat& descriptors,
                      vector<KeyPoint>& keypoints1, Mat& descriptors1)
{
    for (size_t i=0; i<points1.size(); i++)
    {
        double flag = cv::pointPolygonTest(polyPoints1, points1[i], false );
        if (flag != -1)
        {
            keypoints1.push_back(keypoints[i]);
            descriptors1.push_back(descriptors.row(i));
            points2.push_back(points1[i]);
        }

    }
}

void bcategorizePoints(vector<Point2f>& points1, vector<Point2f>& points2, vector<Point>& polyPoints1,vector<KeyPoint>& keypoints, Mat& descriptors,
                      vector<KeyPoint>& keypoints1, Mat& descriptors1)
{
    for (size_t i=0; i<points1.size(); i++)
    {
      // if(bbox.contains(points1[i]))
      //  {
            double flag = cv::pointPolygonTest(polyPoints1, points1[i], false );
            if (flag == -1)
            {
                keypoints1.push_back(keypoints[i]);
                descriptors1.push_back(descriptors.row(i));
                points2.push_back(points1[i]);
            }
       // }

    }
}


void categorizePoints(vector<Point2f>& points1, vector<Point>& polyPoints1, vector<KeyPoint>& keypoints, Mat& descriptors,
                      vector<KeyPoint>& tpkeypoints, Mat& tpdescriptors)
{
    for (size_t i=0; i<points1.size(); i++)
    {
        double flag = cv::pointPolygonTest(polyPoints1, points1[i], false );
        if (flag != -1)
        {
            tpkeypoints.push_back(keypoints[i]);
            tpdescriptors.push_back(descriptors.row(i));
        }
    }
}

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


    for (int k=0; k<kp.size(); k++)
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
    float k;

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


void my_mouse_callback( int event, int x, int y, int flags, void* param ){


    switch( event ){
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
    std::ofstream scaleData;
    scaleData.open("scaleData.txt",ios::out);
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
                    scaleData << count<<"\t"<<scale[count]<<"\t"<<weight[count]<<"\t"<<matchPoints1[i]<<"\t"<<matchPoints1[j]<<"\t"<<matchPoints2[i]<<"\t"<<matchPoints2[j]<<endl;
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

    scaleData.close();
}



void weightScalingAspect ( vector<Point2f>& matchPoints1, vector<Point2f>& matchPoints2, float *overallScale)
{
    int count=0, ind=0, num = matchPoints1.size();
    vector<float> scale;

    if (matchPoints1.size() > 3 )
    {
        for (int i = 0; i < num; i++)
            for (int j=i+1; j< num; j++)
            {
                if ((matchPoints1[i].x - matchPoints1[j].x) != 0 && (matchPoints1[i].y - matchPoints1[j].y) != 0)
                {
                    scale.push_back(norm(matchPoints2[i] - matchPoints2[j])/norm(matchPoints1[i] - matchPoints1[j]));
                    if ( scale[count]< (1+float(SCALE_RANGE)/100) && scale[count] > (1-float(SCALE_RANGE)/100) )
                    {

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

// Applying kernel as weights to the Interest Points for pdf

// weightedPDF ( Input Image cordinates of Ipts, ROI for its center and h, numberOfClusters, Output weighted PDF)
void weightedPDF ( vector<Point2f>& IPoints, CvRect roi, Mat& labels, int noc, vector<float>& pdf )
{
    float C = 0, h; // C is the normalization constant
//    Point2f center;
//    center.x = roi.x + roi.width/2;
//    center.y = roi.y + roi.height/2;

    //    cout << "center = \t" << center << endl;
    // h is the bandwidth of this kernel
//    if ( roi.height > roi.width )
//        h = roi.height;
//    else
//        h = roi.width;

    // finding weighted pdf
    for (unsigned int i = 0; i < IPoints.size(); i++)
    {
        if (labels.at<int>(i) != -1)
        {
            if(!CHANGE_KERNEL)
                pdf[labels.at<int>(i)] += 1 ;//* kernelGaussian(( (norm(center - IPoints[i])) / h ));
            else
                pdf[labels.at<int>(i)] += 1 ;//* EpanechKernel(( (norm(center - IPoints[i])) / h ));

        }
    }

    // finding normalization constant and multiplying with pdf
    for (unsigned int j = 0; j < IPoints.size(); j++)
        if (labels.at<int>(j) != -1)
        {
            if(!CHANGE_KERNEL)
                C += 1 ;//* kernelGaussian( (norm(center - IPoints[j])) / h);
            else
                C += 1;// * EpanechKernel((norm(center - IPoints[j])) / h);

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
void findWeights(vector<float>& prevPDF, vector<float>& predPDF, Mat& labels1, Mat& labels2,
                 vector<float>& weights, vector<int>& queryIdxs, Mat& desc1, Mat& desc2)
{
    eachIter << "Weights for " << labels2.rows << " points" << endl;

    for (int i = 0; i < labels2.rows; i++)
    {
        if (labels2.at<int>(i) == -1)
            weights[i] = 0;
        else
        {
            if (predPDF[labels2.at<int>(i)] == 0 )
            {
                predPDF[labels2.at<int>(i)] = 1;
            }
            weights[i] = sqrt( prevPDF[labels1.at<int>(i)] / predPDF[labels2.at<int>(i)] );
        }
        eachIter << weights[i] << endl;
    }
}


// Finding new location of ROI for next iteration

void findNewCenter ( vector<Point2f>& IPts2, vector<float> weights, CvRect roi, Point2f& z )
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
            z += weights[i] * (IPts2[i]);
        else
            z += weights[i] *  (IPts2[i]);


        wtVsCord << ( kernelGaussian2( (norm(center - IPts2[i])) / h ) ) << "\t" << weights[i] << "\t" << IPts2[i].x << "\t" << IPts2[i].y << endl;

        //        eachIter << z << weights[i] << kernelGaussian2((norm(center - IPts2[i])) / h) << IPts2[i] << endl;
    }

    // finding normalization constant and multiplying with pdf
    for (unsigned int i = 0; i < IPts2.size(); i++)
    {
        if (!CHANGE_KERNEL)
            C += weights[i] ;//* 100 * ( kernelGaussian2( (norm(center - IPts2[i])) / h) );
        else
            C += weights[i] ;//* 100 * ( EpanechKernel2((norm(center - IPts2[i])) / h) );

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
    {
        box.x += 639 - box.width - box.x;
//        box.width = round(( float(boxOrg.width)/float(boxOrg.height) ) * box.height)
    }
    if (box.y < 0)
        box.y = 0;
    else if ((box.y + box.height) > 479 )
    {
        box.y += 479 - box.height - box.y;
//        box.width = round(( float(boxOrg.width)/float(boxOrg.height) ) * box.height);
    }

}

void findDI(Ptr<DescriptorMatcher>& descriptorMatcher, Mat& desc1, Mat& desc2, vector<KeyPoint>& kp2,
            double& DI, vector<Point2f>& Mpoints3)
{
    vector<DMatch> filteredMatches;

    crossCheckMatching( descriptorMatcher, desc1, desc2, filteredMatches, 1 );


    vector<int> queryIdxs( filteredMatches.size() ), trainIdxs( filteredMatches.size() );

    for( int i = 0; i < filteredMatches.size(); i++ )
    { queryIdxs[i] = filteredMatches[i].queryIdx; trainIdxs[i] = filteredMatches[i].trainIdx; }

    Mat matchedDesc1, matchedDesc2;

    for(int i=0; i<queryIdxs.size(); i++)
    {
        matchedDesc1.push_back(desc1.row(queryIdxs[i]));
        matchedDesc2.push_back(desc2.row(trainIdxs[i]));
    }

    vector<Point2f> points2;   KeyPoint::convert(kp2, points2, trainIdxs);

    DI = 0;
    float descDiff = 0;
    for(int i=0; i<matchedDesc1.rows; i++)
    {
        descDiff = pow(norm(matchedDesc1.row(i) - matchedDesc2.row(i)) , 2);
        if(descDiff < 0.005)
        {
            Mpoints3.push_back(points2[i]);
        }
        DI += descDiff;
    }
    DI /= matchedDesc1.rows;

    cout << "DI = \t" << DI << endl;

}
