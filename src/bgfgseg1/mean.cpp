
// Copied from MSTmoreChanges.cpp
// Removing redundant code
// video and file writings also removed

#include "helper_MST.h"
#include "functions_MST.h"

std::ofstream eachIter, eachImage, wtVsCord, trajectory, pointsCount, storePDF, storeBC;
std::ifstream gtZ;
Rect box,boxOrg,GTbox,boxP,boxOuter;//bbox;
bool drawing_box = false;
bool rect_drawn = false;
bool drawing_poly = true;
bool poly_drawn = false;
float prevScale = 1;
Mat clustersOrg, labelsOrg;
vector<Point2f> pointsORG;
vector<int> IndexOrg1, IndexOrg2;
cv::VideoWriter writeOut, circleAnalysis, showMatches, showReproj;
vector<Point> polyPoints, IpolyPoints;

const string winName = "correspondences";

enum { NONE_FILTER = 0, CROSS_CHECK_FILTER = 1 };

void my_mouse_callback2( int event, int x, int y, int flags, void* param ){
    switch( event ){
    case CV_EVENT_LBUTTONDOWN:
        drawing_poly = true;
        polyPoints.push_back(cv::Point(x,y));
        cout << x << "\t" << y << endl;
        IpolyPoints.push_back(cv::Point(x,y));
        break;

    case CV_EVENT_RBUTTONDOWN:
        drawing_poly = false;
        poly_drawn=true;
        break;
    }
}


static void doIteration(Mat& img1, Mat& img2, vector<int>& queryIdxs, vector<int>& trainIdxs, Ptr<DescriptorMatcher>& descriptorMatcher,
                        int matcherFilter,vector<KeyPoint>& keypoints1,  Mat& descriptors1, vector<KeyPoint>& keypoints2,Mat& descriptors2,
                        Mat& matchedDesc1, Mat& matchedDesc2, vector<Point2f>& matchedPoints1, vector<Point2f>& matchedPoints2,
                        vector<Point2f>& MP1, vector<KeyPoint>& tempkey)
{
    assert( !img2.empty());
    cv::SURF mySURF;    mySURF.extended = 0;
    Mat H12;

    mySURF.detect(img2, keypoints2);    mySURF.compute(img2, keypoints2, descriptors2);
    vector<DMatch> filteredMatches;
    switch( matcherFilter )
    {
    case CROSS_CHECK_FILTER :
        crossCheckMatching( descriptorMatcher, descriptors1, descriptors2, filteredMatches, 1 );
        break;
    default :
        simpleMatching( descriptorMatcher, descriptors1, descriptors2, filteredMatches );
    }

    trainIdxs.clear();    queryIdxs.clear();

    for( size_t i = 0; i < filteredMatches.size(); i++ )
    {
        queryIdxs.push_back(filteredMatches[i].queryIdx);
        trainIdxs.push_back(filteredMatches[i].trainIdx);
    }

    //////


    Mat mDesc1, mDesc2;
    for(size_t i=0; i<queryIdxs.size(); i++)
    {
        mDesc1.push_back(descriptors1.row(queryIdxs[i]));
        mDesc2.push_back(descriptors2.row(trainIdxs[i]));
    }


    vector<Point2f> points1; KeyPoint::convert(keypoints1, points1, queryIdxs);
    vector<Point2f> points2; KeyPoint::convert(keypoints2, points2, trainIdxs);
    vector<char> matchesMask( filteredMatches.size(), 0 );//,  matchesMask2( filteredMatches.size(), 1 );;

    Mat drawImg;// drawImg2;

    cout << "points2.size \t" << points2.size() << endl;
    cout <<"HELLO \t" << endl;

    if( RANSAC_THREHOLD >= 0 )
    {
        if (points2.size() < 4 )
        {
            cout << "matchedPoints1 less than 4, hence prev ROI is retained" << endl;

            for(size_t i1=0;i1<points2.size();i1++)
            {
                matchesMask[i1] = 1;
                matchedPoints1.push_back( points1[i1]);
                matchedPoints2.push_back( points2[i1]);

                matchedDesc1.push_back(descriptors1.row(queryIdxs[i1]));
                matchedDesc2.push_back(descriptors2.row(trainIdxs[i1]));

                tempkey.push_back(keypoints2[trainIdxs[i1]]);
                MP1.push_back(points2[i1]);
            }
        }
        else
        {
            H12 = findHomography( Mat(points1), Mat(points2), CV_RANSAC, RANSAC_THREHOLD );

            if( !H12.empty() )
            {

                Mat points1t; perspectiveTransform(Mat(points1), points1t, H12);

                vector<Point2f> points2Shift(points2.size());
                points2Shift = points2;
                shiftPoints(points2Shift, box);
                Point2f boxCenter;
                boxCenter.x = box.x + box.width/2;
                boxCenter.y = box.y + box.height/2;

                for( size_t i1 = 0; i1 < points1.size(); i1++ )
                {
                    double descDiff = pow(norm(mDesc1.row(i1) - mDesc2.row(i1)) , 2);
                    //  if(descDiff < 0.08)
                    {
                        double diff = norm(points2[i1] - points1t.at<Point2f>((int)i1,0));
                        if(diff  <= 30)
                        {
                          //  cout << diff << endl;
                            matchesMask[i1] = 1;
                            matchedPoints1.push_back( points1[i1]);
                            matchedPoints2.push_back( points2[i1]);

                            matchedDesc1.push_back(descriptors1.row(queryIdxs[i1]));
                            matchedDesc2.push_back(descriptors2.row(trainIdxs[i1]));

                            tempkey.push_back(keypoints2[trainIdxs[i1]]);
                            MP1.push_back(points2[i1]);
                        }
                    }

                }
            }
            //              drawMatches( img1, keypoints1, img2, keypoints2, filteredMatches, drawImg2, CV_RGB(255, 255, 0), CV_RGB(255,255, 255), matchesMask2
            //              #if DRAW_RICH_KEYPOINTS_MODE
            //                                   , DrawMatchesFlags::DRAW_RICH_KEYPOINTS
            //              #endif
            //                                 );
            drawMatches( img1, keypoints1, img2, keypoints2, filteredMatches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask
             #if DRAW_RICH_KEYPOINTS_MODE
                         , DrawMatchesFlags::DRAW_RICH_KEYPOINTS
             #endif
                         );

            cout << endl;
            imshow( "doiter", drawImg );

//            Mat newimg = img1.clone();
//             KeyPoint::convert(keypoints1, points1);
//            for(size_t i=0;i<points1.size();i++)
//                 circle(newimg, points1[i], 2, Scalar(255,0,255),2);

//             imshow( "doimg", newimg );
//            points1.clear();
// waitKey(0);

        }
    }
    // waitKey(0);

}


// Mean Shift Algorithm
void meanShift(Mat& img1, Mat& img2, Ptr<DescriptorMatcher>& descriptorMatcher, int matcherFilterType, vector<KeyPoint>& tpkeypoints,
               Mat& tpdescriptors, vector<KeyPoint>& keypoints2, Mat& descriptors2, Mat& clusters1, Point2f &cp, int& flag,
               vector<Point2f>& MP1, Mat& img2ROI, vector<KeyPoint>& bkeypoints, Mat& bdescriptors, Mat& temp,
               int& FG_mp, int&FG, int& BG_mp, int& BG, int& FG_BG, int& msI)
{
    size_t i,j;
    // Mat temp=img2.clone();
    int converged = 0, semiConverged = 0;
    Mat img1ROI, labels1_;
    Point2f lastCenter(box.x+box.width/2,box.y+box.height/2);
    float scale = 1;

    img1ROI = img1(boxOrg);
    vector<Point2f> points1, bmp;
    KeyPoint::convert(tpkeypoints, points1);
    searchBin(tpdescriptors, clusters1, labels1_); // clustering based on Kmeans centers obatined earlier

//vector<Point2f> np;
//    Mat newimg = img1ROI.clone();
//     KeyPoint::convert(tpkeypoints, np);
//    for(size_t i=0;i<np.size();i++)
//        circle(newimg, np[i], 2, Scalar(255,0,255),2);

//     imshow( "msimg", newimg );
//    np.clear();

//    waitKey(0);


    vector<float> prevPDF(NOC); // pdf of source
    weightedPDF( points1, boxOrg, labels1_, NOC, prevPDF); // Making histogram/pdf by normalizing the data and applying weights based on positions

    // Iterations for finding the object

    Point2f zBCmax(0,0); // center corspndng to the iteration with max BC
    float BC, BCmax=0, BCprev=0, BCnow=0; // Bhattachrya coefficient, max val for an image, previous and current val for an iteration
    int stopCount=0; // MS iterations must converege for stopCount < ManualSetThreshold
    vector<Point2f> matchedPoints1, matchedPoints2;
    while ( !converged )
    {
       // ofstream tempF;
        //tempF.open("tempF.txt", ios::out);
        matchedPoints1.clear(); matchedPoints2.clear();

        Mat matchedDesc1, matchedDesc2;
        vector<int> queryIdxs, trainIdxs;

        cv::Rect expandedBox;

#ifdef DEBUG
        cout << "iteration in while = \t" << ++iter << endl;
#endif

        if (EXPANDED_BOX)
        {
            expandedBox = Rect(box.x-25, box.y-25, box.width+50, box.height+50);
            boundaryCheckRect(expandedBox);
            img2ROI = img2(expandedBox);
        }
        else
        {
          //  cout << box.br() << "\t" << box.tl() << "\t" << img2.cols << endl;
            img2ROI = img2(box);
        }

        vector<KeyPoint> tempkey;
       // Mat pointsTransed21;
        MP1.clear();

        doIteration(img1ROI, img2ROI, queryIdxs, trainIdxs,descriptorMatcher, matcherFilterType, tpkeypoints,tpdescriptors,
                    keypoints2,descriptors2, matchedDesc1, matchedDesc2,  matchedPoints1, matchedPoints2, MP1,tempkey);

        if(matchedPoints2.size() < 1)
        {
            FG=0; BG=0;FG_mp=0;BG_mp=0;FG_BG=0; msI=0;
            break;

        }
        //  mdescriptors = matchedDesc2;

        //   KeyPoint::convert(keypoints2, points2);


        if (EXPANDED_BOX)
            shiftPoints(matchedPoints2, expandedBox);
        else
            shiftPoints(matchedPoints2, box);


        // shiftPoints(matchedPoints1,boxOrg);
        vector<float> predPDF(NOC,0);

        Mat labels2, labels2_; // depending on PDF_OF_WHOLE
        Point2f z(0,0);


        //==================== Edited at 8th april =======================//
        bmp.clear();
        Mat tmatchedDesc2, tmatchedDesc1;
        vector<Point2f> tmatchedPoints2;
        msI = stopCount;
        FG_mp = matchedPoints2.size();

        vector<KeyPoint> tempbk;
        Mat tempBd;
        vector<DMatch> filteredMatches;
        crossCheckMatching( descriptorMatcher, bdescriptors, descriptors2, filteredMatches, 1 );
        trainIdxs.clear();    queryIdxs.clear();

        for( i = 0; i < filteredMatches.size(); i++ )
        {
            queryIdxs.push_back(filteredMatches[i].queryIdx);
            trainIdxs.push_back(filteredMatches[i].trainIdx);
        }


        vector<Point2f> points1; KeyPoint::convert(bkeypoints, points1, queryIdxs);
        vector<Point2f> points2;   KeyPoint::convert(keypoints2, points2, trainIdxs);
        vector<char> matchesMask( filteredMatches.size(), 0 );
        /////
        Mat H12;
        Mat drawImg;
        if (points2.size() < 4 )
        {
            cout << "backpoints less than 4, hence prev ROI is retained" << endl;
            return;
            for(i=0;i<points2.size();i++)
            {
                bmp.push_back( points2[i]);
                tempBd.push_back(bdescriptors.row(queryIdxs[i]));
                tempbk.push_back(keypoints2[trainIdxs[i]]);
                tempBd.push_back(descriptors2.row(trainIdxs[i]));
                tempbk.push_back(keypoints2[trainIdxs[i]]);
            }
        }
        else
        {
            H12 = findHomography( Mat(points1), Mat(points2), CV_RANSAC, RANSAC_THREHOLD );

            if( !H12.empty() )
            {
                Mat points1t; perspectiveTransform(Mat(points1), points1t, H12);

                for( size_t i1 = 0; i1 < points1.size(); i1++ )
                {
                    double diff = norm(points2[i1] - points1t.at<Point2f>((int)i1,0));
                    if(diff  <= 20)
                    {
                        matchesMask[i1]=1;
                        bmp.push_back( points2[i1]);
                        tempBd.push_back(bdescriptors.row(queryIdxs[i1]));
                        tempbk.push_back(keypoints2[trainIdxs[i1]]);
                        tempBd.push_back(descriptors2.row(trainIdxs[i1]));
                        tempbk.push_back(keypoints2[trainIdxs[i1]]);

                    }
                }

                drawMatches( img1ROI, bkeypoints, img2ROI, keypoints2, filteredMatches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask
             #if DRAW_RICH_KEYPOINTS_MODE
                             , DrawMatchesFlags::DRAW_RICH_KEYPOINTS
             #endif
                             );

            }
        }
        imshow("bm",drawImg);

        //============edit part ====
        shiftPoints(bmp, box);
        vector<int> bflag(bmp.size(),0);

        for(i=0;i<bmp.size();i++)
            bflag[i]=0;

        vector<int> ft(matchedPoints2.size(),0);

        for(i=0;i<matchedPoints2.size();i++)
        {
            ft[i]=0;
            for(j=0; j< bmp.size(); j++)
            {
                double diff = norm (matchedPoints2[i] - bmp[j]);
                // cout << diff << endl;
                if(diff < 0.5)
                {
                    bflag[j]=1;
                    ft[i]=1;
                    break;
                }
            }
            if(ft[i]==0)
            {
                tmatchedPoints2.push_back(matchedPoints2[i]);
                tmatchedDesc1.push_back(matchedDesc1.row(i));
                tmatchedDesc2.push_back(matchedDesc2.row(i));
            }

        }




        //=================================================================//


        // allot descriptors to the clusters to make histogram
        searchBin(tmatchedDesc1, clusters1, labels1_);
        searchBin( tmatchedDesc2, clusters1, labels2);
        if (PDF_OF_WHOLE)
            searchBin( descriptors2, clusters1, labels2_);

        // find the PDF for the above histogram as per weights
        if (PDF_OF_WHOLE)
            weightedPDF( points2, box, labels2_, NOC, predPDF);
        else
            weightedPDF( tmatchedPoints2, box, labels2, NOC, predPDF);

        // find weights for each IPoint as per the values of weighted PDFs
        vector<float> weights(labels2.rows,0);
        Mat imgTemp = img2.clone();
        findWeights( prevPDF, predPDF, labels1_, labels2, weights, queryIdxs, tmatchedDesc1, tmatchedDesc2);

        // find new ROI center as per above weights
        findNewCenter(tmatchedPoints2, weights, box, z);

        lastCenter = Point2f (box.x+box.width/2, box.y+box.height/2);

        // if current BC is less than previous BC, then take mean of the prev and current centers
        BCnow = findBC(predPDF,prevPDF);
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

            if (stopCount >= MAX_MS_ITER)
            {
                semiConverged = 1;
                flag = 0;
                if (!SHOW_FINAL_ROI)
                    rectangle(temp, box, Scalar(0,0,255),2);
                z = zBCmax;
            }

            box.x = z.x - box.width/2;
            box.y = z.y - box.height/2;
            boundaryCheckRect(box);

            if (stopCount < MAX_MS_ITER)
                if (!SHOW_FINAL_ROI)
                    ;// rectangle(temp, box, Scalar(0,255,0), 2);
        }

        // store values of max BC and corresponding center z
        if ( BCnow > BCmax)
        {
            BCmax = BC;
            zBCmax = z;
        }

        if (semiConverged)
        {
            converged = 1;

            //   FG_mp, FG, BG_mp, BG, FG_BG, msI ;
            //==========edited on 5april ========
            bdescriptors.release();
            bkeypoints.clear();
            for(i=0;i<tempBd.rows;i++)
            {
                bdescriptors.push_back(tempBd.row(i));
                bkeypoints.push_back(tempbk[i]);

            }


            tpdescriptors.release();
            tpkeypoints.clear();
            //============================================//

            for(i=0;i<matchedPoints2.size();i++)
            {
                if(ft[i]==0)
                {
                    tpdescriptors.push_back(matchedDesc1.row(i));
                    tpkeypoints.push_back(tempkey[i]);

                    tpdescriptors.push_back(matchedDesc2.row(i));
                    tpkeypoints.push_back(tempkey[i]);

                }
            }


//=================================
            box.x = z.x - box.width/2;
            box.y = z.y - box.height/2;

           // imgTemp.release();

            trajectory << z.x << "\t" << z.y << "\t" << box.width << "\t" << box.height << endl;

            cp =z;


            cv::circle(temp, z, 3, Scalar(0,255,255), 3);
            cv::rectangle(temp, box, Scalar(255,0,0),2);

            cout << "MP1 \t" << MP1.size() <<"\t" << "bmp \t"  <<bmp.size() << endl;

            for(size_t i=0;i<MP1.size();i++)
            {//circle(temp, MP1[i], 3, Scalar(255,255,255),3);
              circle(temp, matchedPoints2[i], 3, Scalar(255,0,255),3);
            }

            // shiftPoints(bmp,box);
           for(size_t i=0;i<bmp.size();i++)
             { circle(temp, bmp[i], 2, Scalar(0,0,0),2);
              // cout << bmp[i] << endl;
           }

        }

        char c = (char)waitKey(10);
        if( c == '\x1b' ) // esc
        {
            cout << "Exiting from while iterator..." << endl;
            break;
        }
    }




    cv::imshow("Iter", temp);
  //  waitKey(0);
    eachIter.close();

}


int main(int argc, char** argv)
{
    ofstream f1;
    f1.open("result.txt");
    size_t i,j;
    Point2f cp;
    cv::initModule_nonfree();
    vector<Point2f> MP1,MP2;
    vector<int> trainIdxs, queryIdxs;

    //Read Video File
    VideoCapture cap("video1.avi");
    if( !cap.isOpened() )
    { cout << "Could not initialize capturing...\n"; return 0;}



    VideoWriter writer("ms_tracking.avi",CV_FOURCC('D','I','V','3'),
                 10,cvSize(640,480),1);

    cv::SURF mySURF;    mySURF.extended = 0;
    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( "FlannBased" );
    int mactherFilterType = getMatcherFilterType( "CrossCheckFilter" );

    Mat frame,img1,img2;
    cap >> frame;
    if( frame.empty() )
        return -1;
    img1 = frame.clone() ;
    Mat temp,temp1;

    if(img1.empty())
        cout << "Exiting as the input image is empty" << endl;


    const char* name = "Initiate_ROI";
    box = cvRect(-1,-1,0,0);
    cvNamedWindow( name,1);
    cvSetMouseCallback( name, my_mouse_callback2);

    // Main loop
    while( 1 )
    {
        img1.copyTo(temp);

        if( drawing_poly)
        {

            for ( i=0; i < polyPoints.size(); i++)
                circle(temp, polyPoints[i], 2, Scalar(0,255,0), -1,8);
        }
        cv::imshow(name,temp) ;
        char c = (char)waitKey(10);
        if( c == '\x1b' ) // esc
            break;
        if(poly_drawn)
            break;
    }

    //Read the polygon points from a text file

    FILE *f11;
    polyPoints.clear();
    IpolyPoints.clear();
    f11 = fopen("points.txt","r");
    Point a;
    for(int j=0;j<37;j++)
    {
        fscanf(f11,"%d",&(a.x));
        fscanf(f11,"%d",&(a.y));
        polyPoints.push_back(a);
        IpolyPoints.push_back(a);
    }
    fclose(f11);

    // Drawing Polygon
    Point pointArr[polyPoints.size()];
    for (i=0; i< polyPoints.size(); i++)
        pointArr[i] = polyPoints[i];
    const Point* pointsArray[1] = {pointArr};
    int nCurvePts[1] = { polyPoints.size() };
    polylines(temp, pointsArray, nCurvePts, 1, 1, Scalar(0,255,0), 1);

    cout << polyPoints.size() << endl;
    box= boundingRect(polyPoints);

   //boxOrg = Rect(box.x-15, box.y-15, box.width+30, box.height+30);
   boxOuter = Rect(box.x-30, box.y-30, box.width+60, box.height+60);
    //box =boxOrg; // storing the initial selected Box, as "box" variable changes in consecutive matching
    boxP=box;
    Mat img1ROI, labels1, clusters1, descriptors,roidescriptors, descriptors1,bdescriptors, bmdescriptors;
    vector<int> reprojections; // number of reprojections per KP, size same as KP(incresing)
    vector<Point2f> points,points1,points2, Mpoints1,Mpoints2,bpoints,npoints1,npoints2; //bmpoints,tpoints;
    vector<KeyPoint> roikeypoints, bkeypoints,keypoints,keypoints1, keypoints2;


    draw_box(temp, box ); //Show InnerBox  - This is used by the Mean-Shift Tracker
    draw_box(temp,boxOuter); //Show OuterBox - This is used for removing background points
    bpoints.clear();

    //calculating keypoints and descriptors of the selected polygon in image roi
    //==============================================================================================//
    for(i=0;i<polyPoints.size();i++)
    {
        // cout << polyPoints[i] << endl; //
        polyPoints[i].x = polyPoints[i].x -boxOuter.x;
        polyPoints[i].y = polyPoints[i].y- boxOuter.y;
    }

    img1ROI = img1(boxOuter);
    points1.clear();
    mySURF.detect(img1ROI, roikeypoints);
    KeyPoint::convert(roikeypoints, points);
    mySURF.compute(img1ROI, roikeypoints, roidescriptors);

    bdescriptors.release();bkeypoints.clear();
    bcategorizePoints( points, bpoints,polyPoints, roikeypoints, roidescriptors, bkeypoints, bdescriptors);
    shiftPoints(bpoints,boxOuter);
    for(i=0;i<bpoints.size();i++)
        circle(temp, bpoints[i], 2, Scalar(0,255,0),2);

  vector<KeyPoint> tpkeypoints;    Mat tpdescriptors;
    categorizePoints( points, points1,polyPoints, roikeypoints, roidescriptors, tpkeypoints, tpdescriptors);

    shiftPoints(points1, boxOuter);
    for(i=0;i<points1.size();i++)
        circle(temp, points1[i], 2, Scalar(0,0,255),2);
    //====================================================================================================//
    points1.clear();
    Mat img2ROI;

  //  tpkeypoints = keypoints1;    tpdescriptors = descriptors1;
    cv::imshow(name,temp) ;
    imwrite("a.jpg",temp);
    cout << "BD_SIZE \t" << bdescriptors.rows << "\t" << "FD_SIZE \t"  << tpdescriptors.rows << endl;


//    Mat newimg = img1ROI.clone();
//     KeyPoint::convert(tpkeypoints, points1);
//    for(size_t i=0;i<points1.size();i++)
//         circle(newimg, points1[i], 2, Scalar(255,0,255),2);

//     imshow( "newimg", newimg );
//    points1.clear();

    waitKey(0);
    cvDestroyWindow( name );


    int FG_mp, FG, BG_mp, BG, FG_BG, msI ; //Foreground matching points
    struct timeval t1, t2;

    for(int l=0;;l++)
    {
        gettimeofday(&t1, NULL);
        cv::kmeans(tpdescriptors, NOC, labels1, TermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 50, 1.0 ), 1,
                   KMEANS_RANDOM_CENTERS, clusters1);

        cap >> frame;
        img2 = frame.clone() ;
        temp1 =frame.clone() ;

        if(img2.empty() )
        {
            cout<< "Could not open image: " << endl ;
            break;}

        int flag=1;
        Mpoints1.clear();
        Mat descriptors2;

        msI=0;

        meanShift(img1, img2, descriptorMatcher, mactherFilterType, tpkeypoints, tpdescriptors,keypoints2,descriptors2,
                  clusters1, cp, flag, MP1,img2ROI,bkeypoints, bdescriptors, temp1,FG_mp, FG, BG_mp, BG, FG_BG,msI);



        //==========scaling=================
        float scale=1;

       // cout <<"MP1size \t" << MP1.size() <<endl;

        if(APPLY_SCALING)
        {
            vector<DMatch> filteredMatches;

            if(descriptors1.rows > 4 && descriptors2.rows > 4)
            {
                crossCheckMatching( descriptorMatcher, descriptors1, descriptors2, filteredMatches, 1 );

                trainIdxs.clear();    queryIdxs.clear();

                for( i = 0; i < filteredMatches.size(); i++ )
                {
                    queryIdxs.push_back(filteredMatches[i].queryIdx);
                    trainIdxs.push_back(filteredMatches[i].trainIdx);
                }

                points1.clear(); points2.clear();
                KeyPoint::convert(keypoints1, points1, queryIdxs);
                KeyPoint::convert(keypoints2, points2, trainIdxs);
                //  cout << "point2size" << points2.size() << endl;

                //homography

                npoints1.clear();npoints2.clear();
                Mpoints1.clear();Mpoints2.clear();
                Mat H12, points1t;
                double ransacReprojThreshold = 10;
                if( ransacReprojThreshold >= 0  && points1.size() > 4)
                    H12 = findHomography( Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold );
              vector<char> matchesMask( filteredMatches.size(), 0 );// NONmatchesMask( filteredMatches.size(), 0 );
                if( !H12.empty() )
               {

                    perspectiveTransform(Mat(points1), points1t, H12);

                    double maxInlierDist = 10;//ransacReprojThreshold < 0 ? 3 : ransacReprojThreshold;

                    for(i = 0; i < points1.size(); i++ )
                    {
                        if( norm(points2[i] - points1t.at<Point2f>((int)i,0)) <= 5)// maxInlierDist ) // inlier
                        {
                            matchesMask[i] = 1;
                            npoints2.push_back(points2[i]);
                            npoints1.push_back(points1[i]);
                        }
                    }



                    for(i=0; i<npoints2.size();i++)
                    {
                        for(j=0;j<MP1.size();j++)
                        {
                            double dist = norm(npoints2[i]-MP1[j]);
                            // cout <<"dist \t" <<dist << endl;
                            //  waitKey(0);
                            if(dist < 0.1)
                            {
                                Mpoints2.push_back(npoints2[i]);
                                Mpoints1.push_back(npoints1[i]);
                                break;
                            }

                        }
                    }



                }
                Mat drawImg;
                drawMatches( img1ROI, keypoints1, img2ROI, keypoints2, filteredMatches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask
             #if DRAW_RICH_KEYPOINTS_MODE
                             , DrawMatchesFlags::DRAW_RICH_KEYPOINTS
             #endif
                             );
                imshow( "correspondance", drawImg );
                cout << "npoints1.size \t" << Mpoints1.size() << "\t" << Mpoints2.size() << endl;
                if(Mpoints1.size() > 8)
                    weightScalingAspect(Mpoints1,Mpoints2,&scale);

            }

        }


        img1=img2;
        img1ROI = img2ROI;
        boxOrg =box;
        keypoints1 = keypoints2;
        descriptors1 =descriptors2;

        box.x += box.width/2;
        box.y += box.height/2;
        box.height = round(boxOrg.height *scale);
        box.width = round(( float(boxOrg.width)/float(boxOrg.height) ) * box.height);
        box.x -= box.width/2;
        box.y -= box.height/2;

        boundaryCheckRect(box);

        cout <<"SCALE \t" << scale << endl;

        gettimeofday(&t2, NULL);
       double diff = (float)((t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec));
       diff = diff/1000;
        cout <<"Time taken in mili sec \t" <<  diff<< endl;
       // cout << tpdescriptors.rows << endl;
        //cout <<"BD \t" << bdescriptors.rows << endl;
        f1 <<  l << "\t" << FG_mp << "\t"   << BG_mp  << "\t"   << FG   << "\t"<< msI << "\n";
        cout << "l \t" << l << "\t" <<" msI \t"<< msI << endl;
        imshow("img2",temp1);
        writer << temp1;
         waitKey(0);




       // boxOrg = eBox;

        char c = (char)waitKey(10);
        if( c == '\x1b' ) // esc
        {
            cout << "Exiting ..." << endl;
            break;
        }

    }
    trajectory.close();

    return 0;
}










