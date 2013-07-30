#include "helper_MST.h"
#include "functions_MST.h"

typedef unsigned int uint;

//std::ofstream eachIter, eachImage, wtVsCord, trajectory, pointsCount, storePDF, storeBC;
//std::ifstream gtZ;


//Do we need all these Global variables
//==========================================
Rect box,boxOrg,GTbox;
bool drawing_box = false;
bool rect_drawn = false;
float prevScale = 1;
Mat clustersOrg, labelsOrg;
vector<Point2f> pointsORG;
vector<int> IndexOrg1, IndexOrg2;
cv::VideoWriter writeOut, circleAnalysis, showMatches, showReproj;
const string winName = "correspondences";




//=================================================
// Main Function
//===========================================

int main(int argc, char** argv)
{


    Point2f cp;
    cv::initModule_nonfree();

    // Read the VIDEO
    VideoCapture cap("video1.avi");
    if( !cap.isOpened() )
    { cout << "Could not initialize capturing...\n"; return 0;}

    //Initialize Video Writer
    //writeOut.open("MStrack_3.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15, Size(640,480), 1 );

    cv::SURF mySURF;    mySURF.extended = 0;
    Ptr<FeatureDetector> detector = FeatureDetector::create( "SURF"); // SURF,SIFT ,MSER
    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create( "SURF" );
    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( "FlannBased" ); // FlannBased , BruteForce
    int matcherFilterType = getMatcherFilterType( "CrossCheckFilter" );

    // Get the first frame and select the ROI to be tracked in the subsequent frames
    Mat frame, img1, img2;
    cap >> frame;

    if( frame.empty() )
        return -1;
    else
        img1 = frame.clone() ;

    Mat temp = img1.clone() ;

    if(img1.empty())
    {
        cout << "Exiting as the input image is empty" << endl;
        exit(-1);
    }

    const char* name = "Initiate_ROI";
    box = cvRect(-1,-1,0,0);


    cvNamedWindow( name );

    // Set up the callback
    cvSetMouseCallback( name, my_mouse_callback);

    // Wait until ROI is selected by the user
    while( 1 )
    {
        img1.copyTo(temp);

        if( drawing_box )
            draw_box( temp, box );
        cv::imshow(name,temp) ;

        cvWaitKey( 15 );
        if(rect_drawn)
            break;
    }

    // storing the initial selected Box, as "box" variable changes in consecutive matching
    boxOrg = box;

    Mat img1ROI, labels1, clusters1, descriptors1, descriptors2;
    vector<int> reprojections; // number of reprojections per keypoint, size same as keypoint (increasing)
    vector<KeyPoint> keypoints1, keypoints2;

    //human aspect ratio (not used)
    double aspectRatio = (double)box.width / box.height;


    // Compute SURF features within the *selected* ROI

    img1ROI = img1(boxOrg);
    mySURF.detect(img1ROI, keypoints1 );
    mySURF.compute(img1ROI, keypoints1, descriptors1 );


    // Create a Template Pool that contains both descriptors as well as Keypoints (local & Global)

    Mat tpDescriptors;
    vector<KeyPoint> tpKeypoints;
    vector<float> tpWeights;

   int tpMaxSize = 1000;

    //Initially copy of the descriptor of Ist image ROI into it.
    descriptors1.copyTo(tpDescriptors);
    tpWeights.resize(tpDescriptors.rows,2.0); // Initial values of all weights is 2.0

    for(uint i = 0; i < keypoints1.size(); ++i)
        tpKeypoints.push_back(keypoints1.at(i));


    //==========================================
    // Main Iteration Loop starts here : Tracking
    //============================================

    int MP, count;
    struct timeval t1, t2;
    //Rect msBox; // Box obtained from mean-shift tracker

    // Loop over all images
    for(int k=1;;k++) //int i=2;i<1002;i+=1)
    {
        gettimeofday(&t1, NULL);

        //create clusters in the SURF descriptor space
        // clusters are created in the template pool
        cv::kmeans(tpDescriptors, NOC, labels1,
                   TermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 50, 1.0 ), 1,
                   /*KMEANS_PP_CENTERS*/KMEANS_RANDOM_CENTERS, clusters1);


        // img1 - source image
        // img2 - Destination image
        // Mean-shift algorithm returns the window on Destination image given the source ROI in boxOrg.


        //Capture a New frame
        cap >> frame;
        if( frame.empty() )
            return -1;
        else
            img2 = frame.clone() ;

        temp = img2.clone();

        if(img2.empty() )
        {
            cout<< "Could not open image: " << img2 << endl ;
            //continue ;
            exit(-1);
        }

        int flag=1;  // what is this flag ??
        MP=0; count=0;  // ??

        //Call the mean-shift tracker
        vector<int> queryIdxs, trainIdxs;
        meanShift(img1, img2, descriptorMatcher, matcherFilterType, keypoints1, descriptors1,
                  keypoints2, descriptors2, clusters1, reprojections,  cp, flag, count, MP,
                  temp, queryIdxs, trainIdxs);



        DivideAndComputeScaling(img1, img2);

//        box.height = (int)(scaleValue * box.height);
//        box.width = (int)(aspectRatio * box.height);
//        box.x = cp.x - box.width/2.0;
//        box.y = cp.y - box.height/2.0;

        //cout << "Scale Value = " << scaleValue << endl;


         // Add the target ROI descriptors into the template pool.
         for(int i=0;i< descriptors2.rows;i++)
         {
             tpDescriptors.push_back(descriptors2.row(i));
             tpKeypoints.push_back(keypoints2.at(i));
         }

         // If the size of template pool exceeds max size, remove that many number of points from top
         Mat tempMat;
         if(tpDescriptors.rows > tpMaxSize)
         {
             //cout << "Time to Truncate Template Pool" << endl;
             uint dLength = tpDescriptors.rows - tpMaxSize;
             tempMat = tpDescriptors.rowRange(Range(dLength, tpDescriptors.rows));
             tpKeypoints.erase(tpKeypoints.begin(), tpKeypoints.begin()+dLength);

             //tpDescriptors.release(); tpDescriptors = tempMat;
             tpDescriptors = tempMat;
         }
         tempMat.release();
         //cout << "Template Pool size =" << tpDescriptors.rows << endl;

         // Current target image becomes the source image for the next iteration
         img1=img2.clone();
         boxOrg = box;

         // source descriptors and keypoints are taken from the template pool
         keypoints1 = tpKeypoints;
         descriptors1 = tpDescriptors;



        gettimeofday(&t2, NULL);
        double diff = (float)((t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec));
        diff = diff/1000;
        cout << k << "\tTime taken in mili sec \t" <<  diff<< endl;
        //f1 <<  k << "\t" << MP << "\t"   << count  << "\t"   << diff << "\n";



        cv::circle(temp, cp, 2, Scalar(0,255,255), 2);
        //=======================================


        imshow("main", temp);
        //imshow("img2", img2);


        char c = (char)waitKey(10);
        if( c == '\x1b' ) // esc
        {
            cout << "Exiting ..." << endl;
            break;
        }

        waitKey(5);

    }
    return 0;
}






