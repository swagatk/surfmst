// Copied from MSTmoreChanges.cpp
// Removing redundant code
// video and file writings also removed

#include "helper_MST.h"
#include "functions_MST.h"
#include "pfadd.h"      // Particle Filter related functions

typedef unsigned int uint;

std::ofstream eachIter, eachImage, wtVsCord, trajectory, pointsCount, storePDF, storeBC;
std::ifstream gtZ;

Rect box,boxOrg,GTbox;
bool drawing_box = false;
bool rect_drawn = false;
float prevScale = 1;
Mat clustersOrg, labelsOrg;
vector<Point2f> pointsORG;
vector<int> IndexOrg1, IndexOrg2;
cv::VideoWriter writeOut, circleAnalysis, showMatches, showReproj;

const string winName = "correspondences";

//enum { NONE_FILTER = 0, CROSS_CHECK_FILTER = 1 };



//=================================================
// Main Function
//===========================================

int main(int argc, char** argv)
{
    //===========================
    // GSL Random Number Generator

    gsl_rng *rg;
    long seed = time(NULL)*getpid();
    rg = gsl_rng_alloc(gsl_rng_rand48);
    gsl_rng_set(rg,seed);

    //=============================
    uint Nt = ceil(resample_percentage * Ns);

    // Instantiate a Particle Filter
    // Create a point cloud
    PF::pf cloud(Ns, Nx, Nz, SYSTEMATIC  );

    //==============================================

    int MP, count;
    Point2f cp;
    cv::initModule_nonfree();

    // Read the VIDEO
    VideoCapture cap("video1.avi");
    if( !cap.isOpened() )
    { cout << "Could not initialize capturing...\n"; return 0;}

    //Initialize Video Writer
    writeOut.open("MStrack_3.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15, Size(640,480), 1 );

    // trajectory.open("trajectoryCorrected_3.txt", ios::out);

#ifdef DEBUG
    cout << "< Creating detector, descriptor extractor and descriptor matcher ..." << endl;
#endif

    cv::SURF mySURF;    mySURF.extended = 0;
    Ptr<FeatureDetector> detector = FeatureDetector::create( "SURF"); // SURF,SIFT ,MSER
    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create( "SURF" );
    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( "FlannBased" ); // FlannBased , BruteForce
    int matcherFilterType = getMatcherFilterType( "CrossCheckFilter" );  // NoneFilter, CrossCheckFilter

    //   if( detector.empty() || descriptorExtractor.empty() || descriptorMatcher.empty()  )
    //{
    //  cout << "Can not create detector or descriptor extractor or descriptor matcher of given types" << endl;
    //  return -1;
    // }

#ifdef DEBUG
    cout << "< Reading the images..." << endl;
#endif

    // Get the first frame and select the ROI to be tracked in the subsequent frames
    Mat frame,img1,img2;
    cap >> frame;

    if( frame.empty() )
        return -1;
    else
        img1 = frame.clone() ;

    // Mat img1 = imread( "/home/meenu/ros/DataSet/ImageOutHumFol/Image1.jpg" ), img2;
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

#ifdef DEBUG
    printf("Rectangle chosen has cordinates:\nUL-->\tx=%d\ty=%d\nBR-->\tx=%d\ty=%d\n",
           box.x,box.y,box.x+box.width,box.y+box.height);
#endif
    cvDestroyWindow( name );


    boxOrg = box; // storing the initial selected Box, as "box" variable changes in consecutive matching

    Mat img1ROI, labels1, clusters1, descriptors1, descriptors2;
    vector<int> reprojections; // number of reprojections per KP, size same as KP(incresing)
    vector<KeyPoint> keypoints1, keypoints2;

    //human aspect ratio
    double aspectRatio = (double)box.width / box.height;


    // Compute SURF features within the *selected* ROI

    img1ROI = img1(boxOrg);
    mySURF.detect(img1ROI, keypoints1 );
    mySURF.compute(img1ROI, keypoints1, descriptors1 );


    // Create a Template Pool
    Mat tpDescriptors;
    descriptors1.copyTo(tpDescriptors);


    //==========================================================
    //Initialize the Particle Filter with initial SURF key points
    // obtained from selected ROI

    double var_x2 = 20.0;
    double var_y2 = 50.0;

    uint k = 0;

    std::vector<double>xloc(Nx);
    std::vector<double>zpos(Nz);
    std::vector<std::vector<double> > xp(Ns, std::vector<double>(Nx));
    std::vector<double> wt(Ns);

    vector<Point2f> initPtlocn;
    KeyPoint::convert(keypoints1, initPtlocn);

    shiftPoints(initPtlocn,box);

    //Centre of Initial Tracker Window
    zpos[0] = box.x + box.width/2.0;
    zpos[1] = box.y + box.height/2.0;


#if PF_OPT == 0
    cloud.initialize(k, 0, 1.0);
#else
  // Initial states are the current SURF key locations
    double pwt1 = 0.9;
    double pwt2 = 0.1;
    double sumwt = pwt1 *initPtlocn.size() + pwt2 * (Ns - initPtlocn.size());

    uint j = 0;
    Point2f lpt;
    while(j < Ns)
    {
        if(j < initPtlocn.size())
        {
            xp[j][0] = initPtlocn[j].x + gsl_ran_gaussian(rg, 1.0);
            xp[j][1] = 0.0 + gsl_ran_gaussian(rg, 1.0);
            xp[j][2] = initPtlocn[j].y + gsl_ran_gaussian(rg, 1.0);
            xp[j][3] = 0.0 + gsl_ran_gaussian(rg, 1.0);
            wt[j] = pwt1/sumwt; //normalized weights
            //wt[j] = 0.9;

            lpt.x = xp[j][0]; lpt.y = xp[j][2];
            cv::circle(temp, lpt, 2, Scalar(255,255,0), 2);
        }
        else
        {
            double a, b;
            bool flag = false;
            do
            {
                a = zpos[0] + gsl_ran_gaussian(rg, var_x2);
                b = zpos[1] + gsl_ran_gaussian(rg,var_y2);

                if( (a > box.x) && (a < box.x + box.width) &&
                        (b > box.y) && (b < box.y + box.height) )
                {
                    flag = true;
                }
                else
                {
                    flag = false;
                }
            }while(flag == false);

            xp[j][0] = a;
            xp[j][1] = 0.0 + gsl_ran_gaussian(rg,var_xdot);
            xp[j][2] = b;
            xp[j][3] = 0.0 + gsl_ran_gaussian(rg, var_xdot);
            wt[j] = pwt2/sumwt; // normalized weights
            //wt[j] = 0.1;

            lpt.x = xp[j][0]; lpt.y = xp[j][2];
            cv::circle(temp, lpt, 2, Scalar(0,255,0), 2); //display points
        }
        j++;
    }


#ifdef ZBIN
    vector<uint> zbin(Ns);
    for(uint j = 0; j < Ns; ++j)
    {
        if(j < initPtlocn.size())
            zbin[j] = 1;
        else
            zbin[j] = 0;
    }
    cloud.initialize(k, wt, xp, zbin);

#else
    cloud.initialize(k, wt, xp);
#endif

#endif

    imshow("init_image",temp);
    waitKey(0);
    cvDestroyWindow("init_image");


    //Initial value of the state vector

    xloc[0] = zpos[0] + gsl_ran_gaussian(rg, 1.0);
    xloc[1] = 0.0 + gsl_ran_gaussian(rg, var_xdot);
    xloc[2] = zpos[1] + gsl_ran_gaussian(rg, 1.0);
    xloc[3] = 0.0 + gsl_ran_gaussian(rg, var_xdot);

    //===========================================
    // AR Predictor for Window Centre Location
    //==========================================

#ifdef ARLOC
    // Uses only normalized values for state
   std::vector<std::vector<double> > ar_x(arlen, std::vector<double>(arDim));
   std::vector<std::vector<double> > ar_w(arlen, std::vector<double>(arDim));
   std::vector<double> ar_xhat(arDim);
   std::vector<double> ar_e(arDim);
   double ar_eta = 0.7;

   for(uint i = 0; i < arlen; ++i)
   {
       ar_x[i][0] = zpos[0] / 640.0;   // x-coordinate of window centre
       ar_x[i][1] = zpos[1] / 480.0;   // y-coordinate of window centre
       ar_x[i][2] = box.height / 480.0;        // Tracker window height

       ar_w[i][0] = 0.1;
       ar_w[i][1] = 0.1;
       ar_w[i][2] = 0.1;
   }

#endif



//  detector->detect( img1ROI, keypoints1 );
//  descriptorExtractor->compute( img1ROI, keypoints1, descriptors1 );

//    for(uint i = 0; i < keypoints1.size(); i++)
//        reprojections.push_back(0);


    //==========================================
    // Main Iteration Loop starts here : Tracking
    //============================================

    ofstream f1("result.txt");
    ofstream f2("particle.txt");

    struct timeval t1, t2;

    Rect msBox, pfBox;

    //RotatedRect R1;

    //Mat temp;
    // Loop over all images
    for(int k=1;;k++) //int i=2;i<1002;i+=1)
    {
        gettimeofday(&t1, NULL);

        //create clusters in the SURF descriptor space for the template pool
        cv::kmeans(tpDescriptors, NOC, labels1,
                   TermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 50, 1.0 ), 1,
                   /*KMEANS_PP_CENTERS*/KMEANS_RANDOM_CENTERS, clusters1);

        //  char image2Name[100] ;
        //  sprintf(image2Name,"/home/meenu/ros/DataSet/ImageOutHumFol/Image%d.jpg",i);
        // img2 = imread(image2Name);

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

        int flag=1;
        MP=0; count=0;

        vector<int> queryIdxs, trainIdxs;
        meanShift(img1, img2, box,  descriptorMatcher, matcherFilterType, keypoints1, descriptors1,
                  keypoints2, descriptors2, labels1, clusters1, reprojections,  cp, flag, count, MP,
                  temp, queryIdxs, trainIdxs);

        //	cout << "tdescr2size \t " << tdescriptors.rows << endl;
        // (keypoints1, descriptors1): source window
        // (keypoints2, descriptors2): target window
        // The target ROI becomes the source ROI for next iteration
        //cout << "Center of MS windo = " << cp.x << "\t" << cp.y << endl;


        img1=img2.clone();

        msBox = box;  // Box obtained from meanShift()

        double q_msbox = similarityMeasure(descriptors2, tpDescriptors,
                                           descriptorMatcher, matcherFilterType);



         vector<Point2f> targetMatchPoints, targetAllPoints;
         KeyPoint::convert(keypoints2, targetMatchPoints, trainIdxs);
         KeyPoint::convert(keypoints2, targetAllPoints);
         shiftPoints(targetMatchPoints, box);
         shiftPoints(targetAllPoints, box);

//         for(int i=0;i< tdescriptors.rows;i++)
//         {
//             descriptors1.push_back(tdescriptors.row(i));
//             keypoints1.push_back(tkeypoints[i]);
//         }

        gettimeofday(&t2, NULL);
        double diff = (float)((t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec));
        diff = diff/1000;
        cout << k << "\tTime taken in mili sec \t" <<  diff<< endl;
        //f1 <<  k << "\t" << MP << "\t"   << count  << "\t"   << diff << "\n";



        //====================================
        // Particle Filter Update
        // Estimated location of tracker window in the next frame
        process(xloc, xloc, (void*)rg);

        //Actual location of tracker window obtained from Mean-Shift Tracking
        zpos[0] = cp.x;
        zpos[1] = cp.y;



#ifdef ZBIN
        // For every point in the source window, if there is a correspondence in
        // the target window, we assign zbin to 1 for that particle
        vector<uint> zdbin(Ns);
        for(uint j = 0; j < Ns; ++j)
        {
            if(j < initPtlocn.size()) // for the original set
            {
                bool FOUND = false;
                for(uint i = 0; i < queryIdxs.size(); ++i)
                {
                    if(j == queryIdxs[i])
                        FOUND = true;
                }
                if(FOUND)
                    zdbin[j] = 1;
                else
                    zdbin[j] = 0;
            }
            else // for all other points
            {
                zdbin[j] = 0;
            }
        }

        cloud.pfupdate1(process,observation2,likelihood2,zdbin,0); //don't resample

        uint neff = cloud.getEffectivePopulation();

        cout << " Nt = " << Nt << "\tNeff = " << neff << endl;

        if(neff < Nt)
        {
            std::vector<double> x(Nx);
            double tempw;
            for(uint i = 0; i < Ns; ++i)
            {
                if(i < targetAllPoints.size())
                {
                    x[0] = targetAllPoints[i].x;
                    x[1] = 0.0;
                    x[2] = targetAllPoints[i].y;
                    x[3] = 0.0;
                    tempw = pwt1/sumwt;
                    zbin[i] = 1;
                    cloud.setParticleState(i, x, tempw, zbin[i]);
                    x.clear();
                }
                else
                {
                    zbin[i] = 0;
                    tempw = pwt2/sumwt;
                    cloud.setParticleState(i, x, tempw, zbin[i]);
                }
            }
           // cloud.resample();
        }

#else

        //cloud.particleFilterUpdate(process, observation, likelihood, zpos, Nt);

        std::vector<std::vector<double> > zdata(targetAllPoints.size(), std::vector<double>(Nz));
        for(uint i = 0; i < targetAllPoints.size(); ++i)
        {
            zdata[i][0] = targetAllPoints[i].x;
            zdata[i][1] = targetAllPoints[i].y;
        }

        //cloud.particleFilterUpdate2(process, observation, likelihood3, zdata, Nt);
        cloud.particleFilterUpdate2(process, observation, likelihood3, zdata, 0); // do not resample


#endif

        //Output of the particle Filter
        std::vector<double>xfloc(Nx);
        cloud.filterOutput(xfloc);

#ifdef ARLOC
        // Output of the AR Predictor
        arPredictor4Location(ar_xhat, ar_x, ar_w);

        for(uint i = 0; i < Nz; ++i)
        {
            if(ar_xhat[i] > 1.0)
                ar_xhat[i] = 1.0;

            if(ar_xhat[i] < 0.0)
                ar_xhat[i] = 0.0;
        }


        //cout << "AR location = " << ar_xhat[0] << "\t" << ar_xhat[1] << endl;

        // Error in Window location
        ar_e[0] = (cp.x/640.0 - ar_xhat[0]);
        ar_e[1] = (cp.y/480.0 - ar_xhat[1]);
        ar_e[2] = (double)(box.height)/480.0 - ar_xhat[2];

        //cout << "AR Error = " << ar_xhat[0] << "\t" << ar_xhat[1] << endl;

        // Update the parameters of AR Model
        arPredictorUpdateGD(ar_w, ar_e, ar_x, ar_eta);

//        cout << "AR weights = " << endl;
//        for(uint j = 0; j < arlen; ++j)
//        {
//            for(uint i = 0; i < Nz; ++i)
//                cout << ar_w[i][j] << "\t";
//            cout << endl;
//        }

        //Update the AR States
        for(uint j = 0; j < arlen; ++j)
        {
            uint p = arlen-1-j;

            if(p > 0)
            {
                for(uint i = 0; i < arDim; ++i)
                    ar_x[p][i] = ar_x[p-1][i];
            }
            else
            {
                for(uint i = 0; i < arDim; ++i)
                    ar_x[p][i] = ar_xhat[i];
            }
        }
#endif


        Point2f Cpf, Car;
        Cpf.x = xfloc[0]; Cpf.y = xfloc[2];
        Car.x = ar_xhat[0] * 640.0; Car.y = ar_xhat[1] * 480.0;
        circle(temp,Cpf, 3, Scalar(255,255,255),8,0);
        //cout << "AR location = " << Car.x << "\t" << Car.y << endl;

        // Print the Particles on the image
        std::vector<double> tx(Nx);
        std::vector<double> tz(Nz);
        std::vector<double> tw(Ns);
        std::vector<Point2f> pfloc(Ns);

        // Read the particle states
        for(uint i = 0; i < Ns; ++i)
        {
            tw[i] = cloud.getParticleState(tx, tz, i);

            pfloc[i].x = tx[0];
            pfloc[i].y = tx[2];

            cv::circle(temp,pfloc[i], 2, Scalar(255,0,0), 2);

            f2 << tx[0] << "\t" << tx[2] << "\t" << tw[i] << endl;
        }
        f2 << endl << endl;

        Rect PfboundingBox = boundingRect(pfloc);
        rectangle(temp, PfboundingBox, Scalar(0,255,0),1,8,0);

       pfBox = PfboundingBox;  // Box obtained from Particle Filter


        boxOrg.height = PfboundingBox.height;
        //boxOrg.width = PfboundingBox.width;
        boxOrg.width = (int)(aspectRatio*boxOrg.height);
        boxOrg.x = PfboundingBox.x + PfboundingBox.width/2.0 - boxOrg.width/2.0;
        boxOrg.y = PfboundingBox.y + PfboundingBox.height/2.0 - boxOrg.height/2.0;

        keypoints1.clear();
        descriptors1.release();

        vector<KeyPoint> tempkey;
        Mat tempdesc;

        img1ROI = img2(PfboundingBox);
        mySURF.detect(img1ROI,tempkey);
        mySURF.compute(img1ROI,tempkey, tempdesc);

        double q_pfbox = similarityMeasure(tempdesc, tpDescriptors, descriptorMatcher, matcherFilterType);
        cout << "Similarity Measure = " << q_msbox << "\t" << q_pfbox << endl;

        if(q_msbox > q_pfbox)   // if mean-shift window is better, put the corresponding descriptors into
        {                       // the template pool
            box = msBox;
            for(uint i = 0; i < descriptors2.rows; ++i)
                tpDescriptors.push_back(descriptors2.row(i));

            descriptors2.copyTo(descriptors1);
            keypoints1 = keypoints2;
        }
        else  // if particle filter window is better, put the corresponding descriptors into the template pool
        {
            box = pfBox;
            for(uint i = 0; i < tempdesc.rows; ++i)
                tpDescriptors.push_back(tempdesc.row(i));

            tempdesc.copyTo(descriptors1);
            keypoints1 = tempkey;
        }

        //Initialize the source window
        boxOrg = box;


#ifdef PF_OPT

        //Compute Global point locations corresponding to tempkey obtained from PF
        vector<Point2f>  tempPoints;
        vector<float> tempwt(descriptors1.rows);
        Mat clabels;
        KeyPoint::convert(keypoints1, tempPoints);

        shiftPoints(tempPoints, box);
        searchBin(descriptors1,clusters1,clabels);
        weights4Descriptors(descriptors1, clabels, clusters1.rows, tempwt);

        cout << "Descriptors1.size = " << descriptors1.rows << "\t tempwt.size = " << tempwt.size() << endl;
        cout << "No. of Clusters = " << clusters1.rows << endl;

        // Debug
//        for(uint j = 0; j < clusters1.rows; ++j)
//        {
//            for(uint i = 0; i < clusters1.cols; ++i)
//                cout << clusters1.row(j).< "\t";
//            cout << endl;
//        }
//        cout << endl;
//        for(uint j = 0; j < tempwt.size(); ++j)
//            cout << tempwt[j] << "\t";
//        cout << endl;
//        getchar();

        double sum_wt2 = 1.0 + 0.1*(Ns - tempPoints.size()); // Normalization factor


        uint j = 0;
        while(j < Ns)
        {
            if(j < tempPoints.size())
            {
                xp[j][0] = tempPoints[j].x;
                xp[j][1] = tempPoints[j].y;
                wt[j] = tempwt[j]/sum_wt2;
                j++;
            }
            else
            {
                double a, b;
                bool tempFlag = false;
                do
                {
                    a = box.x + box.width/2.0 + gsl_ran_gaussian(rg, var_x2);
                    b = box.y + box.height/2.0 + gsl_ran_gaussian(rg, var_y2);

                    cout << "a = " << a << "\t b = " << b << endl;
                    cout << "box.x = " << box.x << "\t box.y = " << box.y << endl;
                    cout << "box.x2 = " << box.x+box.width << "\t box.y2 = " << box.y + box.height << endl;

                    if( (a > box.x) && (a < (box.x + box.width)) &&
                            (b > box.y) && (b < (box.y + box.height)) )
                    {
                        tempFlag = true;
                        cout << "tempFlag = " << tempFlag << endl;
                    }
                    else
                    {
                        tempFlag = false;
                    }

                }while(!tempFlag);

                xp[j][0] = a;
                xp[j][1] = b;
                wt[j] = 0.1/sum_wt2;
                j++;
            }

        }


        double neff = cloud.getEffectivePopulation();
        cout << "Neff = " << neff << "\t Nt = " << Nt << endl;

        if(neff < Nt) // Now resample
        {
            cout << "Reinitializing PF ..." << endl;
            cloud.initialize(k,wt,xp);
        }

#endif









//        Point2f tempCoord;
//        tempCoord.x = PfboundingBox.x;
//        tempCoord.y = PfboundingBox.y;

//        for(uint i = 0; i < pfloc.size(); ++i)
//        {
//            pfloc[i] -= tempCoord;
//        }

        //keypoints1.clear();
        //descriptors1.release();
        //KeyPoint::convert(pfloc, keypoints1,1,1,0,-1);
        //mySURF.compute(img1ROI, keypoints1, descriptors1);

        //cout << descriptors1 << endl;

//        boxOrg = PfboundingBox;
//        box = boxOrg;

        //rectangle(temp, box, Scalar(0,0,255),1,8,0);

        //shiftPoints(pfloc,box);
        //R1 = fitEllipse(pfloc);
        //ellipse(temp,R1, Scalar(255,255,255), 2, 8);


        cv::circle(temp, Cpf, 3, Scalar(0,255,0), 3);
        cv::circle(temp, cp, 2, Scalar(0,255,255), 2);
        cv::circle(temp,Car, 3, Scalar(0,0,255),8,0);
        //=======================================

        f1 << cp.x << "\t" << cp.y << "\t" << Cpf.x << "\t" << Cpf.y << endl;

        imshow("main", temp);
        //imshow("img2", img2);


        char c = (char)waitKey(10);
        if( c == '\x1b' ) // esc
        {
            cout << "Exiting ..." << endl;
            break;
        }




        waitKey(0);


    }
    f1.close();
   
    // cout << "desc1_size \t" <<descriptors1.rows << endl;

   // trajectory.close();

    return 0;
}








