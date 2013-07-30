#ifndef NEWFUNCTIONS_H
#define NEWFUNCTIONS_H


double similarityMeasure(
        Mat &bDescriptors, Mat &tpDescriptors,
        Ptr<DescriptorMatcher>& descriptorMatcher, int matcherFilter)
{
    vector<DMatch> filteredMatches;

    switch( matcherFilter )
    {
    case CROSS_CHECK_FILTER :
        crossCheckMatching( descriptorMatcher, bDescriptors, tpDescriptors, filteredMatches, 1 );
        break;
    default:
        simpleMatching( descriptorMatcher, bDescriptors, tpDescriptors, filteredMatches );
    }

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
        mDesc2.push_back(tpdescriptors.row(trainIdxs[i]));

    }


    int MP=0;;
    double disSum=0.0;
    for( size_t i = 0; i < mDesc1.rows; i++ )
    {
        double descDiff = pow(norm(mDesc1.row(i) - mDesc2.row(i)), 2);
        if(descDiff < 0.06)
        {
            MP++;
            disSum += descDiff;
        }
    }
    disSum = disSum / MP;

    double percMatch = (double) MP / bDescriptors.rows * exp(-1.0 * disSum);


    mDesc1.release();
    mDesc2.release();
    queryIdxs.clear();
    trainIdxs.clear();
    filteredMatches.clear();

    return percMatch;
}




#endif // NEWFUNCTIONS_H
