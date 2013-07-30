/**
    helper.cpp
    Purpose: all the helper utilites found here
    // Derived from opencv sample feature dicriptor extractor and matcher
    @author Mayank Jain ( mailTo: mayank10.j@tcs.com )
    @version 0.1 16/10/2012
*/


#include "helper_MST.h"


void exch(vector<double>& a, vector<int>& index, int i, int j) ;
void quicksort(vector<double>& a, vector<int>& index, int left, int right) ;
int partition(vector<double>& a, vector<int>& index, int left, int right) ;
bool isLess(double x, double y) ;
inline bool isMore(double x, double y) ;

// Indexed sort
void quicksort(vector<double>& main, vector<int>& index) {
    quicksort(main, index, 0, index.size() - 1);
}

// quicksort a[left] to a[right]
void quicksort(vector<double>& a, vector<int>& index, int left, int right) {
    if (right <= left) return;
    int i = partition(a, index, left, right);
    quicksort(a, index, left, i-1);
    quicksort(a, index, i+1, right);
}

// partition a[left] to a[right], assumes left < right
int partition(vector<double>& a, vector<int>& index, int left, int right)
{
    int i = left - 1;
    int j = right;
    while (true) {
        while (isLess(a[++i], a[right]))      // find item on left to swap
            ;                               // a[right] acts as sentinel
        while (isLess(a[right], a[--j]))      // find item on right to swap
            if (j == left) break;           // don't go out-of-bounds
        if (i >= j) break;                  // check if pointers cross
        exch(a, index, i, j);               // swap two elements into place
    }
    exch(a, index, i, right);               // swap with partition element
    return i;
}

// is x < y ?
inline bool isLess(double x, double y) {
    return (x < y);
}

inline bool isMore(double x, double y) {
    return (x > y);
}


// exchange a[i] and a[j]
void exch(vector<double>& a, vector<int>& index, int i, int j) {
    double swap = a[i];
    a[i] = a[j];
    a[j] = swap;
    int b = index[i];
    index[i] = index[j];
    index[j] = b;
}



void appendMatrix( cv::Mat &originalMat,const cv::Mat& matToBeAppended )
{


    if(! originalMat.empty() )
    {
        assert( originalMat.cols == matToBeAppended.cols ) ;
        assert( originalMat.type() == matToBeAppended.type() ) ;


        cv::Mat newTemp( originalMat.rows+ matToBeAppended.rows , matToBeAppended.cols,matToBeAppended.type() ) ;

        int i ;
        for( i=0;i< originalMat.rows ; i++)
        {
            Mat rowI = newTemp.row(i) ;
            //originalMat.row(i).copyTo( newTemp.row(i) );
            originalMat.row(i).copyTo( rowI );

        }
        for(int j=0; j< matToBeAppended.rows ; j++ )
        {
            Mat rowJpluI =  newTemp.row( j+i ) ;
            //matToBeAppended.row(j).copyTo( newTemp.row( j+i ) );
            matToBeAppended.row(j).copyTo( rowJpluI );
        }

        originalMat = newTemp ;

    }
    else
    {
        originalMat = matToBeAppended ;
    }
    return ;

}


// Append matrix horizontally
void appendMatrixHorz( cv::Mat &originalMat,const cv::Mat& matToBeAppended )
{


    if(! originalMat.empty() )
    {
        assert( originalMat.rows == matToBeAppended.rows ) ;
        assert( originalMat.type() == matToBeAppended.type() ) ;


        cv::Mat newTemp( originalMat.rows ,originalMat.cols + matToBeAppended.cols,matToBeAppended.type() ) ;

        int i ;
        for( i=0;i< originalMat.cols ; i++)
        {
            Mat colI = newTemp.col(i) ;
            //originalMat.row(i).copyTo( newTemp.row(i) );
            originalMat.col(i).copyTo( colI );

        }
        for(int j=0; j< matToBeAppended.cols ; j++ )
        {
            Mat colJpluI =  newTemp.col( j+i ) ;
            //matToBeAppended.row(j).copyTo( newTemp.row( j+i ) );
            matToBeAppended.col(j).copyTo( colJpluI );
        }

        originalMat = newTemp ;

    }
    else
    {
        originalMat = matToBeAppended ;
    }
    return ;

}




void printVector(vector<double>& vec)
{
    for(int l=0; l< vec.size() ; l++ )
    {
        cout << vec[l] << "\t" ;
    }
    cout << endl ;


}




void printVector(vector<int>& vec)
{
    for(int l=0; l< vec.size() ; l++ )
    {
        cout << vec[l] << "\t" ;
    }
    cout << endl ;


}



void printVector(vector<float>& vec)
{
    for(int l=0; l< vec.size() ; l++ )
    {
        cout << vec[l] << "\t" ;
    }
    cout << endl ;


}



// Function for normalizing <double> histogram

void normalizeHistogram(vector<double>& hist)
{
    double sumHist = 0.0 ;
    for( int i=0;i< hist.size() ;++i)
    {
        sumHist += hist[i] ;
    }

    double oneBySumHist = 1.0 / sumHist ;
    for( int i=0 ; i< hist.size() ; ++i)
    {
        hist[i] = hist[i] * oneBySumHist  ;
    }

}




int max_element_index ( vector<double>& data )
{
    int largest = 0 ;
    // if (first==last) return last;
    for( int i=0;i<data.size() ; i++)
        if (data[largest]< data[i])    // or: if (comp(*largest,*first)) for the comp version
            largest= i ;


    return largest;
}



int min_element_index ( vector<double>& data )
{
    int smallest = 0 ;
    // if (first==last) return last;
    for( int i=0;i<data.size() ; i++)
        if (data[smallest] > data[i])    // or: if (comp(*largest,*first)) for the comp version
            smallest= i ;


    return smallest;
}
