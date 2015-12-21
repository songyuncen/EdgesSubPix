#include "EdgesSubPix.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void EdgesSubPix(Mat &gray, double alpha, double low, double high,
    vector<Contour> &contours, OutputArray hierarchy,
    int mode, Point2f offset)
{
}

void EdgesSubPix(Mat &gray, double alpha, double low, double high,
    vector<Contour> &contours, Point2f offset)
{
    vector<Vec4i> hierarchy;
    EdgesSubPix(gray, alpha, low, high, contours, hierarchy, RETR_LIST);
}
