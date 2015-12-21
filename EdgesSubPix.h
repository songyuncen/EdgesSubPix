#ifndef __EDGES_SUBPIX_H__
#define __EDGES_SUBPIX_H__
#include <opencv2/opencv.hpp>
#include <vector>

struct Contour
{
    std::vector<cv::Point2f> points;
    std::vector<float> direction;
    std::vector<float> response;
};

CV_EXPORTS void EdgesSubPix(cv::Mat &gray, double alpha, int low, int high, 
                           std::vector<Contour> &contours, cv::OutputArray hierarchy,
                           int mode, cv::Point2f offset = cv::Point2f());

CV_EXPORTS void EdgesSubPix(cv::Mat &gray, double alpha, int low, int high, 
                           std::vector<Contour> &contours, cv::Point2f offset = cv::Point2f());

#endif // __EDGES_SUBPIX_H__
