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
// only 8-bit
CV_EXPORTS void EdgesSubPix(cv::Mat &gray, double alpha, int low, int high,
                            std::vector<Contour> &contours, cv::OutputArray hierarchy,
                            int mode);

CV_EXPORTS void EdgesSubPix(cv::Mat &gray, double alpha, int low, int high, 
                           std::vector<Contour> &contours);

#endif // __EDGES_SUBPIX_H__
