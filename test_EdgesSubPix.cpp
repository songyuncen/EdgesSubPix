#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include "EdgesSubPix.h"
using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    ocl::setUseOpenCL(false);
    const String keys =
                        "{help h usage ? |          | print this message            }"
                        "{@image         |          | image for edge detection      }"
                        "{@output        |edge.tiff | image for draw contours       }"
                        "{data           |          | edges data in txt format      }"
                        "{low            |40        | low threshold                 }"
                        "{high           |100       | high threshold                }"
                        "{mode           |1         | same as cv::findContours      }"
                        "{alpha          |1.0       | gaussian alpha              }";
    CommandLineParser parser(argc, argv, keys);
    parser.about("subpixel edge detection");

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    if (!parser.has("@image"))
    {
        parser.printMessage();
        return 0;
    }

    String imageFile = parser.get<String>(0);
    String outputFile = parser.get<String>("@output");
    int low = parser.get<int>("low");
    int high = parser.get<int>("high");
    double alpha = parser.get<double>("alpha");
    int mode = parser.get<int>("mode");

    Mat image = imread(imageFile, IMREAD_GRAYSCALE);
    vector<Contour> contours;
    vector<Vec4i> hierarchy;
    int64 t0 = getCPUTickCount();
    EdgesSubPix(image, alpha, low, high, contours, hierarchy, mode);
    int64 t1 = getCPUTickCount();
    cout << "execution time is " << (t1 - t0) / (double)getTickFrequency() << " seconds" << endl;

    if (parser.has("data"))
    {
        FileStorage fs(parser.get<String>("data"), FileStorage::WRITE | FileStorage::FORMAT_YAML);
        fs << "contours" << "[";
        for (size_t i = 0; i < contours.size(); ++i)
        {
            fs << "{:";
            fs << "points" << contours[i].points;
            fs << "response" << contours[i].response;
            fs << "direction" << contours[i].direction;
            fs << "}";
        }
        fs << "]";
        fs.release();
    }

    return 0;
}
