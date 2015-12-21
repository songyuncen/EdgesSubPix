#include <opencv2/opencv.hpp>
#include <iostream>
#include "EdgesSubPix.h"
using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    const String keys =
                        "{help h usage ? |      | print this message            }"
                        "{@image         |      | image for edge detection      }"
                        "{@output        |      | image for draw contours       }"
                        "{data           |p.txt | edges data in txt format      }"
                        "{low            |40    | low threshold                 }"
                        "{high           |100   | high threshold                }"
                        "{mode           |1     | same as cv::findContours      }"
                        "{alpha          |1.0   | gaussian alpha              }";
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

    if (!parser.has("image"))
    {
        parser.printMessage();
        return 0;
    }

    if (!parser.has("output"))
    {
        parser.printMessage();
        return 0;
    }

    String imageFile = parser.get<String>(0);
    String outputFile = parser.get<String>(1);
    int low = parser.get<int>("low");
    int high = parser.get<int>("high");
    double alpha = parser.get<double>("alpha");
    int mode = parser.get<int>("mode");

    Mat image = imread(imageFile, IMREAD_GRAYSCALE);
    vector<Contour> contours;
    vector<Vec4i> hierarchy;
    EdgesSubPix(image, alpha, low, high, contours, hierarchy, mode);

    return 0;
}
