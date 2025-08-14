#pragma once
#include <opencv2/opencv.hpp>

/**
 * Builds a binary mask (CV_8UC1) of all pixels that fall into the allowed
 * marker colors (red, green, yellow, blue, magenta, cyan) in HSV space.
 * Returned mask: 255 for allowed pixels, 0 otherwise.
 */
class ColorSegmenter {
public:
    // Returns a cleaned binary mask of allowed colors in HSV
    static cv::Mat allowedMaskHSV(const cv::Mat& bgr);
};
