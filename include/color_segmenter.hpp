#pragma once
#include <opencv2/opencv.hpp>


/**
 * Tunable options for HSV segmentation.
 * blur_ksize: Gaussian blur kernel (odd, >=3). 0 disables blur.
 * open_iter / close_iter: morphological cleanup.
 * smin / vmin: global minimums for S and V (clamped with per-color ranges).
 */
struct SegOptions {
    int blur_ksize = 3;
    int open_iter = 1;
    int close_iter = 2;
    int smin = 90;
    int vmin = 80;
};

/**
 * Builds a binary mask (CV_8UC1) of all pixels that fall into the allowed
 * marker colors (red, green, yellow, blue, magenta, cyan) in HSV space.
 * Returned mask: 255 for allowed pixels, 0 otherwise.
 */
class ColorSegmenter {
public:
    // Returns a cleaned binary mask of allowed colors in HSV
    static cv::Mat allowedMaskHSV(const cv::Mat& bgr);

    // Tunable version with SegOptions overrides
    static cv::Mat allowedMaskHSV(const cv::Mat& bgr, const SegOptions& opt);

};
