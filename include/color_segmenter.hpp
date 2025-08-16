#pragma once
#include <opencv2/opencv.hpp>

/**
 * @brief Options for HSV color segmentation.
 * blur_ksize: odd ≥3 (0 = no blur). open/close: morphology cleanup.
 * smin/vmin clamp with per-color ranges.
 */
struct SegOptions {
    int blur_ksize = 3;
    int open_iter = 0;
    int close_iter = 2;
    int smin = 90;
    int vmin = 80;
};

/**
 * @brief Build a binary mask (CV_8UC1) of allowed marker colors (red/green/yellow/blue/magenta/cyan) in HSV.
 * 255 = allowed, 0 = otherwise.
 */
class ColorSegmenter {
public:
    /// @brief Mask with default options. @param bgr BGR input. @return CV_8UC1 mask.
    static cv::Mat allowedMaskHSV(const cv::Mat& bgr);

    /// @brief Mask with custom options. @param bgr BGR input. @param opt thresholds/cleanup. @return CV_8UC1 mask.
    static cv::Mat allowedMaskHSV(const cv::Mat& bgr, const SegOptions& opt);
};
