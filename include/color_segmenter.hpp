/**
 * @file color_segmenter.hpp
 * @brief HSV-based color segmentation for marker detection
 * 
 * Implements robust color detection for 6 marker colors using HSV color space
 * with adaptive thresholding and morphological post-processing.
 */
#pragma once
#include <opencv2/opencv.hpp>

/**
 * @brief Configuration options for HSV color segmentation
 * 
 * Controls preprocessing, color thresholds, and morphological cleanup.
 * Default values are optimized for typical indoor lighting conditions.
 */
struct SegOptions {
    /// @brief Gaussian blur kernel size for noise reduction (must be odd ≥3, 0 = disable)
    int blur_ksize = 3;
    
    /// @brief Morphological opening iterations (removes small noise blobs)
    int open_iter = 0;
    
    /// @brief Morphological closing iterations (fills small gaps in color regions)
    int close_iter = 2;
    
    /// @brief Global minimum saturation threshold (0-255, higher = more selective)
    int smin = 90;
    
    /// @brief Global minimum value/brightness threshold (0-255, higher = reject darker colors)
    int vmin = 80;
};

/**
 * @brief HSV-based color segmentation for 3x3 marker detection
 * 
 * Detects 6 specific marker colors in HSV color space:
 * - Red (H: 0-10° ∪ 170-180°)  
 * - Green (H: 40-85°)
 * - Yellow (H: 20-35°)
 * - Blue (H: 90-130°)
 * - Magenta (H: 135-165°)
 * - Cyan (H: 85-100°)
 * 
 * @note Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) on V channel
 * @note Includes "White Rim Booster" to remove blurry white borders
 * @note Automatically relaxes thresholds if mask is extremely sparse (<0.1%)
 */
class ColorSegmenter {
public:
    /**
     * @brief Create binary mask of allowed marker colors using default settings
     * 
     * @param bgr Input BGR image (CV_8UC3)
     * @return Binary mask (CV_8UC1): 255 = allowed color, 0 = background
     * 
     * @note Uses default SegOptions with moderate blur and morphological cleanup
     */
    static cv::Mat allowedMaskHSV(const cv::Mat& bgr);

    /**
     * @brief Create binary mask of allowed marker colors with custom options
     * 
     * @param bgr Input BGR image (CV_8UC3) 
     * @param opt Segmentation parameters and thresholds
     * @return Binary mask (CV_8UC1): 255 = allowed color, 0 = background
     * 
     * @warning Input image must be non-empty BGR format
     * @note Automatically applies CLAHE enhancement before color detection
     * @note May relax thresholds if resulting mask is too sparse
     */
    static cv::Mat allowedMaskHSV(const cv::Mat& bgr, const SegOptions& opt);
};
