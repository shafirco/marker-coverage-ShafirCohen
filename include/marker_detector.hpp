/**
 * @file marker_detector.hpp
 * @brief Main detection pipeline for 3x3 color grid markers
 * 
 * MarkerDetector implements a complete computer vision pipeline for detecting
 * colored 3x3 grid markers in images. The algorithm is designed to be robust
 * against varying lighting conditions, perspective distortion, and background clutter.
 */
#pragma once
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include "marker_types.hpp"

/**
 * @brief Complete pipeline for detecting 3×3 color grid markers
 * 
 * The detection process consists of 6 main stages:
 * 1. HSV color segmentation (6 supported colors)
 * 2. Quadrilateral boundary extraction  
 * 3. Perspective correction to square
 * 4. Grid structure validation (seams + cells)
 * 5. Coverage calculation
 * 6. Result validation and output
 * 
 * @note This class is thread-safe and stateless - the same instance
 *       can be used to process multiple images concurrently.
 * 
 * @example
 * ```cpp
 * MarkerDetector detector;
 * DetectOptions opts;
 * opts.debug = true;
 * opts.strict_grid = false;  // More permissive
 * 
 * cv::Mat image = cv::imread("marker.jpg");
 * auto result = detector.detect(image, opts, "marker.jpg");
 * if (result) {
 *     std::cout << "Coverage: " << result->coverage_percent << "%\n";
 * }
 * ```
 */
class MarkerDetector {
public:
    /**
     * @brief Default constructor - creates a stateless detector instance
     */
    MarkerDetector() = default;

    /**
     * @brief Detect a 3x3 color marker in a BGR image
     * 
     * @param bgr Input image in BGR color format (CV_8UC3)
     * @param opt Detection and validation options
     * @param image_path_hint Optional filename for debug logging and output naming
     * 
     * @return DetectionResult if marker found and validated, std::nullopt otherwise
     * 
     * @note In strict mode, requires both grid seams AND (cell validation OR colorful fallback)
     * @note In non-strict mode, requires (cell validation OR colorful fallback) only
     * 
     * @warning Input image must be non-empty BGR format, otherwise returns nullopt
     * @warning Markers covering <0.5% of image area are rejected as false positives
     */
    std::optional<DetectionResult>
        detect(const cv::Mat& bgr,
            const DetectOptions& opt,
            const std::string& image_path_hint = "") const;
};
