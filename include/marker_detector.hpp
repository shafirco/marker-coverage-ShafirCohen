#pragma once
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include "marker_types.hpp"

/// @brief End-to-end pipeline for detecting a 3×3 color marker.
/// Steps: HSV color segmentation → quad extraction → warp to square → grid validation → polygon coverage.
class MarkerDetector {
public:
    MarkerDetector() = default;

    /**
     * @brief Run marker detection on a BGR image.
     * @param bgr Input image (BGR).
     * @param opt Detection options (thresholds, debug flags, etc.).
     * @param image_path_hint Optional: file path for logging/debug.
     * @return DetectionResult on success, std::nullopt if no valid quad/marker found.
     */
    std::optional<DetectionResult>
        detect(const cv::Mat& bgr,
            const DetectOptions& opt,
            const std::string& image_path_hint = "") const;
};
