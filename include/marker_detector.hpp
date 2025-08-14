#pragma once
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include "marker_types.hpp"

/**
 * MarkerDetector: end-to-end pipeline
 *  - color segmentation (HSV)
 *  - quad extraction
 *  - warp to square
 *  - grid seams check (3ª3)
 *  - polygon coverage
 */
class MarkerDetector {
public:
    MarkerDetector() = default;

    /**
     * Runs the detection on a BGR image.
     * Returns std::nullopt if no reasonable quad was found (marker not detected).
     */
    std::optional<DetectionResult>
        detect(const cv::Mat& bgr, const DetectOptions& opt,
            const std::string& image_path_hint = "") const;
};
