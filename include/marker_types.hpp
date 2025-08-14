#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <optional>

/**
 * Holds the detection outcome for a single image.
 */
struct DetectionResult {
    std::vector<cv::Point2f> polygon;  // bounding polygon in original image coords
    double coverage_percent = 0.0;      // 0..100
    bool grid_ok = false;               // true if grid validation passed (seams for now)
};

/**
 * Controls runtime and debug behavior for the detector.
 */
struct DetectOptions {
    bool debug = false;                 // print debug logs to stderr
    bool save_debug = false;            // save debug images (masks/warps/overlay)
    std::string save_debug_dir = "out"; // folder for debug artifacts
    int warp_size = 300;                // warped square resolution
};
