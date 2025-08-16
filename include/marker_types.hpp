#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <optional>

/// @brief Detection output for a single marker.
struct DetectionResult {
    std::vector<cv::Point2f> polygon;   ///< Bounding polygon in original image coordinates
    double coverage_percent = 0.0;      ///< Marker area relative to image (0–100)
    bool grid_ok = false;               ///< True if seams & cells validated (strict mode)
};

/// @brief Options to control marker detection and validation.
struct DetectOptions {
    bool debug = false;                 ///< Print debug logs to stderr
    bool save_debug = false;            ///< Save debug images (masks, warps, overlays)
    std::string save_debug_dir = "out"; ///< Directory for debug artifacts
    int warp_size = 320;                ///< Resolution of warped square (N×N)

    // Grid validation
    bool strict_grid = true;            ///< If true, require both seams & cells → else return "not found"
    double min_cell_fraction = 0.15;    ///< Minimum allowed-color fraction per cell (0–1)

    // Performance / accuracy
    int   max_side = 1024;              ///< Resize so max(width,height) ≤ max_side (0 = disable)
    int   pre_blur_ksize = 3;           ///< Gaussian blur kernel (odd ≥3, 0 = disable)
    int   morph_open_iter = 1;          ///< Morphological open iterations
    int   morph_close_iter = 2;         ///< Morphological close iterations
    int   seg_smin = 80;                ///< Global lower bound for S channel
    int   seg_vmin = 65;                ///< Global lower bound for V channel
};
