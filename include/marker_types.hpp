/**
 * @file marker_types.hpp
 * @brief Core data structures for 3x3 color marker detection
 * 
 * This file defines the main types used throughout the marker detection pipeline:
 * - DetectionResult: Output of successful marker detection
 * - DetectOptions: Configuration parameters for detection behavior
 * 
 * Supported marker colors: red, green, yellow, blue, magenta, cyan
 */
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <optional>

/**
 * @brief Result of marker detection for a single image
 * 
 * Contains the detected marker polygon and associated metadata.
 * Only valid when MarkerDetector::detect() returns a value.
 */
struct DetectionResult {
    /// @brief Bounding polygon vertices in original image coordinates (typically 4 points, clockwise from top-left)
    std::vector<cv::Point2f> polygon;
    
    /// @brief Marker coverage as percentage of total image area (0.0-100.0)
    double coverage_percent = 0.0;
    
    /// @brief Whether grid validation passed (seams detection + cell validation)
    bool grid_ok = false;
};

/**
 * @brief Configuration options for marker detection pipeline
 * 
 * Controls detection sensitivity, validation strictness, and debug output.
 * Default values are tuned for typical indoor lighting conditions.
 */
struct DetectOptions {
    // === Debug and Output ===
    /// @brief Enable verbose debug logging to stderr
    bool debug = false;
    
    /// @brief Save intermediate debug images (mask, warped, poly overlay)
    bool save_debug = false;
    
    /// @brief Output directory for debug artifacts (created if non-existent)
    std::string save_debug_dir = "out";
    
    /// @brief Resolution of warped square image for grid validation (NxN pixels)
    /// @note Minimum 32, recommended 320+ for accuracy
    int warp_size = 320;

    // === Grid Validation ===
    /// @brief Strict mode: require both seams AND (cells OR colorful fallback)
    /// @note Non-strict mode: cells OR colorful fallback (seams not required)
    bool strict_grid = true;
    
    /// @brief Minimum fraction of allowed-color pixels per grid cell (0.0-1.0)
    /// @note Lower values increase sensitivity but may cause false positives
    double min_cell_fraction = 0.15;

    // === Performance and Preprocessing ===
    /// @brief Resize input so max(width,height) ≤ max_side (0 = disable)
    /// @note Improves performance for high-resolution images
    int max_side = 1024;
    
    /// @brief Gaussian blur kernel size for preprocessing (odd ≥3, 0 = disable)
    /// @note Helps with noisy images but may blur fine details
    int pre_blur_ksize = 3;
    
    /// @brief Morphological opening iterations (removes small noise)
    int morph_open_iter = 1;
    
    /// @brief Morphological closing iterations (fills small gaps)
    int morph_close_iter = 2;

    // === Color Segmentation ===
    /// @brief Global minimum saturation threshold (0-255)
    /// @note Higher values = more selective color detection
    int seg_smin = 80;
    
    /// @brief Global minimum value/brightness threshold (0-255)
    /// @note Higher values = reject darker colors
    int seg_vmin = 65;
};
