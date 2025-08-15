#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <optional>

struct DetectionResult {
    std::vector<cv::Point2f> polygon;  // bounding polygon in original image coords
    double coverage_percent = 0.0;      // 0..100
    bool grid_ok = false;               // seams && cells (when strict)
};

struct DetectOptions {
    bool debug = false;                 // print debug logs to stderr
    bool save_debug = false;            // save debug images (masks/warps/overlay)
    std::string save_debug_dir = "out"; // folder for debug artifacts
    int warp_size = 300;                // warped square resolution

    // grid validation tuning
    bool strict_grid = true;            // if true => require both seams & cells OK, else return "not found"
    double min_cell_fraction = 0.20;    // per-cell minimum fraction of allowed-color pixels (0..1)

    // Performance / accuracy tuning
    int   max_side = 1024;     // resize input so max(width,height) <= max_side (0 disables)
    int   pre_blur_ksize = 3;  // 0 disables Gaussian blur; use odd >=3
    int   morph_open_iter = 1;
    int   morph_close_iter = 2;
    int   seg_smin = 83;       // global lower-bound for S channel
    int   seg_vmin = 70;       // global lower-bound for V channel
};
