#include <iostream>
#include <opencv2/opencv.hpp>
#include "color_segmenter.hpp"

// Prints usage instructions for the CLI application
static void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [--debug] <image1> [image2 ...]\n";
}

int main(int argc, char** argv) {
    // Print OpenCV version to verify linking
    std::cout << "OpenCV version: " << CV_VERSION << "\n";

    // If no image paths are provided, print usage and exit with code 2
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    // For now, process only the first image just to validate the color segmentation pipeline.
    const char* path = argv[1];
    cv::Mat bgr = cv::imread(path);
    if (bgr.empty()) {
        std::cerr << "Failed to load image: " << path << "\n";
        return 1;
    }

    // Build allowed-color mask in HSV and print some quick stats
    cv::Mat mask = ColorSegmenter::allowedMaskHSV(bgr);
    int nz = cv::countNonZero(mask);
    std::cout << "Allowed-color mask nonzero pixels: " << nz << "\n";

    // Placeholder: in later steps we will proceed to geometry/quad/coverage.
    std::cout << "Color segmentation OK.\n";
    return 0;
}
