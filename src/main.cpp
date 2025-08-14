#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "marker_detector.hpp"
#include "marker_types.hpp"

// Prints usage instructions for the CLI application
static void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [--debug] [--save-debug <dir>] <image1> [image2 ...]\n";
}

int main(int argc, char** argv) {
    // Silence OpenCV internal logs
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    cv::setNumThreads(1); // optional

    bool debug = false;
    DetectOptions opt;
    std::vector<std::string> paths;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--debug") {
            debug = true;
            opt.debug = true;
        }
        else if (s == "--save-debug") {
            if (i + 1 >= argc) {
                std::cerr << "Missing directory after --save-debug\n";
                return 2;
            }
            opt.save_debug = true;
            opt.save_debug_dir = argv[++i];
        }
        else {
            paths.push_back(std::move(s));
        }
    }
    if (paths.empty()) {
        print_usage(argv[0]);
        return 2;
    }
    if (debug) std::cerr << "[debug] OpenCV: " << CV_VERSION << "\n";

    MarkerDetector detector;
    int exit_code = 0;

    for (const auto& path : paths) {
        cv::Mat bgr = cv::imread(path);
        if (bgr.empty()) {
            if (debug) std::cerr << "[debug] failed to load: " << path << "\n";
            exit_code = 1;
            continue;
        }
        auto resOpt = detector.detect(bgr, opt, path);
        if (!resOpt) {
            exit_code = 1; // do not print a line for this image
            continue;
        }

        // Round to nearest integer and print as required: "<image_file> <percent>%"
        int rounded = (int)std::lround(resOpt->coverage_percent);
        std::cout << path << " " << rounded << "%\n";
    }
    return exit_code;
}
