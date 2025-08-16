#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "marker_detector.hpp"
#include "marker_types.hpp"

/// @brief Print CLI usage.
static void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
        << " [--debug]"
        << " [--save-debug <dir>]"
        << " [--mode strict|loose]"
        << " [--grid-threshold <0..1>]"
        << " <image1> [image2 ...]\n";
}

int main(int argc, char** argv) {
    // Quieter OpenCV logs; avoid thread noise.
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    cv::setNumThreads(1); // optional

    bool debug = false;
    DetectOptions opt;                 // defaults: strict_grid=true, min_cell_fraction=0.20, warp_size=300
    std::vector<std::string> paths;

    // --- Parse arguments ---
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
        else if (s == "--mode") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after --mode (strict|loose)\n";
                return 2;
            }
            std::string mode = argv[++i];
            if (mode == "strict")      opt.strict_grid = true;
            else if (mode == "loose")  opt.strict_grid = false;
            else {
                std::cerr << "Invalid --mode. Use strict|loose\n";
                return 2;
            }
        }
        else if (s == "--grid-threshold") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after --grid-threshold\n";
                return 2;
            }
            opt.min_cell_fraction = std::stod(argv[++i]);
            if (opt.min_cell_fraction < 0.0 || opt.min_cell_fraction > 1.0) {
                std::cerr << "--grid-threshold must be in [0,1]\n";
                return 2;
            }
        }
        else {
            // Treat as image path.
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

    // --- Process images ---
    for (const auto& path : paths) {
        cv::Mat bgr = cv::imread(path);
        if (bgr.empty()) {
            if (debug) std::cerr << "[debug] failed to load: " << path << "\n";
            exit_code = 1;
            continue; // no output line for this image
        }

        auto resOpt = detector.detect(bgr, opt, path);
        if (!resOpt) {
            // In strict mode, failing grid validation or no quad → "not found".
            // Emit a minimal warning (stderr); do not print a result line.
            if (!debug) std::cerr << "[warn] no marker detected (strict mode): " << path << "\n";
            exit_code = 1;
            continue;
        }

        // Required output format to stdout: "<image_file> <coverage_percent>%"
        int rounded = (int)std::lround(resOpt->coverage_percent);
        std::cout << path << " " << rounded << "%\n";
    }

    return exit_code;
}
