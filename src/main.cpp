#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "color_segmenter.hpp"
#include "geometry.hpp"
#include "grid_detector.hpp"

// Prints usage instructions for the CLI application
static void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [--debug] <image1> [image2 ...]\n";
}

int main(int argc, char** argv) {
    // Silence OpenCV logs (INFO/DEBUG). Use ERROR or SILENT:
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    // Optional: avoid thread-backend noise
    cv::setNumThreads(1);

    // Parse args
    bool debug = false;
    std::vector<std::string> paths;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--debug") debug = true;
        else paths.push_back(std::move(s));
    }
    if (paths.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    int exit_code = 0;

    for (const auto& path : paths) {
        cv::Mat bgr = cv::imread(path);
        if (bgr.empty()) {
            if (debug) std::cerr << "[debug] failed to load: " << path << "\n";
            exit_code = 1;
            continue;
        }

        // 1) Allowed-color mask
        cv::Mat mask = ColorSegmenter::allowedMaskHSV(bgr);
        if (debug) std::cerr << "[debug] mask nonzero: " << cv::countNonZero(mask) << "\n";

        // 2) Quad detection
        auto quadOpt = geom::findStrongQuad(mask);
        if (!quadOpt) {
            if (debug) std::cerr << "[debug] no quad found -> marker not detected\n";
            exit_code = 1;
            continue; // do not print line for this image
        }
        const auto& quad = *quadOpt;

        // 3) Warp to normalized square and seam check (debug info)
        if (debug) {
            cv::Mat warped = geom::warpToSquare(bgr, quad, 300);
            cv::Mat wmask = ColorSegmenter::allowedMaskHSV(warped);
            auto seams = grid::checkGridSeams(wmask);
            std::cerr << "[debug] seams: "
                << "cx1=" << seams.cx1 << ", cx2=" << seams.cx2
                << ", cy1=" << seams.cy1 << ", cy2=" << seams.cy2
                << ", ok=" << seams.ok << "\n";
        }

        // 4) Coverage computation
        double cov = geom::polygonCoveragePercent(quad, bgr.size());
        int rounded = (int)std::lround(cov);

        // 5) Print in required format: "<image_file> <coverage_percent>%"
        std::cout << path << " " << rounded << "%\n";
    }

    return exit_code;
}
