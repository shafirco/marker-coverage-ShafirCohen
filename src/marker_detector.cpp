#include "marker_detector.hpp"
#include "color_segmenter.hpp"
#include "geometry.hpp"
#include "grid_detector.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>

using namespace cv;
namespace fs = std::filesystem;

namespace {
    // Save helper that tolerates failures and only runs if enabled.
    static void saveIf(const cv::Mat& img, const fs::path& path, bool enabled) {
        if (!enabled) return;
        try {
            fs::create_directories(path.parent_path());
            cv::imwrite(path.string(), img);
        }
        catch (...) {
            // silently ignore debug-save errors
        }
    }

    // Draw polygon overlay on top of the original image (for debug)
    static cv::Mat drawPolyOverlay(const cv::Mat& bgr, const std::vector<cv::Point2f>& poly) {
        cv::Mat vis = bgr.clone();
        if (poly.size() >= 2) {
            for (size_t i = 0; i < poly.size(); ++i) {
                cv::Point p0 = poly[i], p1 = poly[(i + 1) % poly.size()];
                cv::line(vis, p0, p1, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                cv::circle(vis, p0, 3, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
            }
        }
        return vis;
    }

    // Create a stable base file name for debug artifacts from the input image path
    static std::string makeBaseName(const std::string& path_hint) {
        if (path_hint.empty()) return "image";
        fs::path p(path_hint);
        std::string stem = p.stem().string();
        if (stem.empty()) return "image";
        return stem;
    }
}

std::optional<DetectionResult>
MarkerDetector::detect(const cv::Mat& bgr, const DetectOptions& opt,
    const std::string& image_path_hint) const
{
    if (bgr.empty() || bgr.type() != CV_8UC3) return std::nullopt;

    const std::string base = makeBaseName(image_path_hint);
    const fs::path outdir = fs::path(opt.save_debug_dir);

    // 1) Allowed-color mask (HSV)
    cv::Mat mask = ColorSegmenter::allowedMaskHSV(bgr);
    if (opt.debug) std::cerr << "[debug] mask nonzero=" << cv::countNonZero(mask) << "\n";
    saveIf(mask, outdir / (base + "_mask.png"), opt.save_debug);

    // 2) Quad extraction
    auto quadOpt = geom::findStrongQuad(mask);
    if (!quadOpt) {
        if (opt.debug) std::cerr << "[debug] no quad found\n";
        return std::nullopt;
    }
    const auto& quad = *quadOpt;
    saveIf(drawPolyOverlay(bgr, quad), outdir / (base + "_poly.png"), opt.save_debug);

    // 3) Warp for grid validation
    const int N = std::max(32, opt.warp_size);
    cv::Mat warped = geom::warpToSquare(bgr, quad, N);
    cv::Mat warpedMask = ColorSegmenter::allowedMaskHSV(warped);
    saveIf(warped, outdir / (base + "_warped.png"), opt.save_debug);
    saveIf(warpedMask, outdir / (base + "_warped_mask.png"), opt.save_debug);

    // 4) Grid validation via seams (3ª3)
    auto seams = grid::checkGridSeams(warpedMask);
    if (opt.debug) {
        std::cerr << "[debug] seams: cx1=" << seams.cx1 << ", cx2=" << seams.cx2
            << ", cy1=" << seams.cy1 << ", cy2=" << seams.cy2
            << ", ok=" << seams.ok << "\n";
    }

    // 5) Coverage
    double cov = geom::polygonCoveragePercent(quad, bgr.size());

    DetectionResult res;
    res.polygon = quad;
    res.coverage_percent = cov;
    res.grid_ok = seams.ok;
    return res;
}
