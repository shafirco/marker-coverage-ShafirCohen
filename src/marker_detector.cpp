#include "marker_detector.hpp"
#include "marker_types.hpp"
#include "color_segmenter.hpp"
#include "geometry.hpp"
#include "grid_detector.hpp"
#include "timer.hpp"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>

using namespace cv;
namespace fs = std::filesystem;

namespace {
    static bool saveIf(const cv::Mat& img,
        const fs::path& path,
        bool enabled,
        bool verbose)
    {
        if (!enabled) return false;
        try {
            fs::create_directories(path.parent_path());
            cv::imwrite(path.string(), img);
            if (verbose) std::cerr << "[debug] saved: " << path.string() << "\n";
            return true;
        }
        catch (const cv::Exception& e) {
            if (verbose) std::cerr << "[debug] save failed: " << path.string()
                << " | cv::Exception: " << e.what() << "\n";
        }
        catch (const std::exception& e) {
            if (verbose) std::cerr << "[debug] save failed: " << path.string()
                << " | std::exception: " << e.what() << "\n";
        }
        catch (...) {
            if (verbose) std::cerr << "[debug] save failed: " << path.string()
                << " | unknown error\n";
        }
        return false;
    }

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

    static std::string makeBaseName(const std::string& path_hint) {
        if (path_hint.empty()) return "image";
        fs::path p(path_hint);
        std::string stem = p.stem().string();
        return stem.empty() ? "image" : stem;
    }
}

std::optional<DetectionResult>
MarkerDetector::detect(const cv::Mat& bgr, const DetectOptions& opt,
    const std::string& image_path_hint) const
{
    if (bgr.empty() || bgr.type() != CV_8UC3) return std::nullopt;

    const std::string base = makeBaseName(image_path_hint);
    const fs::path outdir = fs::path(opt.save_debug_dir);
    if (opt.save_debug && opt.debug) {
        std::cerr << "[debug] save dir: " << fs::absolute(outdir).string() << "\n";
    }

    Timer total;
    double t_seg = 0, t_quad = 0, t_warp = 0, t_grid = 0;

    // 1) Allowed-color mask (HSV) with tuning options
    Timer t1;
    SegOptions sopt;
    sopt.blur_ksize = opt.pre_blur_ksize;
    sopt.open_iter = opt.morph_open_iter;
    sopt.close_iter = opt.morph_close_iter;
    sopt.smin = opt.seg_smin;
    sopt.vmin = opt.seg_vmin;

    cv::Mat mask = ColorSegmenter::allowedMaskHSV(bgr, sopt);
    t_seg = t1.ms();

    if (opt.debug) std::cerr << "[debug] mask nonzero=" << cv::countNonZero(mask) << "\n";
    saveIf(mask, outdir / (base + "_mask.png"), opt.save_debug, opt.debug);

    // 2) Quad extraction
    Timer t2;
    auto quadOpt = geom::findStrongQuad(mask);
    t_quad = t2.ms();

    if (!quadOpt) {
        if (opt.debug) std::cerr << "[debug] no quad found\n";
        return std::nullopt;
    }
    const auto& quad = *quadOpt;
    saveIf(drawPolyOverlay(bgr, quad), outdir / (base + "_poly.png"), opt.save_debug, opt.debug);

    // 3) Warp + warped mask
    Timer t3;
    const int N = std::max(32, opt.warp_size);
    cv::Mat warped = geom::warpToSquare(bgr, quad, N);
    cv::Mat warpedMask = ColorSegmenter::allowedMaskHSV(warped, sopt);
    t_warp = t3.ms();

    saveIf(warped, outdir / (base + "_warped.png"), opt.save_debug, opt.debug);
    saveIf(warpedMask, outdir / (base + "_warped_mask.png"), opt.save_debug, opt.debug);

    // 4) Grid validation (seams + cells)
    Timer t4;
    auto seams = grid::checkGridSeams(warpedMask);
    auto cells = grid::checkGridCells(warpedMask, opt.min_cell_fraction);
    t_grid = t4.ms();

    if (opt.debug) {
        std::cerr << "[debug] seams: cx1=" << seams.cx1 << ", cx2=" << seams.cx2
            << ", cy1=" << seams.cy1 << ", cy2=" << seams.cy2
            << ", ok=" << seams.ok << "\n";
        std::cerr << "[debug] cells ok=" << (cells.ok ? 1 : 0)
            << " (min=" << opt.min_cell_fraction << ")\n";
    }

    const bool grid_ok = seams.ok && cells.ok;

    // 5) Coverage (area ratio)
    double cov = geom::polygonCoveragePercent(quad, bgr.size());

    // Print timing summary under debug
    if (opt.debug) {
        double t_total = total.ms();
        std::cerr << "[time] seg=" << t_seg << " ms, "
            << "quad=" << t_quad << " ms, "
            << "warp=" << t_warp << " ms, "
            << "grid=" << t_grid << " ms, "
            << "total=" << t_total << " ms\n";
    }

    DetectionResult res;
    res.polygon = quad;
    res.coverage_percent = cov;
    res.grid_ok = grid_ok;

    if (opt.strict_grid && !grid_ok) {
        if (opt.debug) std::cerr << "[debug] strict_grid=true -> not found\n";
        return std::nullopt;
    }
    return res;
}
