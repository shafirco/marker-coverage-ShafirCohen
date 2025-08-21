
// High-level detector for a 3×3 colored marker.
// Pipeline:
//   1) HSV color segmentation → binary mask of allowed colors.
//   2) Extract a strong quadrilateral (outer board boundary).
//   3) Warp quad to a square; validate 3×3 grid (debug/validation).
//   4) Final polygon = initial quad (no refinement).
//   5) Compute coverage: polygon area / image area.
//
// Notes:
// - In strict mode, grid validation (cells check) is required for success.
// - Seams are primarily diagnostic.
// - Debug artifacts are saved only if requested.

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
    /// @brief Save helper with optional debug logging. Returns true on success.
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

    /// @brief Draw a polygon overlay on a BGR image (for debug snapshots).
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

    /// @brief Derive a base filename (without extension) for debug artifacts.
    static std::string makeBaseName(const std::string& path_hint) {
        if (path_hint.empty()) return "image";
        fs::path p(path_hint);
        std::string stem = p.stem().string();
        return stem.empty() ? "image" : stem;
    }
}

std::optional<DetectionResult>
MarkerDetector::detect(const cv::Mat& bgr,
    const DetectOptions& opt,
    const std::string& image_path_hint) const
{
    // Input guard
    if (bgr.empty() || bgr.type() != CV_8UC3) return std::nullopt;

    const std::string base = makeBaseName(image_path_hint);
    const fs::path outdir = fs::path(opt.save_debug_dir);
    if (opt.save_debug && opt.debug) {
        std::cerr << "[debug] save dir: " << fs::absolute(outdir).string() << "\n";
    }

    // Timers (profiling in debug mode)
    Timer total;
    double t_seg = 0, t_quad = 0, t_warp = 0, t_grid = 0, t_refine = 0;

    // ---------------------------------------------------------------------
    // (1) HSV segmentation of allowed marker colors
    // ---------------------------------------------------------------------
    Timer t1;
    SegOptions sopt;
    sopt.blur_ksize = opt.pre_blur_ksize;   // optional pre-blur
    sopt.open_iter = opt.morph_open_iter;  // morphological cleanup
    sopt.close_iter = opt.morph_close_iter;
    sopt.smin = opt.seg_smin;         // global S floor
    sopt.vmin = opt.seg_vmin;         // global V floor

    cv::Mat mask = ColorSegmenter::allowedMaskHSV(bgr, sopt);
    t_seg = t1.ms();

    if (opt.debug) std::cerr << "[debug] mask nonzero=" << cv::countNonZero(mask) << "\n";
    saveIf(mask, outdir / (base + "_mask.png"), opt.save_debug, opt.debug);

    // ---------------------------------------------------------------------
    // (2) Extract a strong quadrilateral from the mask (outer board boundary)
    // ---------------------------------------------------------------------
    Timer t2;
    auto quadOpt = geom::findStrongQuad(mask);
    t_quad = t2.ms();

    if (!quadOpt) {
        if (opt.debug) std::cerr << "[debug] no quad found\n";
        return std::nullopt;
    }
    const auto& quad = *quadOpt;
    saveIf(drawPolyOverlay(bgr, quad), outdir / (base + "_poly.png"), opt.save_debug, opt.debug);

    // ---------------------------------------------------------------------
    // (3) Warp to square & compute warped mask (for grid validation)
    // ---------------------------------------------------------------------
    Timer t3;
    const int N = std::max(32, opt.warp_size);

    // Warp the original image to a square using the quad.
    auto warpRes = geom::warpToSquareWithH(bgr, quad, N);
    cv::Mat warped = warpRes.image;

    // Build a warped mask using the same segmentation options,
    // but allow a one-time, local relaxation if the mask is too sparse.
    // NOTE: this does NOT change the global mask in (1)—it only
    // adapts the warped-view where blur/low saturation could hide colors.
    SegOptions sopt_warp = sopt;
    cv::Mat warpedMask = ColorSegmenter::allowedMaskHSV(warped, sopt_warp);

    // One-shot local relaxation on warped-mask only (helps blurred/low-S cases).
    {
        const double totalW = std::max(1.0, (double)warped.total());
        const double frac = (double)cv::countNonZero(warpedMask) / totalW;

        // If the warped mask is very sparse (<3%), relax S/V a bit and try once more.
        if (frac < 0.03) {
            sopt_warp.smin = std::max(0, sopt_warp.smin - 20);
            sopt_warp.vmin = std::max(0, sopt_warp.vmin - 20);
            // A slightly stronger close helps reconnect split color blobs.
            sopt_warp.close_iter = std::max(1, sopt_warp.close_iter);
            warpedMask = ColorSegmenter::allowedMaskHSV(warped, sopt_warp);
        }
    }

    t_warp = t3.ms();

    saveIf(warped, outdir / (base + "_warped.png"), opt.save_debug, opt.debug);
    saveIf(warpedMask, outdir / (base + "_warped_mask.png"), opt.save_debug, opt.debug);

    // ---------------------------------------------------------------------
    // (4) Grid validation: seams (diagnostics) + cells (decision)
    //     - Strict mode requires cells.ok == true.
    //     - Added a color/saturation fallback: if ≥7 of 9 cells are "colorful enough",
    //       we accept the grid (helps blurred low-mask cases like p5).
    // ---------------------------------------------------------------------

    Timer t4;
    auto seams = grid::checkGridSeams(warpedMask);
    auto cells = grid::checkGridCells(warpedMask, opt.min_cell_fraction);

    // Fallback: consider cells "colorful" if their mean S and V are high enough.
    // This check runs on the warped BGR image (not on the mask), so it can still pass
    // when the warped mask is under-segmented but colors are visibly present.
    auto colorful_cells_ge7 = [&]()->bool {
        CV_Assert(warped.type() == CV_8UC3 && warped.rows == warped.cols);
        const int Nw = warped.rows;
        const int cell = Nw / 3;

        cv::Mat hsv; cv::cvtColor(warped, hsv, cv::COLOR_BGR2HSV);

        int okCells = 0;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                const int x = c * cell;
                const int y = r * cell;
                const int w = (c == 2 ? Nw - x : cell);
                const int h = (r == 2 ? Nw - y : cell);
                cv::Mat roi = hsv(cv::Rect(x, y, w, h));

                // Average saturation and value in the cell.
                const cv::Scalar meanHSV = cv::mean(roi);
                const double meanS = meanHSV[1];
                const double meanV = meanHSV[2];

                // Soft thresholds tuned for blurred, low-contrast markers.
                if (meanS >= 70.0 && meanV >= 60.0) {
                    okCells++;
                }
            }
        }
        return okCells >= 7; // accept if at least 7 of 9 look "colorful"
    }();

    t_grid = t4.ms();

    if (opt.debug) {
        std::cerr << "[debug] seams: cx1=" << seams.cx1 << ", cx2=" << seams.cx2
            << ", cy1=" << seams.cy1 << ", cy2=" << seams.cy2
            << ", ok=" << seams.ok << "\n";
        std::cerr << "[debug] cells ok=" << (cells.ok ? 1 : 0)
            << " (min=" << opt.min_cell_fraction << ")\n";
        std::cerr << "[debug] colorful>=7=" << (colorful_cells_ge7 ? "true" : "false") << "\n";
    }

    // Decision: in strict mode require BOTH seams and cells.
  // In non-strict mode, allow the colorful fallback as a helper.
    const bool grid_ok =
        (opt.strict_grid)
        ? (seams.ok && cells.ok)
        : (cells.ok || colorful_cells_ge7);


    // ---------------------------------------------------------------------
    // (5) Final polygon = initial quad (no refinement)
    // ---------------------------------------------------------------------
    Timer t5;
    std::vector<cv::Point2f> final_poly = quad;
    // Kept for parity with prior runs; same content as _poly.png.
    saveIf(drawPolyOverlay(bgr, final_poly), outdir / (base + "_poly_refined.png"), opt.save_debug, opt.debug);
    t_refine = t5.ms(); // near-zero; included for timing symmetry

    // ---------------------------------------------------------------------
    // (6) Coverage computation
    // ---------------------------------------------------------------------
    double cov = geom::polygonCoveragePercent(final_poly, bgr.size());
    // Reject unrealistically tiny polygons (prevents 0% false positives).
    const double kMinCovPct = 0.5; // percent
    if (cov < kMinCovPct) {
        if (opt.debug) std::cerr << "[debug] coverage guard failed (" << cov << "%)\n";
        return std::nullopt;
    }

    // Timing summary
    if (opt.debug) {
        double t_total = total.ms();
        std::cerr << "[time] seg=" << t_seg << " ms, "
            << "quad=" << t_quad << " ms, "
            << "warp=" << t_warp << " ms, "
            << "grid=" << t_grid << " ms, "
            << "refine=" << t_refine << " ms, "
            << "total=" << t_total << " ms\n";
    }

    // Package result
    DetectionResult res;
    res.polygon = final_poly;
    res.coverage_percent = cov;
    res.grid_ok = grid_ok;

    // Strict mode: fail if grid validation did not pass.
    if (opt.strict_grid && !grid_ok) {
        if (opt.debug) std::cerr << "[debug] strict_grid=true -> not found\n";
        return std::nullopt;
    }
    return res;
}
