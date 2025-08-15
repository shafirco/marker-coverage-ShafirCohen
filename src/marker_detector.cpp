// marker_detector.cpp
//
// High-level detector that locates the 3x3 colored marker, computes a
// tight polygon around it and reports the relative area coverage.
// The pipeline is:
//   1) Segment “allowed colors” in HSV to produce a binary mask.
//   2) Extract a strong quadrilateral from the mask (outer board boundary).
//   3) Warp the quad to a square; validate a 3x3 grid exists (debug aid).
//   4) Refine the polygon INSIDE the quad to avoid border bleed/bridges.
//   5) Compute coverage percentage (polygon area / image area).
//
// Notes:
// - Grid validation is used for decision in strict mode (cells check),
//   while seams are primarily for diagnostics.
// - All optional debug artifacts are saved only when requested.

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
    // Save helper with optional debug logging. Returns true on success.
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

    // Visualize a polygon overlay on top of an image (for debug snapshots).
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

    // Create a reasonable base name from a path, used for debug file names.
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

    // Timers (for profiling in debug mode)
    Timer total;
    double t_seg = 0, t_quad = 0, t_warp = 0, t_grid = 0, t_refine = 0;

    // ---------------------------------------------------------------------
    // (1) HSV segmentation of allowed marker colors
    // ---------------------------------------------------------------------
    Timer t1;
    SegOptions sopt;
    sopt.blur_ksize = opt.pre_blur_ksize;      // optional pre-blur
    sopt.open_iter = opt.morph_open_iter;     // morphological cleanup
    sopt.close_iter = opt.morph_close_iter;
    sopt.smin = opt.seg_smin;            // global S lower bound
    sopt.vmin = opt.seg_vmin;            // global V lower bound

    cv::Mat mask = ColorSegmenter::allowedMaskHSV(bgr, sopt);
    t_seg = t1.ms();

    if (opt.debug) std::cerr << "[debug] mask nonzero=" << cv::countNonZero(mask) << "\n";
    saveIf(mask, outdir / (base + "_mask.png"), opt.save_debug, opt.debug);

    // ---------------------------------------------------------------------
    // (2) Extract a strong quadrilateral from the allowed-color mask
    //     (This is the outer boundary of the board in image space)
    // ---------------------------------------------------------------------
    Timer t2;
    auto quadOpt = geom::findStrongQuad(mask);
    t_quad = t2.ms();

    if (!quadOpt) {
        if (opt.debug) std::cerr << "[debug] no quad found\n";
        return std::nullopt; // no detection
    }
    const auto& quad = *quadOpt;
    saveIf(drawPolyOverlay(bgr, quad), outdir / (base + "_poly.png"), opt.save_debug, opt.debug);

    // ---------------------------------------------------------------------
    // (3) Warp the quad to a square & compute a warped mask (debug aid)
    //     We use the warped mask to validate the grid geometry (3x3).
    // ---------------------------------------------------------------------
    Timer t3;
    const int N = std::max(32, opt.warp_size);

    // קבל גם H/Hinv לשלב ה-Back-project
    auto warpRes = geom::warpToSquareWithH(bgr, quad, N);
    cv::Mat warped = warpRes.image;
    cv::Mat warpedMask = ColorSegmenter::allowedMaskHSV(warped, sopt);
    t_warp = t3.ms();

    saveIf(warped, outdir / (base + "_warped.png"), opt.save_debug, opt.debug);
    saveIf(warpedMask, outdir / (base + "_warped_mask.png"), opt.save_debug, opt.debug);

    // ---------------------------------------------------------------------
    // (4) Grid validation: seams (for diagnostics) + cells (for decision)
    //     - In strict mode, we require cells.ok == true.
    // ---------------------------------------------------------------------
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
    const bool grid_ok = cells.ok;

    // ---------------------------------------------------------------------
    // (5) Polygon refinement INSIDE the quad
    //
    // The goal here is to avoid accidental “bleed” into a bright/colored
    // background touching the board edge, or over-segmentation into sub-rects.
    // We:
    //   (a) rasterize the quad into a mask (image space),
    //   (b) keep only allowed pixels INSIDE that quad,
    //   (c) apply morphology to fill small gaps and break thin bridges,
    //   (d) take the largest component, convex-hull it, and approximate
    //       with a quad (fallback to min-area-rect for stability).
    // ---------------------------------------------------------------------
    // 5) REFINE polygon using tight box in warped space, then back-project
    Timer t5;

    // a) חשב פרקציה לכל עמודה/שורה (שומר על Mat "אמיתי" וניגש דרך ptr<>)
    const int W = warpedMask.cols, H = warpedMask.rows;
    cv::Mat fracPerCol = cv::Mat::zeros(1, W, CV_32F);
    cv::Mat fracPerRow = cv::Mat::zeros(H, 1, CV_32F);

    cv::Mat colsum32s, rowsum32s;
    cv::reduce(warpedMask, colsum32s, 0, cv::REDUCE_SUM, CV_32S); // 1 x W
    cv::reduce(warpedMask, rowsum32s, 1, cv::REDUCE_SUM, CV_32S); // H x 1

    const double colTotal = (double)H * 255.0;
    const double rowTotal = (double)W * 255.0;

    {
        const int* colptr = colsum32s.ptr<int>(0);
        float* fcol = fracPerCol.ptr<float>(0);
        for (int x = 0; x < W; ++x)
            fcol[x] = (float)((double)colptr[x] / colTotal);

        float* frow = fracPerRow.ptr<float>(0);
        for (int y = 0; y < H; ++y) {
            const int* rowptr = rowsum32s.ptr<int>(y);
            frow[y] = (float)((double)rowptr[0] / rowTotal);
        }
    }

    // b) מצא טווח הדוק (בלי structured bindings בשביל תאימות)
    const float colThresh = 0.10f, rowThresh = 0.10f;

    auto findTightRange = [](const cv::Mat& fracVec, float thr, bool isRowVec) -> std::pair<int, int> {
        int L = 0, R = (isRowVec ? fracVec.cols : fracVec.rows) - 1;

        auto over = [&](int i)->bool {
            if (isRowVec) return fracVec.at<float>(0, i) > thr;  // 1 x W
            else          return fracVec.at<float>(i, 0) > thr;  // H x 1
        };

        for (; L <= R; ++L) if (over(L)) break;
        for (; R >= L; --R) if (over(R)) break;

        int len = std::max(0, R - L + 1);
        const int pad = std::max(1, (len * 3) / 100); // ~3%
        L = std::max(0, L + pad);
        R = std::max(L, R - pad);
        return std::make_pair(L, R);
    };

    std::pair<int, int> rangeX = findTightRange(fracPerCol, colThresh, /*isRowVec=*/true);
    std::pair<int, int> rangeY = findTightRange(fracPerRow, rowThresh, /*isRowVec=*/false);
    int lx = rangeX.first, rx = rangeX.second;
    int ty = rangeY.first, by = rangeY.second;

    // c) בנה קופסה הדוקה במרחב המרובע
    std::vector<cv::Point2f> boxWarped;
    boxWarped.push_back(cv::Point2f((float)lx, (float)ty));
    boxWarped.push_back(cv::Point2f((float)rx, (float)ty));
    boxWarped.push_back(cv::Point2f((float)rx, (float)by));
    boxWarped.push_back(cv::Point2f((float)lx, (float)by));

    // d) הקרן חזרה למרחב המקורי בעזרת Hinv + ייצוב קטן
    std::vector<cv::Point2f> refined;
    {
        cv::Mat pts(1, 4, CV_32FC2);
        for (int i = 0; i < 4; ++i) pts.at<cv::Point2f>(0, i) = boxWarped[i];

        cv::Mat ptsBack;
        perspectiveTransform(pts, ptsBack, warpRes.Hinv);

        refined.resize(4);
        for (int i = 0; i < 4; ++i) refined[i] = ptsBack.at<cv::Point2f>(0, i);

        std::vector<cv::Point> refined_i; refined_i.reserve(4);
        for (size_t i = 0; i < refined.size(); ++i)
            refined_i.emplace_back(cvRound(refined[i].x), cvRound(refined[i].y));

        std::vector<cv::Point> hull; cv::convexHull(refined_i, hull, true, true);
        std::vector<cv::Point> approx;
        double eps = 0.01 * cv::arcLength(hull, true);
        cv::approxPolyDP(hull, approx, eps, true);
        if (approx.size() == 4) {
            refined.clear();
            for (size_t i = 0; i < approx.size(); ++i)
                refined.emplace_back((float)approx[i].x, (float)approx[i].y);
        }
    }

    saveIf(drawPolyOverlay(bgr, refined), outdir / (base + "_poly_refined.png"), opt.save_debug, opt.debug);
    t_refine = t5.ms();

    // ---------------------------------------------------------------------
    // (6) Coverage computation
    // ---------------------------------------------------------------------
    double cov = geom::polygonCoveragePercent(refined, bgr.size());

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
    res.polygon = refined;
    res.coverage_percent = cov;
    res.grid_ok = grid_ok;

    // In strict mode, fail the detection if grid validation did not pass.
    if (opt.strict_grid && !grid_ok) {
        if (opt.debug) std::cerr << "[debug] strict_grid=true -> not found\n";
        return std::nullopt;
    }
    return res;
}
