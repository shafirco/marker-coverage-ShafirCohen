#include <opencv2/imgproc.hpp>
#include "color_segmenter.hpp"
using namespace cv;

namespace {
    struct HsvRange { int hmin, hmax, smin, smax, vmin, vmax; };

    static Mat inRangeH(const Mat& hsv, const HsvRange& r) {
        Mat mask;
        inRange(hsv, Scalar(r.hmin, r.smin, r.vmin),
            Scalar(r.hmax, r.smax, r.vmax), mask);
        return mask;
    }

    static int clampi(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }

    // Build the allowed-colors mask from base HSV ranges, clamped by global S/V floors.
    static Mat buildAllowedMaskHSV(const Mat& hsv, int smin, int vmin) {
        HsvRange red1{ 0,  10,  80,255,  50,255 };
        HsvRange red2{ 170, 180,  80,255,  50,255 };
        HsvRange green{ 40,  85,  60,255,  50,255 };
        HsvRange yellow{ 20,  35,  80,255,  70,255 };
        HsvRange blue{ 90, 130,  60,255,  50,255 };
        HsvRange magenta{ 135, 165,  60,255,  50,255 };
        HsvRange cyan{ 85, 100,  60,255,  60,255 };

        auto applySV = [&](HsvRange r) {
            r.smin = clampi(std::max(r.smin, smin), 0, 255);
            r.vmin = clampi(std::max(r.vmin, vmin), 0, 255);
            return r;
        };

        red1 = applySV(red1);
        red2 = applySV(red2);
        green = applySV(green);
        yellow = applySV(yellow);
        blue = applySV(blue);
        magenta = applySV(magenta);
        cyan = applySV(cyan);

        Mat mask = inRangeH(hsv, red1) | inRangeH(hsv, red2)
            | inRangeH(hsv, green) | inRangeH(hsv, yellow)
            | inRangeH(hsv, blue) | inRangeH(hsv, magenta)
            | inRangeH(hsv, cyan);

        // Suppress white/highlights (low S, high V).
        Mat white;
        inRange(hsv, Scalar(0, 0, 210), Scalar(180, 60, 255), white);
        bitwise_and(mask, ~white, mask);

        return mask;
    }

    // Gentle S/V relaxation if the mask is extremely sparse.
    static void gentleRelaxIfSparse(const Mat& hsv, Mat& mask, int& smin, int& vmin) {
        const double total = (double)hsv.total();
        for (int attempt = 0; attempt < 2; ++attempt) {
            const double frac = (double)countNonZero(mask) / std::max(1.0, total);
            if (frac >= 0.001) break; // ≥0.1% is enough — do not relax further
            smin = clampi(smin - 10, 0, 255);
            vmin = clampi(vmin - 10, 0, 255);
            mask = buildAllowedMaskHSV(hsv, smin, vmin);
        }
    }
}

namespace {
    // --- White-Rim Booster helpers ---------------------------------------

    // Mild unsharp mask on the V channel to emphasize blurry bright rims.
    static void unsharp_on_V(const cv::Mat& hsv, cv::Mat& V_sharp, double sigma = 1.0, double amount = 1.0) {
        CV_Assert(hsv.type() == CV_8UC3);
        std::vector<cv::Mat> ch; cv::split(hsv, ch);
        cv::Mat V_blur; cv::GaussianBlur(ch[2], V_blur, cv::Size(), sigma, sigma);
        cv::Mat diff; cv::subtract(ch[2], V_blur, diff, cv::noArray(), CV_16S);
        cv::Mat add;  cv::addWeighted(ch[2], 1.0, diff, amount, 0.0, add, CV_8U);
        V_sharp = add;
    }

    // Bright edges from V_sharp (Sobel magnitude + threshold).
    static cv::Mat bright_edges_from_V(const cv::Mat& V_sharp, int edge_thresh = 25) {
        CV_Assert(V_sharp.type() == CV_8U);
        cv::Mat gx, gy, mag;
        cv::Sobel(V_sharp, gx, CV_16S, 1, 0, 3);
        cv::Sobel(V_sharp, gy, CV_16S, 0, 1, 3);
        cv::Mat ax = cv::abs(gx), ay = cv::abs(gy);
        cv::add(ax, ay, mag, cv::noArray(), CV_16S);
        cv::Mat mag8u; cv::convertScaleAbs(mag, mag8u, 0.5); // light normalization
        cv::Mat edges; cv::threshold(mag8u, edges, edge_thresh, 255, cv::THRESH_BINARY);
        return edges;
    }

    // Build white_rim: white candidate (low S, high V) ∧ expanded bright edges.
    static cv::Mat build_white_rim(const cv::Mat& hsv, int s_max = 110, int v_min = 200,
        int edge_thresh = 25, int dil_iter = 1)
    {
        CV_Assert(hsv.type() == CV_8UC3);
        // (1) white candidates
        cv::Mat white1, white2, white_cand;
        cv::inRange(hsv, cv::Scalar(0, 0, v_min), cv::Scalar(180, s_max, 255), white1); // low S & high V
        cv::inRange(hsv, cv::Scalar(0, 0, 220), cv::Scalar(180, 255, 255), white2);   // strong highlights
        white_cand = white1 | white2;

        // (2) bright edges from V_sharp
        cv::Mat V_sharp; unsharp_on_V(hsv, V_sharp);
        cv::Mat edges = bright_edges_from_V(V_sharp, edge_thresh);
        cv::Mat edges_dil = edges;
        if (dil_iter > 0) {
            cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::dilate(edges_dil, edges_dil, k, cv::Point(-1, -1), dil_iter);
        }

        // (3) rim = white that lies on/near a bright edge
        cv::Mat white_rim; cv::bitwise_and(white_cand, edges_dil, white_rim);
        if (dil_iter > 0) {
            cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::dilate(white_rim, white_rim, k, cv::Point(-1, -1), 1);
        }
        return white_rim;
    }
}

Mat ColorSegmenter::allowedMaskHSV(const Mat& bgr) {
    SegOptions def; // use defaults
    return allowedMaskHSV(bgr, def);
}

Mat ColorSegmenter::allowedMaskHSV(const Mat& bgr, const SegOptions& opt) {
    CV_Assert(!bgr.empty() && bgr.type() == CV_8UC3);

    // Optional Gaussian blur (as in original).
    Mat src = bgr;
    if (opt.blur_ksize >= 3 && (opt.blur_ksize % 2) == 1) {
        GaussianBlur(bgr, src, Size(opt.blur_ksize, opt.blur_ksize), 0.0);
    }

    // Convert to HSV + CLAHE on V (preserved from original).
    Mat hsv; cvtColor(src, hsv, COLOR_BGR2HSV);
    std::vector<Mat> ch; split(hsv, ch);
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(2.0);
    clahe->setTilesGridSize(Size(8, 8));
    clahe->apply(ch[2], ch[2]);
    merge(ch, hsv);

    // Base color mask with global S/V floors.
    int smin = opt.smin;
    int vmin = opt.vmin;
    Mat mask = buildAllowedMaskHSV(hsv, smin, vmin);

    // --- Stage 0.5: White Rim Booster (detach blurry white border if present) ---
    {
        // Build a plausible white rim and subtract it from the mask, with safety brake.
        cv::Mat white_rim = build_white_rim(hsv, /*s_max=*/110, /*v_min=*/200, /*edge_thresh=*/25, /*dil=*/1);

        const double nz0 = std::max(1.0, (double)cv::countNonZero(mask));
        cv::Mat mask_after = mask.clone();
        cv::bitwise_and(mask_after, ~white_rim, mask_after);
        const double nz1 = (double)cv::countNonZero(mask_after);

        // Brake: if we removed too much (>30%), keep the original mask.
        if (nz1 >= 0.65 * nz0) {
            mask = mask_after;
        }
        // else: keep mask as-is
    }


    // Gentle relaxation only if the mask is extremely sparse.
    gentleRelaxIfSparse(hsv, mask, smin, vmin);

    // Morphological cleanup (same as original).
    Mat k = getStructuringElement(MORPH_RECT, Size(3, 3));
    if (opt.open_iter > 0)  morphologyEx(mask, mask, MORPH_OPEN, k, Point(-1, -1), opt.open_iter);
    if (opt.close_iter > 0) morphologyEx(mask, mask, MORPH_CLOSE, k, Point(-1, -1), opt.close_iter);

    return mask;
}
