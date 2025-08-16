#include <cassert>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "color_segmenter.hpp"
#include "grid_detector.hpp"
#include "geometry.hpp"

static inline bool approx(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // === Synthetic 3ª3 board (300ª300) ===================================
    // Each cell is ~100ª100 in "allowed" colors.
    cv::Mat img(300, 300, CV_8UC3, cv::Scalar(0, 0, 0));
    auto paint = [&](int r, int c, const cv::Scalar& bgr) {
        cv::rectangle(img, cv::Rect(c * 100, r * 100, 100, 100), bgr, cv::FILLED);
    };
    // Row 0
    paint(0, 0, { 255,   0,   0 });   // Blue
    paint(0, 1, { 0, 255, 255 });   // Yellow
    paint(0, 2, { 0,   0, 255 });   // Red
    // Row 1
    paint(1, 0, { 0, 255,   0 });   // Green
    paint(1, 1, { 255,   0, 255 });   // Magenta
    paint(1, 2, { 255, 255,   0 });   // Cyan
    // Row 2
    paint(2, 0, { 255,   0,   0 });   // Blue
    paint(2, 1, { 0, 255, 255 });   // Yellow
    paint(2, 2, { 0,   0, 255 });   // Red

    // Mask of allowed colors
    cv::Mat mask = ColorSegmenter::allowedMaskHSV(img);
    assert(!mask.empty());
    assert(cv::countNonZero(mask) > 0 && "Allowed-color mask should not be empty");

    // Seams near ~1/3 and ~2/3
    auto seams = grid::checkGridSeams(mask);
    assert(seams.ok && "Expected ~3ª3 seams to be detected");

    // Cells: each of the 9 cells should have enough allowed-color pixels
    auto cells = grid::checkGridCells(mask, /*minFraction=*/0.2);
    assert(cells.ok && "All 9 cells should pass the per-cell fraction threshold");

    // Coverage: full-quad over the image should be ~100%
    std::vector<cv::Point2f> quad = {
        {0.f, 0.f}, {299.f, 0.f}, {299.f, 299.f}, {0.f, 299.f}
    };
    double cov = geom::polygonCoveragePercent(quad, img.size());
    assert(approx(cov, 100.0, 1e-6) && "Full image polygon should yield ~100% coverage");

    // === Negative cases ===================================================
    // 1) All-black image -> empty mask
    cv::Mat empty(300, 300, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat m0 = ColorSegmenter::allowedMaskHSV(empty);
    assert(cv::countNonZero(m0) == 0 && "Black image should yield empty allowed-color mask");

    // 2) Gray image (low saturation) -> should also be mostly empty after white suppression
    cv::Mat gray(300, 300, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat mg = ColorSegmenter::allowedMaskHSV(gray);
    // Not strictly zero (CLAHE/threshold nuance), but should be very sparse:
    double frac_g = (double)cv::countNonZero(mg) / (double)mg.total();
    assert(frac_g < 0.01 && "Gray image should not be classified as colored (low S)");

    return 0;
}
