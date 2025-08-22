#include <cassert>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "color_segmenter.hpp"
#include "grid_detector.hpp"
#include "geometry.hpp"

static inline bool approx(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) <= eps;
}

// helper: rotate with border fill
static cv::Mat rotate_keep_all(const cv::Mat& src, double angle_deg) {
    using namespace cv;
    Point2f center(src.cols * 0.5f, src.rows * 0.5f);
    Mat R = getRotationMatrix2D(center, angle_deg, 1.0);
    Rect2f bb = RotatedRect(center, src.size(), angle_deg).boundingRect2f();
    R.at<double>(0, 2) += bb.width * 0.5 - center.x;
    R.at<double>(1, 2) += bb.height * 0.5 - center.y;
    Mat dst;
    warpAffine(src, dst, R, Size((int)bb.width, (int)bb.height),
        INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
    return dst;
}

int main() {
    // === Synthetic 3x3 board (300x300) ===================================
    cv::Mat img(300, 300, CV_8UC3, cv::Scalar(0, 0, 0));
    auto paint = [&](int r, int c, const cv::Scalar& bgr) {
        cv::rectangle(img, cv::Rect(c * 100, r * 100, 100, 100), bgr, cv::FILLED);
    };
    // Row 0
    paint(0, 0, { 255,   0,   0 });   // Blue
    paint(0, 1, { 0, 255, 255 });     // Yellow
    paint(0, 2, { 0,   0, 255 });     // Red
    // Row 1
    paint(1, 0, { 0, 255,   0 });     // Green
    paint(1, 1, { 255,   0, 255 });   // Magenta
    paint(1, 2, { 255, 255,   0 });   // Cyan
    // Row 2
    paint(2, 0, { 255,   0,   0 });   // Blue
    paint(2, 1, { 0, 255, 255 });     // Yellow
    paint(2, 2, { 0,   0, 255 });     // Red

    // Mask of allowed colors
    cv::Mat mask = ColorSegmenter::allowedMaskHSV(img);
    assert(!mask.empty());
    assert(cv::countNonZero(mask) > 0);

    // Seams
    auto seams = grid::checkGridSeams(mask);
    assert(seams.ok && "Expected 3x3 seams");

    // Cells
    auto cells = grid::checkGridCells(mask, 0.2);
    assert(cells.ok && "Expected all cells to pass");

    // Coverage: full quad ~100%
    std::vector<cv::Point2f> quad = {
        {0.f,0.f},{300.f,0.f},{300.f,300.f},{0.f,300.f}
    };
    double cov = geom::polygonCoveragePercent(quad, img.size());
    assert(approx(cov, 100.0, 1.0));  // Allow 1% tolerance for rounding

    // === Negative cases ===
    cv::Mat empty(300, 300, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat m0 = ColorSegmenter::allowedMaskHSV(empty);
    assert(cv::countNonZero(m0) == 0);

    cv::Mat gray(300, 300, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat mg = ColorSegmenter::allowedMaskHSV(gray);
    double frac_g = (double)cv::countNonZero(mg) / (double)mg.total();
    assert(frac_g < 0.01);

    // === Rotation tests ===
    // Test 30° rotation
    cv::Mat rot30 = rotate_keep_all(img, 30.0);
    cv::Mat mask_rot30 = ColorSegmenter::allowedMaskHSV(rot30);
    assert(cv::countNonZero(mask_rot30) > 0 && "30° rotated grid should still be detected");
    
    // Test 45° rotation (requirement: robust to ±45°)
    cv::Mat rot45 = rotate_keep_all(img, 45.0);
    cv::Mat mask_rot45 = ColorSegmenter::allowedMaskHSV(rot45);
    assert(cv::countNonZero(mask_rot45) > 0 && "45° rotated grid should still be detected");
    
    // Test negative rotation
    cv::Mat rot_neg30 = rotate_keep_all(img, -30.0);
    cv::Mat mask_rot_neg30 = ColorSegmenter::allowedMaskHSV(rot_neg30);
    assert(cv::countNonZero(mask_rot_neg30) > 0 && "-30° rotated grid should still be detected");

    return 0;
}
