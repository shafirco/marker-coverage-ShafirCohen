#include <cassert>
#include <opencv2/opencv.hpp>
#include "color_segmenter.hpp"
#include "grid_detector.hpp"

int main() {
    // Synthetic 3x3 board (300x300), each cell ~100x100 in allowed colors
    cv::Mat img(300, 300, CV_8UC3, cv::Scalar(0, 0, 0));
    auto paint = [&](int r, int c, cv::Scalar bgr) {
        cv::rectangle(img, cv::Rect(c * 100, r * 100, 100, 100), bgr, cv::FILLED);
    };
    paint(0, 0, { 255,0,0 });    // Blue
    paint(0, 1, { 0,255,255 });  // Yellow
    paint(0, 2, { 0,0,255 });    // Red
    paint(1, 0, { 0,255,0 });    // Green
    paint(1, 1, { 255,0,255 });  // Magenta
    paint(1, 2, { 255,255,0 });  // Cyan
    paint(2, 0, { 255,0,0 });    // Blue
    paint(2, 1, { 0,255,255 });  // Yellow
    paint(2, 2, { 0,0,255 });    // Red

    // Color mask
    cv::Mat mask = ColorSegmenter::allowedMaskHSV(img);
    assert(cv::countNonZero(mask) > 0 && "Allowed mask should not be empty");

    // Seam check ~1/3 and ~2/3
    auto seams = grid::checkGridSeams(mask);
    assert(seams.ok && "Expected 3x3 seams to be detected");

    return 0;
}
