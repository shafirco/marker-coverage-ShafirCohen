#pragma once
#include <opencv2/opencv.hpp>

namespace grid {

    struct Seams {
        int cx1 = -1, cx2 = -1;
        int cy1 = -1, cy2 = -1;
        bool ok = false;
    };

    // NEW: per-cell coverage report for the 3x3 grid
    struct CellsReport {
        double frac[3][3];  // fraction of "allowed" pixels in each cell
        bool ok = false;    // true iff every cell >= minFraction
    };

    // Existing:
    Seams checkGridSeams(const cv::Mat& mask);

    // NEW: checks each of the 9 cells in the warped mask (N×N).
    // Returns per-cell fractions and overall pass/fail by minFraction.
    CellsReport checkGridCells(const cv::Mat& mask, double minFraction);
}
