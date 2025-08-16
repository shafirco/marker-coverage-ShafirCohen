#pragma once
#include <opencv2/opencv.hpp>

namespace grid {

    /// @brief Approximate seam (grid-line) positions in X/Y.
    struct Seams {
        int cx1 = -1, cx2 = -1;
        int cy1 = -1, cy2 = -1;
        bool ok = false;   // true if both vertical and horizontal seams found
    };

    /// @brief Per-cell coverage for the 3×3 grid.
    struct CellsReport {
        double frac[3][3]; // fraction of "allowed" pixels in each cell
        bool ok = false;   // true if all cells ≥ minFraction
    };

    /// @brief Check vertical/horizontal seams at ~1/3 and ~2/3 of the mask.
    Seams checkGridSeams(const cv::Mat& mask);

    /// @brief Check each of the 9 cells in a warped N×N mask.
    /// @param mask Warped binary mask.
    /// @param minFraction Minimum fraction of "allowed" pixels per cell.
    /// @return Fractions per cell + overall pass/fail.
    CellsReport checkGridCells(const cv::Mat& mask, double minFraction);
}
