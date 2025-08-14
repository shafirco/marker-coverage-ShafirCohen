#pragma once
#include <opencv2/opencv.hpp>

namespace grid {

    struct Seams {
        int cx1 = -1; // vertical seam near ~1/3 width
        int cx2 = -1; // vertical seam near ~2/3 width
        int cy1 = -1; // horizontal seam near ~1/3 height
        int cy2 = -1; // horizontal seam near ~2/3 height
        bool ok = false;
    };

    /**
     * Given a binary mask of the warped marker (NªN), detect four "seams":
     * two vertical and two horizontal lines where color transitions occur
     * around ~1/3 and ~2/3 of the width/height.
     *
     * Implementation:
     *  - Reduce columns/rows (sum) to get 1D signals.
     *  - Find minima in [W/6..W/2] & [W/2..5W/6] and similarly for rows.
     *  - Check if they are near W/3 and 2W/3 (tolerant threshold).
     */
    Seams checkGridSeams(const cv::Mat& mask);
}
