#include "grid_detector.hpp"
using namespace cv;

namespace {
    // Find the index of the minimum value in a 1D CV_32S Mat within [lo, hi).
    static int minIndexInRange(const Mat& vec, int lo, int hi, bool isColVec) {
        int bestIdx = lo;
        int bestVal = INT_MAX;
        if (isColVec) {
            // vec is 1 x W
            for (int x = lo; x < hi; ++x) {
                int v = vec.at<int>(0, x);
                if (v < bestVal) { bestVal = v; bestIdx = x; }
            }
        }
        else {
            // vec is H x 1
            for (int y = lo; y < hi; ++y) {
                int v = vec.at<int>(y, 0);
                if (v < bestVal) { bestVal = v; bestIdx = y; }
            }
        }
        return bestIdx;
    }

    static bool near(int x, int target, int tol) {
        return std::abs(x - target) <= tol;
    }
}

namespace {
    // Safe fraction of non-zero pixels in ROI (returns 0 on empty/invalid).
    static double safeFraction(const Mat& roi) {
        if (roi.empty()) return 0.0;
        const double total = (double)roi.total();
        if (total <= 0.0) return 0.0;
        return (double)countNonZero(roi) / total;
    }
}

grid::Seams grid::checkGridSeams(const Mat& mask) {
    CV_Assert(!mask.empty() && mask.type() == CV_8UC1);
    const int W = mask.cols, H = mask.rows;

    Mat colsum, rowsum;
    reduce(mask, colsum, 0, REDUCE_SUM, CV_32S); // 1 x W
    reduce(mask, rowsum, 1, REDUCE_SUM, CV_32S); // H x 1

    // Flexible search ranges for seams around ~1/3 and ~2/3.
    const int vx1_lo = W / 5, vx1_hi = 2 * W / 5;
    const int vx2_lo = 3 * W / 5, vx2_hi = 4 * W / 5;
    const int hy1_lo = H / 5, hy1_hi = 2 * H / 5;
    const int hy2_lo = 3 * H / 5, hy2_hi = 4 * H / 5;

    Seams s;
    s.cx1 = minIndexInRange(colsum, vx1_lo, vx1_hi, /*isColVec=*/true);
    s.cx2 = minIndexInRange(colsum, vx2_lo, vx2_hi, /*isColVec=*/true);
    s.cy1 = minIndexInRange(rowsum, hy1_lo, hy1_hi, /*isColVec=*/false);
    s.cy2 = minIndexInRange(rowsum, hy2_lo, hy2_hi, /*isColVec=*/false);

    // Tolerant thresholds around ideal thirds.
    const int tolX = W / 6;
    const int tolY = H / 6;

    const bool okX = near(s.cx1, W / 3, tolX) && near(s.cx2, 2 * W / 3, tolX);
    const bool okY = near(s.cy1, H / 3, tolY) && near(s.cy2, 2 * H / 3, tolY);

    // Ensure seams are reasonably separated.
    const bool spacing_ok = (s.cx2 - s.cx1) > W / 6 && (s.cy2 - s.cy1) > H / 6;

    s.ok = okX && okY && spacing_ok;
    return s;
}

grid::CellsReport grid::checkGridCells(const Mat& mask, double minFraction) {
    CV_Assert(!mask.empty() && mask.type() == CV_8UC1);
    CV_Assert(mask.rows == mask.cols); // expect a square (warped) mask

    const int N = mask.rows;
    const int cell = N / 3;

    CellsReport rep{};
    rep.ok = true;

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            const int x = c * cell;
            const int y = r * cell;
            const int w = (c == 2 ? N - x : cell);
            const int h = (r == 2 ? N - y : cell);
            Mat roi = mask(Rect(x, y, w, h));
            const double f = safeFraction(roi);
            rep.frac[r][c] = f;
            if (f < minFraction) rep.ok = false;
        }
    }
    return rep;
}
