#include "grid_detector.hpp"
using namespace cv;

namespace {
    // Find index of minimum value in a 1D CV_32S Mat range [lo, hi)
    static int minIndexInRange(const Mat& vec, int lo, int hi, bool isColVec) {
        int bestIdx = lo;
        int bestVal = INT_MAX;
        if (isColVec) {
            // 1 x W
            for (int x = lo; x < hi; ++x) {
                int v = vec.at<int>(0, x);
                if (v < bestVal) { bestVal = v; bestIdx = x; }
            }
        }
        else {
            // H x 1
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

grid::Seams grid::checkGridSeams(const Mat& mask) {
    CV_Assert(!mask.empty() && mask.type() == CV_8UC1);
    const int W = mask.cols, H = mask.rows;

    Mat colsum, rowsum;
    reduce(mask, colsum, 0, REDUCE_SUM, CV_32S); // 1 x W
    reduce(mask, rowsum, 1, REDUCE_SUM, CV_32S); // H x 1

    const int vx1_lo = W / 6, vx1_hi = W / 2;
    const int vx2_lo = W / 2, vx2_hi = 5 * W / 6;
    const int hy1_lo = H / 6, hy1_hi = H / 2;
    const int hy2_lo = H / 2, hy2_hi = 5 * H / 6;

    Seams s;
    s.cx1 = minIndexInRange(colsum, vx1_lo, vx1_hi, /*isColVec=*/true);
    s.cx2 = minIndexInRange(colsum, vx2_lo, vx2_hi, /*isColVec=*/true);
    s.cy1 = minIndexInRange(rowsum, hy1_lo, hy1_hi, /*isColVec=*/false);
    s.cy2 = minIndexInRange(rowsum, hy2_lo, hy2_hi, /*isColVec=*/false);

    const int tolX = W / 12; // tolerant thresholds
    const int tolY = H / 12;

    const bool okX = near(s.cx1, W / 3, tolX) && near(s.cx2, 2 * W / 3, tolX);
    const bool okY = near(s.cy1, H / 3, tolY) && near(s.cy2, 2 * H / 3, tolY);
    s.ok = okX && okY;
    return s;
}
