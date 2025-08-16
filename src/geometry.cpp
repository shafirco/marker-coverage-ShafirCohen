#include "geometry.hpp"
#include <algorithm>
#include <cmath>

using namespace cv;
using std::vector;

namespace {
    /// @brief Sort 4 points clockwise around centroid, starting from top-left.
    static vector<Point2f> sortClockwiseTL(const vector<Point2f>& ptsIn) {
        CV_Assert(ptsIn.size() == 4);
        Point2f c(0, 0);
        for (auto& p : ptsIn) c += p;
        c *= (1.f / 4.f);

        vector<std::pair<float, Point2f>> withAngle;
        withAngle.reserve(4);
        for (auto& p : ptsIn) {
            float a = std::atan2(p.y - c.y, p.x - c.x);
            withAngle.push_back({ a, p });
        }
        std::sort(withAngle.begin(), withAngle.end(),
            [](auto& a, auto& b) { return a.first < b.first; });

        // Rotate so the first is top-left (min x+y heuristic).
        int start = 0;
        float best = 1e9f;
        for (int i = 0; i < 4; ++i) {
            auto p = withAngle[i].second;
            float score = p.x + p.y;
            if (score < best) { best = score; start = i; }
        }

        vector<Point2f> out(4);
        for (int i = 0; i < 4; ++i) {
            out[i] = withAngle[(start + i) % 4].second; // TL, TR, BR, BL
        }
        return out;
    }
}

std::optional<std::vector<Point2f>>
geom::findStrongQuad(const Mat& allowedMask) {
    CV_Assert(!allowedMask.empty() && allowedMask.type() == CV_8UC1);

    // Conservative tweak: single 3×3 close to fill small holes.
    Mat work = allowedMask.clone();
    Mat k3 = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(work, work, MORPH_CLOSE, k3, Point(-1, -1), 1);

    vector<vector<Point>> contours;
    findContours(work, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return std::nullopt;

    // Select largest contour by area.
    double bestA = 0.0;
    vector<Point> best;
    for (auto& c : contours) {
        double a = contourArea(c);
        if (a > bestA) { bestA = a; best = c; }
    }
    if (best.empty()) return std::nullopt;

    // Try direct polygon approximation.
    vector<Point> approx;
    approxPolyDP(best, approx, 0.02 * arcLength(best, true), true);
    if (approx.size() == 4 && isContourConvex(approx)) {
        vector<Point2f> q;
        q.reserve(4);
        for (auto& p : approx) q.push_back(Point2f((float)p.x, (float)p.y));
        return sortClockwiseTL(q);
    }

    // Fallback: minAreaRect box.
    RotatedRect rr = minAreaRect(best);
    Point2f p4[4];
    rr.points(p4);
    vector<Point2f> q{ p4[0], p4[1], p4[2], p4[3] };
    return sortClockwiseTL(q);
}

cv::Mat geom::warpToSquare(const Mat& bgr, const vector<Point2f>& quad, int N) {
    CV_Assert(!bgr.empty() && bgr.type() == CV_8UC3);
    CV_Assert(quad.size() == 4 && N > 0);

    vector<Point2f> src = sortClockwiseTL(quad);
    vector<Point2f> dst{
        {0.f, 0.f}, {(float)N - 1, 0.f},
        {(float)N - 1, (float)N - 1}, {0.f, (float)N - 1}
    };
    Mat H = getPerspectiveTransform(src, dst);
    Mat out;
    warpPerspective(bgr, out, H, Size(N, N), INTER_LINEAR, BORDER_REPLICATE);
    return out;
}

geom::WarpResult geom::warpToSquareWithH(const cv::Mat& bgr,
    const std::vector<cv::Point2f>& quad,
    int N)
{
    CV_Assert(!bgr.empty() && bgr.type() == CV_8UC3);
    CV_Assert(quad.size() == 4 && N > 0);

    std::vector<Point2f> src = sortClockwiseTL(quad);
    std::vector<Point2f> dst{
        Point2f(0.f,          0.f),
        Point2f((float)N - 1, 0.f),
        Point2f((float)N - 1,(float)N - 1),
        Point2f(0.f,         (float)N - 1)
    };

    Mat H = getPerspectiveTransform(src, dst);
    Mat out;
    warpPerspective(bgr, out, H, Size(N, N), INTER_LINEAR, BORDER_REPLICATE);

    Mat Hinv; invert(H, Hinv, DECOMP_SVD);
    return { out, H, Hinv };
}

double geom::polygonCoveragePercent(const vector<Point2f>& poly, const Size& sz) {
    if (poly.size() < 3) return 0.0;
    double A = contourArea(poly);
    double total = (double)sz.width * (double)sz.height;
    if (total <= 0.0) return 0.0;
    return 100.0 * A / total;
}
