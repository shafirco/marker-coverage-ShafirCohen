#include "color_segmenter.hpp"
using namespace cv;

namespace {
    struct HsvRange { int hmin, hmax, smin, smax, vmin, vmax; };

    static Mat inRangeH(const Mat& hsv, const HsvRange& r) {
        Mat mask;
        inRange(hsv, Scalar(r.hmin, r.smin, r.vmin), Scalar(r.hmax, r.smax, r.vmax), mask);
        return mask;
    }
}

Mat ColorSegmenter::allowedMaskHSV(const Mat& bgr) {
    // Preconditions
    CV_Assert(!bgr.empty() && bgr.type() == CV_8UC3);

    // Convert to HSV for robust color thresholding
    Mat hsv; cvtColor(bgr, hsv, COLOR_BGR2HSV);

    // Initial rough HSV ranges (will likely need tuning on your dataset)
    // Red wraps around (0 and 180), so we combine two ranges.
    HsvRange red1{ 0, 10,   80,255,  50,255 };
    HsvRange red2{ 170,180,  80,255,  50,255 };
    HsvRange green{ 40, 85,  60,255,  50,255 };
    HsvRange yellow{ 20, 35,  80,255,  70,255 };
    HsvRange blue{ 90,130,  60,255,  50,255 };
    HsvRange magenta{ 135,165,  60,255,  50,255 };
    HsvRange cyan{ 85,100,  60,255,  60,255 };

    // Merge all allowed color masks
    Mat mask = inRangeH(hsv, red1) | inRangeH(hsv, red2)
        | inRangeH(hsv, green) | inRangeH(hsv, yellow)
        | inRangeH(hsv, blue) | inRangeH(hsv, magenta)
        | inRangeH(hsv, cyan);

    // Morphological cleanup to reduce noise
    Mat k = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_OPEN, k, Point(-1, -1), 1);
    morphologyEx(mask, mask, MORPH_CLOSE, k, Point(-1, -1), 2);

    return mask;
}
