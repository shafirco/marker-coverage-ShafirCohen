#include <opencv2/imgproc.hpp>
#include "color_segmenter.hpp"
using namespace cv;

namespace {
    struct HsvRange { int hmin, hmax, smin, smax, vmin, vmax; };

    static Mat inRangeH(const Mat& hsv, const HsvRange& r) {
        Mat mask;
        inRange(hsv, Scalar(r.hmin, r.smin, r.vmin),
            Scalar(r.hmax, r.smax, r.vmax), mask);
        return mask;
    }

    // Clamp helper
    static int clamp(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }
}

Mat ColorSegmenter::allowedMaskHSV(const Mat& bgr) {
    SegOptions def; // use defaults
    return allowedMaskHSV(bgr, def);
}

Mat ColorSegmenter::allowedMaskHSV(const Mat& bgr, const SegOptions& opt) {
    CV_Assert(!bgr.empty() && bgr.type() == CV_8UC3);

    Mat src = bgr;
    if (opt.blur_ksize >= 3 && (opt.blur_ksize % 2) == 1) {
        GaussianBlur(bgr, src, Size(opt.blur_ksize, opt.blur_ksize), 0.0);
    }

    Mat hsv; cvtColor(src, hsv, COLOR_BGR2HSV);
    std::vector<Mat> hsvch; split(hsv, hsvch);

    Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0); 
    clahe->setTilesGridSize(Size(8, 8));
    clahe->apply(hsvch[2], hsvch[2]);
    merge(hsvch, hsv);

    // Base HSV ranges (can be tightened/loosened via global smin/vmin)
    HsvRange red1{ 0, 10,  80,255, 50,255 };
    HsvRange red2{ 170,180,  80,255, 50,255 };
    HsvRange green{ 40, 85,  60,255, 50,255 };
    HsvRange yellow{ 20, 35,  80,255, 70,255 };
    HsvRange blue{ 90,130,  60,255, 50,255 };
    HsvRange magenta{ 135,165,  60,255, 50,255 };
    HsvRange cyan{ 85,100,  60,255, 60,255 };

    // Apply global lower-bounds for S and V
    auto applySV = [&](HsvRange r) {
        r.smin = clamp(std::max(r.smin, opt.smin), 20, 200);
        r.vmin = clamp(std::max(r.vmin, opt.vmin), 0, 255);
        return r;
    };

    red1 = applySV(red1); red2 = applySV(red2);
    green = applySV(green); yellow = applySV(yellow);
    blue = applySV(blue); magenta = applySV(magenta); cyan = applySV(cyan);

    Mat mask = inRangeH(hsv, red1) | inRangeH(hsv, red2)
        | inRangeH(hsv, green) | inRangeH(hsv, yellow)
        | inRangeH(hsv, blue) | inRangeH(hsv, magenta)
        | inRangeH(hsv, cyan);

    // Remove whites/highlights: low-S & high-V
    cv::Mat white;
    inRange(hsv, cv::Scalar(0, 0, 210), cv::Scalar(180, 60, 255), white);
    bitwise_and(mask, ~white, mask);

    // Morphological cleanup
    Mat k = getStructuringElement(MORPH_RECT, Size(3, 3));
    if (opt.open_iter > 0)  morphologyEx(mask, mask, MORPH_OPEN, k, Point(-1, -1), opt.open_iter);
    if (opt.close_iter > 0) morphologyEx(mask, mask, MORPH_CLOSE, k, Point(-1, -1), opt.close_iter);

    return mask;
}

