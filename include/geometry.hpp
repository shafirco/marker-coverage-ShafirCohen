#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <optional>

namespace geom {

    /**
     * Try to extract a strong quadrilateral from a binary mask (CV_8UC1).
     * Strategy:
     *  - Find external contours over the mask of allowed colors.
     *  - Take the largest contour by area.
     *  - If approxPolyDP gives a convex 4-vertex polygon → use it.
     *  - Otherwise, fall back to the minAreaRect box.
     * Returns the 4 corners in clockwise order (float).
     */
    std::optional<std::vector<cv::Point2f>>
        findStrongQuad(const cv::Mat& allowedMask);

    /**
     * Warp the input BGR image to a square N×N using the given quadrilateral.
     * The quad is expected to be in clockwise order.
     */
    cv::Mat warpToSquare(const cv::Mat& bgr,
        const std::vector<cv::Point2f>& quad,
        int N);

    /**
     * Compute polygon coverage in percent relative to the whole image size.
     * poly: polygon in the ORIGINAL image coordinates.
     * sz: original image size.
     * returns: 100.0 * area(poly) / (W*H)
     */
    double polygonCoveragePercent(const std::vector<cv::Point2f>& poly,
        const cv::Size& sz);
}
