#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <optional>

namespace geom {

    /// @brief Warped square image + homographies (src->dst, dst->src).
    struct WarpResult {
        cv::Mat image;   // N x N
        cv::Mat H;       // 3x3, src->dst
        cv::Mat Hinv;    // 3x3, dst->src
    };

    /**
     * @brief Find a strong quadrilateral in a binary mask (CV_8UC1).
     * Strategy: largest contour → approxPolyDP (convex 4-pt) else minAreaRect.
     * @return 4 points (clockwise) or nullopt.
     */
    std::optional<std::vector<cv::Point2f>>
        findStrongQuad(const cv::Mat& allowedMask);

    /**
     * @brief Warp BGR to N×N square using a clockwise quad.
     * @return Warped CV_8UC3 image. Throws on invalid homography.
     */
    cv::Mat warpToSquare(const cv::Mat& bgr,
        const std::vector<cv::Point2f>& quad,
        int N);

    /**
     * @brief Warp and also return H/Hinv.
     * @return WarpResult with image, H (src->dst), Hinv (dst->src).
     */
    WarpResult warpToSquareWithH(const cv::Mat& bgr,
        const std::vector<cv::Point2f>& quad,
        int N);

    /**
     * @brief 100 * area(poly) / (W*H). Poly in original coords.
     * @return Percentage in [0, 100]. 0 if invalid/empty.
     */
    double polygonCoveragePercent(const std::vector<cv::Point2f>& poly,
        const cv::Size& sz);
}
