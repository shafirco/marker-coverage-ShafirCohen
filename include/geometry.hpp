/**
 * @file geometry.hpp
 * @brief Geometric operations for marker detection and perspective correction
 * 
 * Provides functions for quadrilateral detection, perspective transformation,
 * and coverage calculation used in the marker detection pipeline.
 */
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <optional>

namespace geom {

    /**
     * @brief Result of perspective transformation with homography matrices
     * 
     * Contains the warped image and transformation matrices for coordinate conversion.
     */
    struct WarpResult {
        /// @brief Perspective-corrected square image (N×N pixels)
        cv::Mat image;
        
        /// @brief Forward homography matrix (3×3, source → destination)
        cv::Mat H;
        
        /// @brief Inverse homography matrix (3×3, destination → source)  
        cv::Mat Hinv;
    };

    /**
     * @brief Extract the strongest quadrilateral from a binary mask
     * 
     * Uses a two-stage approach:
     * 1. Find largest contour by area
     * 2. Try polygon approximation to 4 convex points
     * 3. Fallback to minimum-area rectangle if approximation fails
     * 
     * @param allowedMask Binary mask (CV_8UC1) where 255 = marker pixels
     * @return 4 corner points ordered clockwise from top-left, or nullopt if no suitable quadrilateral found
     * 
     * @note Applies morphological closing to fill small gaps before contour detection
     * @note Points are automatically sorted clockwise starting from top-left corner
     */
    std::optional<std::vector<cv::Point2f>>
        findStrongQuad(const cv::Mat& allowedMask);

    /**
     * @brief Apply perspective correction to transform quadrilateral to square
     * 
     * @param bgr Input BGR image (CV_8UC3)
     * @param quad 4 corner points of the quadrilateral region
     * @param N Output square size (N×N pixels)
     * @return Perspective-corrected square image (CV_8UC3)
     * 
     * @throws cv::Exception on invalid homography computation
     * @note Uses INTER_LINEAR interpolation with BORDER_REPLICATE for edge handling
     * @note Quadrilateral points are automatically reordered if needed
     */
    cv::Mat warpToSquare(const cv::Mat& bgr,
        const std::vector<cv::Point2f>& quad,
        int N);

    /**
     * @brief Apply perspective correction and return transformation matrices
     * 
     * Same as warpToSquare() but also returns the computed homography matrices
     * for coordinate transformations between original and warped images.
     * 
     * @param bgr Input BGR image (CV_8UC3)
     * @param quad 4 corner points of the quadrilateral region  
     * @param N Output square size (N×N pixels)
     * @return WarpResult containing warped image and transformation matrices
     * 
     * @throws cv::Exception on invalid homography computation
     * @note Useful when you need to transform coordinates between image spaces
     */
    WarpResult warpToSquareWithH(const cv::Mat& bgr,
        const std::vector<cv::Point2f>& quad,
        int N);

    /**
     * @brief Calculate polygon coverage as percentage of total image area
     * 
     * @param poly Polygon vertices in image coordinates
     * @param sz Image size (width × height)
     * @return Coverage percentage [0.0, 100.0], or 0.0 if polygon is invalid
     * 
     * @note Uses OpenCV's contourArea() for accurate area calculation
     * @note Handles non-convex polygons correctly
     * @note Returns 0.0 for polygons with fewer than 3 vertices
     */
    double polygonCoveragePercent(const std::vector<cv::Point2f>& poly,
        const cv::Size& sz);
}
