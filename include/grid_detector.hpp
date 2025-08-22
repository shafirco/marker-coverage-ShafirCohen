/**
 * @file grid_detector.hpp
 * @brief Grid structure validation for 3x3 marker patterns
 * 
 * Validates that detected quadrilaterals contain proper 3x3 grid structure
 * by analyzing seam positions and cell content in warped marker images.
 */
#pragma once
#include <opencv2/opencv.hpp>

namespace grid {

    /**
     * @brief Grid seam detection results
     * 
     * Stores the positions of vertical and horizontal grid lines that should
     * separate the 3x3 cells at approximately 1/3 and 2/3 positions.
     */
    struct Seams {
        /// @brief First vertical seam position (expected ~1/3 from left)
        int cx1 = -1;
        
        /// @brief Second vertical seam position (expected ~2/3 from left)  
        int cx2 = -1;
        
        /// @brief First horizontal seam position (expected ~1/3 from top)
        int cy1 = -1;
        
        /// @brief Second horizontal seam position (expected ~2/3 from top)
        int cy2 = -1;
        
        /// @brief True if both vertical and horizontal seams found within tolerance
        bool ok = false;
    };

    /**
     * @brief Grid cell validation results  
     * 
     * Contains coverage fractions for each of the 9 cells in the 3x3 grid
     * and overall validation status.
     */
    struct CellsReport {
        /// @brief Fraction of allowed-color pixels in each cell [0.0-1.0]
        /// @note Indexed as frac[row][col] where (0,0) is top-left
        double frac[3][3];
        
        /// @brief True if all 9 cells meet minimum coverage threshold
        bool ok = false;
    };

    /**
     * @brief Detect grid seam positions in a binary mask
     * 
     * Analyzes column and row sums to find minimum points that indicate
     * the grid lines separating the 3x3 cells. Searches for seams at
     * approximately 1/3 and 2/3 positions with tolerance.
     * 
     * @param mask Binary mask (CV_8UC1) of warped marker region
     * @return Seams structure with detected positions and validation status
     * 
     * @note Uses flexible search ranges (±1/6) around ideal 1/3 and 2/3 positions
     * @note Requires adequate spacing between detected seams to avoid false positives
     */
    Seams checkGridSeams(const cv::Mat& mask);

    /**
     * @brief Validate coverage of each cell in the 3x3 grid
     * 
     * Divides the warped mask into 9 equal regions and calculates the
     * fraction of allowed-color pixels in each cell. All cells must
     * meet the minimum threshold for validation to pass.
     * 
     * @param mask Square binary mask (CV_8UC1) of warped marker region
     * @param minFraction Minimum required fraction of allowed pixels per cell [0.0-1.0]
     * @return CellsReport with per-cell fractions and overall validation status
     * 
     * @note Expects square mask (rows == cols) from perspective correction
     * @note Edge cells may be slightly larger due to integer division remainder
     * @warning Returns all zeros if mask is empty or non-square
     */
    CellsReport checkGridCells(const cv::Mat& mask, double minFraction);
}
