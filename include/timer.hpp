/**
 * @file timer.hpp
 * @brief High-precision timing utilities for performance profiling
 * 
 * Provides a simple, lightweight timer class for measuring execution time
 * of code sections during development and debugging.
 */
#pragma once
#include <chrono>

/**
 * @brief High-precision timer for performance measurement
 * 
 * Uses std::chrono::high_resolution_clock for accurate timing measurements.
 * Automatically starts timing on construction and provides millisecond precision.
 * 
 * @note Thread-safe for individual timer instances
 * @note Precision depends on system clock resolution (typically nanoseconds)
 * 
 * @example
 * ```cpp
 * Timer t;
 * // ... some expensive operation ...
 * std::cout << "Operation took: " << t.ms() << " ms\n";
 * 
 * t.reset();  
 * // ... another operation ...
 * std::cout << "Second operation: " << t.ms() << " ms\n";
 * ```
 */
class Timer {
public:
    /// @brief Clock type used for timing measurements
    using clock = std::chrono::high_resolution_clock;

    /**
     * @brief Constructor - automatically starts the timer
     * 
     * Records the current time as the starting reference point.
     */
    Timer() : t0(clock::now()) {}

    /**
     * @brief Reset the timer to current time
     * 
     * Sets a new starting reference point for subsequent measurements.
     */
    void reset() { t0 = clock::now(); }

    /**
     * @brief Get elapsed time in milliseconds
     * 
     * @return Time elapsed since construction or last reset() call
     * 
     * @note Returns double for sub-millisecond precision
     * @note Safe to call multiple times without affecting the timer
     */
    double ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }

private:
    /// @brief Timer start reference point
    clock::time_point t0;
};
