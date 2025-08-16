#pragma once
#include <chrono>

/// @brief Lightweight timer for profiling (milliseconds).
class Timer {
public:
    using clock = std::chrono::high_resolution_clock;

    Timer() : t0(clock::now()) {}

    /// @brief Reset the timer start point.
    void reset() { t0 = clock::now(); }

    /// @brief Elapsed time in milliseconds since last reset/creation.
    double ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }

private:
    clock::time_point t0;
};
