#pragma once
#include <chrono>

/**
 * Lightweight millisecond timer for profiling.
 */
class Timer {
public:
    using clock = std::chrono::high_resolution_clock;
    Timer() : t0(clock::now()) {}
    void reset() { t0 = clock::now(); }
    double ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }
private:
    clock::time_point t0;
};
