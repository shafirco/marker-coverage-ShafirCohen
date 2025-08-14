#include <iostream>
#include <opencv2/opencv.hpp>

// Prints usage instructions for the CLI application
static void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [--debug] <image1> [image2 ...]\n";
}

int main(int argc, char** argv) {
    // Print OpenCV version to verify linking
    std::cout << "OpenCV version: " << CV_VERSION << "\n";

    // If no image paths are provided, print usage and exit with code 2
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    std::cout << "Marker Coverage Estimator (OpenCV linked)\n";
    return 0;
}
