#include <iostream>

// Prints usage instructions for the CLI application
static void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [--debug] <image1> [image2 ...]\n";
}

int main(int argc, char** argv) {
    // If no image paths are provided, print usage and exit with code 2
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    // Placeholder: we will implement the actual logic in later steps.
    std::cout << "Marker Coverage Estimator (skeleton)\n";
    return 0;
}
