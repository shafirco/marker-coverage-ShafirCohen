# Enable testing framework
enable_testing()

# Make sure OpenCV is available for tests
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui)

# Include headers from project
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add test executable
add_executable(mce_tests
    tests_basic.cpp
)

# Link either to your core library (if exists) or directly to OpenCV
# Example if you have a target mce_core:
# target_link_libraries(mce_tests PRIVATE mce_core ${OpenCV_LIBS})

# If no core library target, link OpenCV only:
target_link_libraries(mce_tests PRIVATE ${OpenCV_LIBS})

# Register the test
add_test(NAME BasicTest COMMAND mce_tests)
