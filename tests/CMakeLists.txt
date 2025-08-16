cmake_minimum_required(VERSION 3.21)

# Single basic test
add_executable(mce_tests
    tests_basic.cpp
)

# Link against the core library and OpenCV
target_link_libraries(mce_tests PRIVATE mce_core ${OpenCV_LIBS})

# Include project and OpenCV headers
target_include_directories(mce_tests PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

# Enable running the tests through CTest (optional)
enable_testing()
add_test(NAME mce_tests COMMAND mce_tests)
