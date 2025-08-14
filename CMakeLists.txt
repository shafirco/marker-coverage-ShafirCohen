cmake_minimum_required(VERSION 3.20)
project(marker_coverage LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Core library (now with real sources)
add_library(mce_core
    src/color_segmenter.cpp
    src/geometry.cpp
    src/grid_detector.cpp 
)
target_include_directories(mce_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mce_core PUBLIC ${OpenCV_LIBS})

# CLI application
add_executable(marker_coverage src/main.cpp)
target_link_libraries(marker_coverage PRIVATE mce_core)

# Warnings
if (MSVC)
  target_compile_options(mce_core PRIVATE /W4 /permissive-)
  target_compile_options(marker_coverage PRIVATE /W4 /permissive-)
else()
  target_compile_options(mce_core PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(marker_coverage PRIVATE -Wall -Wextra -Wpedantic)
endif()
