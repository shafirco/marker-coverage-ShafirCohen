cmake_minimum_required(VERSION 3.20)
project(marker_coverage LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# CLI application
add_executable(marker_coverage src/main.cpp)

# Link OpenCV
target_link_libraries(marker_coverage PRIVATE ${OpenCV_LIBS})

# Include OpenCV headers
target_include_directories(marker_coverage PRIVATE ${OpenCV_INCLUDE_DIRS})

# Warnings
if (MSVC)
  target_compile_options(marker_coverage PRIVATE /W4 /permissive-)
else()
  target_compile_options(marker_coverage PRIVATE -Wall -Wextra -Wpedantic)
endif()
