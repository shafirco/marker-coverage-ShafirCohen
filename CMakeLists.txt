cmake_minimum_required(VERSION 3.20)
project(marker_coverage LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Core library – removed for now to avoid empty target issues
# We'll reintroduce it once we add sources in later steps.

# CLI application
add_executable(marker_coverage src/main.cpp)

# Public headers directory (for later)
target_include_directories(marker_coverage PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Warnings
if (MSVC)
  target_compile_options(marker_coverage PRIVATE /W4 /permissive-)
else()
  target_compile_options(marker_coverage PRIVATE -Wall -Wextra -Wpedantic)
endif()
