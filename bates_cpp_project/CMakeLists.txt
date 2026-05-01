cmake_minimum_required(VERSION 3.18)
project(batespricer_cpp LANGUAGES CXX)

# ──────────────────────────────────────────────
# C++ Standard
# ──────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ──────────────────────────────────────────────
# Build-type defaults
# ──────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# ──────────────────────────────────────────────
# FetchContent: External Dependencies
# ──────────────────────────────────────────────
include(FetchContent)

# --- nlohmann/json (header-only JSON library) ---
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# --- Eigen3 (linear algebra, required by LBFGSpp) ---
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

# --- LBFGSpp (header-only L-BFGS-B optimizer) ---
FetchContent_Declare(
    lbfgspp
    GIT_REPOSITORY https://github.com/yixuan/LBFGSpp.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(lbfgspp)

# ──────────────────────────────────────────────
# OpenMP (optional, for Monte Carlo parallelism)
# ──────────────────────────────────────────────
find_package(OpenMP QUIET)

# ──────────────────────────────────────────────
# Executable
# ──────────────────────────────────────────────
add_executable(batespricer_cpp
    src/main.cpp
    src/analytics.cpp
    src/monte_carlo.cpp
    src/calibration.cpp
)

target_include_directories(batespricer_cpp PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${lbfgspp_SOURCE_DIR}/include
)

target_link_libraries(batespricer_cpp PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(batespricer_cpp PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(batespricer_cpp PRIVATE BATESPRICER_USE_OPENMP)
    message(STATUS "OpenMP found — Monte Carlo parallelism enabled")
else()
    message(STATUS "OpenMP not found — falling back to single-threaded Monte Carlo")
endif()

# ──────────────────────────────────────────────
# Compiler warnings (stricter in Debug)
# ──────────────────────────────────────────────
if(MSVC)
    target_compile_options(batespricer_cpp PRIVATE /W4)
else()
    target_compile_options(batespricer_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()
