# ---   CMAKE LISTS   --- #
# ----------------------- #

# CMake project
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(vl3dpp LANGUAGES CXX)

# Configure CMake
message("Configuring CMake ...")
include(cmake/CMakeConfig.cmake)
message("CMake configured!\n\n")

# Handle PyBind11
message("Handling PyBind11 ...")
include(cmake/CMakePyBind11.cmake)
message("PyBind11 handled!")

# Handle sources
message("Handling sources ...")
include(cmake/CMakeSources.cmake)
message("Sources handled!\n\n")

# Load libraries
message("Loading libraries ...")
include(cmake/CMakeLibraries.cmake)
message("Libraries loaded!\n\n")

# Build library
message("Building VL3D++ ...")
include(cmake/CMakeBuildLib.cmake)
message("VL3D++ built!\n\n")

# CMake finished
message("${PROJECT_NAME} CMake finished!")
