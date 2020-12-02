cmake_minimum_required(VERSION 3.13)
project(CHREST)

# Load the Required 3rd Party Libaries
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSc REQUIRED PETSc)
pkg_check_modules(MPI REQUIRED mpich)

# Set the c/c++ Standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Setup the framework library
add_library(frameworkLibrary "")

# Load in the subdirectories
add_subdirectory(src)

# Keep a simple main statement
add_executable(Framework main.c)
target_link_libraries(Framework PRIVATE frameworkLibrary)