cmake_minimum_required(VERSION 3.13)
project(CHREST VERSION 0.0.3)

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

# Check if we should enable testing options
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
    include(CTest)
endif()

# Setup and configure testing
if(BUILD_TESTING)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
endif()

# Load in the subdirectories
add_subdirectory(src)

# Add a target to allow linting check
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
    add_custom_target(
            format-check
            COMMAND ${PROJECT_SOURCE_DIR}/extern/run-clang-format/run-clang-format.py
            --style=file
            -r
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/tests
            USES_TERMINAL
    )
endif()

# Keep a simple main statement
add_executable(Framework main.c)
target_link_libraries(Framework PRIVATE frameworkLibrary)