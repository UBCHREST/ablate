cmake_minimum_required(VERSION 3.14)

# Set the compilers based upon the PETSc package
find_package(PkgConfig REQUIRED)
pkg_get_variable(PETSC_C_COMPILER PETSc ccompiler)
set(CMAKE_C_COMPILER ${PETSC_C_COMPILER})
pkg_get_variable(PETSC_CXX_COMPILER PETSc cxxcompiler)
set(CMAKE_CXX_COMPILER ${PETSC_CXX_COMPILER})

# Set the project details
project(ablateLibrary VERSION 0.3.5)

# Load the Required 3rd Party Libaries
pkg_check_modules(PETSc REQUIRED PETSc)

# Set the c/c++ Standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# FetContent will allow downloading of other libraries
include(FetchContent)

# Check if we should enable testing options
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
    include(CTest)
endif()

# Load in the subdirectories
add_subdirectory(ablateCore)
add_subdirectory(ablateLibrary)

# Set the ablate library requirements
set_property(TARGET ablateLibrary PROPERTY CXX_EXTENSIONS ON)
set_property(TARGET ablateLibrary PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(TARGET ablateLibrary PROPERTY CXX_STANDARD 17)

# Setup and configure testing
if(BUILD_TESTING)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)

    # Include the format command only if we are building tests
    include(clangFormatter.cmake)
endif()

# keep a separate main statement
add_executable(ablate main.cpp)
target_link_libraries(ablate PRIVATE ablateLibrary)
