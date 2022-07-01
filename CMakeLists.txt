cmake_minimum_required(VERSION 3.21)
find_package(PkgConfig REQUIRED)

# Check if a C compiler is explicitly stated
if (NOT DEFINED CMAKE_C_COMPILER)
    # Set the compilers based upon the PETSc package
    pkg_get_variable(PETSC_C_COMPILER PETSc ccompiler)
    set(CMAKE_C_COMPILER ${PETSC_C_COMPILER})
endif ()

# Check if a CXX compiler is explicitly stated
if (NOT DEFINED CMAKE_CXX_COMPILER)
    # Set the compilers based upon the PETSc package
    pkg_get_variable(PETSC_CXX_COMPILER PETSc cxxcompiler)
    set(CMAKE_CXX_COMPILER ${PETSC_CXX_COMPILER})
endif ()

# Set the project details
project(ablateLibrary VERSION 0.8.29)

# Load the Required 3rd Party Libaries
pkg_check_modules(PETSc REQUIRED IMPORTED_TARGET GLOBAL PETSc)

# Set the c/c++ Standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# FetContent will allow downloading of other libraries
include(FetchContent)

# Check if we should enable testing options
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
    include(CTest)
endif ()

# Create the library for ablate
add_library(ablateLibrary SHARED "")

# Load in the subdirectories for the ablate library
add_subdirectory(src)

# find and load required libraries
include(config/findYamlCpp.cmake)
include(config/findCppParser.cmake)
include(config/findMuParser.cmake)
include(config/findXdmfGenerator.cmake)
include(config/findTChem.cmake)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
include(config/findOpenCascade.cmake)
include(config/findJson.cmake)
include(config/findTensorFlow.cmake)

# specify in the required libaries
target_link_libraries(ablateLibrary
        PUBLIC
        PkgConfig::PETSc
        CHREST::cppParserLibrary
        yaml-cpp
        muparser
        CHREST::xdmfGeneratorLibrary
        TChem::TChem
        Threads::Threads
        nlohmann_json::nlohmann_json
        PRIVATE
        chrestCompilerFlags)

# Set the ablate library requirements
set_property(TARGET ablateLibrary PROPERTY CXX_EXTENSIONS ON)
set_property(TARGET ablateLibrary PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(TARGET ablateLibrary PROPERTY CXX_STANDARD 17)

# Setup and configure testing
if (BUILD_TESTING)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)

    # Include the format command only if we are building tests
    include(clangFormatter.cmake)
endif ()

# keep a separate main statement
add_executable(ablate main.cpp)
target_link_libraries(ablate PRIVATE ablateLibrary)

# include system specific hacks
include(systemHacks.cmake)
