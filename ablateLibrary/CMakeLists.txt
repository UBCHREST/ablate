# Setup any ablateLibrary specific dependencies
# turn off yaml-cpp options
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG yaml-cpp-0.6.3
)
FetchContent_MakeAvailable(yaml-cpp)

# install the library for generating xdmf files
set(DISABLE_PETSCXDMFGENERATOR_TESTS ON CACHE BOOL "" FORCE)
FetchContent_Declare(
        petscXdmfGenerator
        GIT_REPOSITORY https://github.com/UBCHREST/petscXdmfGenerator.git
        GIT_TAG v0.0.5
)
FetchContent_MakeAvailable(petscXdmfGenerator)

# Setup the ablate core
add_library(ablateLibrary SHARED "")

# Allow public access to the header files in the directory
target_include_directories(ablateLibrary PUBLIC ${CMAKE_CURRENT_LIST_DIR})

target_sources(ablateLibrary
        PUBLIC
        builder.cpp
        builder.hpp
        )

# Include the code in any subdirectory
add_subdirectory(flow)
add_subdirectory(utilities)
add_subdirectory(mesh)
add_subdirectory(parameters)
add_subdirectory(solve)
add_subdirectory(parser)
add_subdirectory(mathFunctions)
add_subdirectory(particles)
add_subdirectory(monitors)
add_subdirectory(eos)
add_subdirectory(environment)

# Keep a simple main statement
target_link_libraries(ablateLibrary PUBLIC ablateCore yaml-cpp muparser petscXdmfGeneratorLibrary)
