# Allow public access to the header files in the directory
target_include_directories(ablateLibrary PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# Tag the version file
configure_file (
        "version.h.in"
        "version.h"
)
target_include_directories(ablateLibrary PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_sources(ablateLibrary
        PRIVATE
        builder.cpp
        PUBLIC
        builder.hpp
        )

# Include the code in any subdirectory
add_subdirectory(finiteVolume)
add_subdirectory(finiteElement)
add_subdirectory(boundarySolver)
add_subdirectory(utilities)
add_subdirectory(domain)
add_subdirectory(parameters)
add_subdirectory(solver)
add_subdirectory(mathFunctions)
add_subdirectory(particles)
add_subdirectory(monitors)
add_subdirectory(eos)
add_subdirectory(environment)
add_subdirectory(io)
add_subdirectory(levelSet)
add_subdirectory(radiation)
