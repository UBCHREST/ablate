# Allow public access to the header files in the directory
target_include_directories(ablateLibrary PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Tag the version file
configure_file (
        "version.h.in"
        "version.h"
)
target_include_directories(ablateLibrary PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

target_sources(ablateLibrary
        PUBLIC
        builder.cpp
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
add_subdirectory(chemistry)
add_subdirectory(radiation)
