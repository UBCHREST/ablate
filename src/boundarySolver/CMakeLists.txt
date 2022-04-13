target_sources(ablateLibrary
        PUBLIC
        boundarySolver.hpp
        boundarySolver.cpp
        boundaryProcess.hpp
        )

add_subdirectory(lodi)
add_subdirectory(physics)