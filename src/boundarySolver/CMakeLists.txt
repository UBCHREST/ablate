target_sources(ablateLibrary
        PRIVATE
        boundarySolver.cpp
        PUBLIC
        boundarySolver.hpp
        boundaryProcess.hpp
        )

add_subdirectory(lodi)
add_subdirectory(physics)