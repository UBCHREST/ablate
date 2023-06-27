target_sources(ablateLibrary
        PRIVATE
        boundarySolver.cpp
        debugBoundarySolver.cpp
        PUBLIC
        boundarySolver.hpp
        boundaryProcess.hpp
        debugBoundarySolver.hpp
        )

add_subdirectory(lodi)
add_subdirectory(physics)
add_subdirectory(subModels)