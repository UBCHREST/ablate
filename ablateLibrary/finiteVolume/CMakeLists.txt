target_sources(ablateLibrary
        PUBLIC
        finiteVolume.hpp
        finiteVolume.cpp
        compressibleFlow.cpp
        compressibleFlow.hpp
        reactingCompressibleFlow.hpp
        reactingCompressibleFlow.cpp
        )

add_subdirectory(boundaryConditions)
add_subdirectory(fluxCalculator)
add_subdirectory(processes)
add_subdirectory(fieldFunctions)