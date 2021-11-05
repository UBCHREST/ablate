target_sources(ablateLibrary
        PUBLIC
        finiteVolume.hpp
        finiteVolume.cpp
        compressibleFlowFields.hpp
        compressibleFlowFields.cpp
        compressibleFlow.cpp
        compressibleFlow.hpp
        reactingCompressibleFlow.hpp
        reactingCompressibleFlow.cpp
        )

add_subdirectory(boundaryConditions)
add_subdirectory(fluxCalculator)
add_subdirectory(processes)
add_subdirectory(fieldFunctions)