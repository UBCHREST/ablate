target_sources(ablateLibrary
        PUBLIC
        finiteVolumeSolver.hpp
        finiteVolumeSolver.cpp
        compressibleFlowFields.hpp
        compressibleFlowFields.cpp
        compressibleFlowSolver.cpp
        compressibleFlowSolver.hpp
        reactingCompressibleFlowSolver.hpp
        reactingCompressibleFlowSolver.cpp
        )

add_subdirectory(boundaryConditions)
add_subdirectory(fluxCalculator)
add_subdirectory(processes)
add_subdirectory(fieldFunctions)
add_subdirectory(resources)