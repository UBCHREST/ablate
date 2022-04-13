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
        fvSupport.h
        fvSupport.c
        )

add_subdirectory(boundaryConditions)
add_subdirectory(fluxCalculator)
add_subdirectory(processes)
add_subdirectory(fieldFunctions)