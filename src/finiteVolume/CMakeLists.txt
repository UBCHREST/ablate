target_sources(ablateLibrary
        PRIVATE
        finiteVolumeSolver.cpp
        compressibleFlowFields.cpp
        compressibleFlowSolver.cpp
        faceInterpolant.cpp
        cellInterpolant.cpp
        turbulenceFlowFields.cpp

        PUBLIC
        finiteVolumeSolver.hpp
        compressibleFlowFields.hpp
        compressibleFlowSolver.hpp
        faceInterpolant.hpp
        cellInterpolant.hpp
        turbulenceFlowFields.hpp
        )

add_subdirectory(boundaryConditions)
add_subdirectory(fluxCalculator)
add_subdirectory(processes)
add_subdirectory(fieldFunctions)
add_subdirectory(stencils)