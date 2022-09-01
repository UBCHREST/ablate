target_sources(ablateLibrary
        PRIVATE
        finiteVolumeSolver.cpp
        compressibleFlowFields.cpp
        compressibleFlowSolver.cpp
        reactingCompressibleFlowSolver.cpp
        faceInterpolant.cpp
        cellInterpolant.cpp
                        lesCompressibleFlowSolver.cpp

        PUBLIC
        finiteVolumeSolver.hpp
        compressibleFlowFields.hpp
        compressibleFlowSolver.hpp
        reactingCompressibleFlowSolver.hpp
        faceInterpolant.hpp
        cellInterpolant.hpp
                lesCompressibleFlowSolver.hpp

        )

add_subdirectory(boundaryConditions)
add_subdirectory(fluxCalculator)
add_subdirectory(processes)
add_subdirectory(fieldFunctions)
add_subdirectory(stencils)
