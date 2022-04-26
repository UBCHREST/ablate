target_sources(libraryTests
        PRIVATE
        compressibleShockTubeTests.cpp
        compressibleFlowEulerDiffusionTests.cpp
        compressibleFlowMmsSourceTests.cpp
        compressibleFlowSpeciesAdvectionTests.cpp
        compressibleFlowSpeciesDiffusionTests.cpp
        compressibleFlowEvAdvectionTests.cpp
        compressibleFlowEvDiffusionTests.cpp
        faceInterpolantTests.cpp
        )

add_subdirectory(fluxCalculator)
add_subdirectory(fieldFunctions)
add_subdirectory(processes)