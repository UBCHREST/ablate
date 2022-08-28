target_sources(unitTests
        PRIVATE
        compressibleShockTubeTests.cpp
        compressibleFlowEulerDiffusionTests.cpp
        compressibleFlowMmsSourceTests.cpp
        compressibleFlowSpeciesAdvectionTests.cpp
        compressibleFlowSpeciesDiffusionTests.cpp
        compressibleFlowEvAdvectionTests.cpp
        compressibleFlowEvDiffusionTests.cpp
        faceInterpolantTests.cpp
        compressibleFlowlesDiffusionTests.cpp
        )

add_subdirectory(fluxCalculator)
add_subdirectory(fieldFunctions)
add_subdirectory(processes)
