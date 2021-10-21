target_sources(libraryTests
        PRIVATE
        compressibleShockTubeTests.cpp
        compressibleFlowFluxTests.cpp
        compressibleFlowDiffusionTests.cpp
        compressibleFlowMmsSourceTests.cpp
        compressibleFlowAdvectionTests.cpp
        compressibleFlowSpeciesDiffusionTests.cpp
        )

add_subdirectory(fluxCalculator)
add_subdirectory(fieldFunctions)