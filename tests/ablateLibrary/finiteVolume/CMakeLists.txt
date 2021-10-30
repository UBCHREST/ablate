target_sources(libraryTests
        PRIVATE
        compressibleShockTubeTests.cpp
        compressibleFlowDiffusionTests.cpp
        compressibleFlowMmsSourceTests.cpp
        compressibleFlowAdvectionTests.cpp
        compressibleFlowSpeciesDiffusionTests.cpp
        )

add_subdirectory(fluxCalculator)
add_subdirectory(fieldFunctions)
add_subdirectory(processes)