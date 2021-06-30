target_sources(libraryTests
        PRIVATE
        feFlowFunctionSourceTests.cpp
        feFlowDynamicSourceTests.cpp
        compressibleShockTubeTests.cpp
        compressibleFlowFluxTests.cpp
        compressibleFlowDiffusionTests.cpp
        compressibleFlowMmsSourceTests.cpp
        compressibleFlowAdvectionTests.cpp
        )

add_subdirectory(fluxDifferencer)