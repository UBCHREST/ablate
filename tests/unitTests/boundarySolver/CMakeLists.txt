target_sources(ablateUnitTestLibrary
        PRIVATE
        boundarySolverPointTests.cpp
        boundarySolverDistributedTests.cpp
        boundarySolverFluxTests.cpp
        )

add_subdirectory(lodi)
add_subdirectory(physics)