
# Setup the framework library
add_library(testingResources OBJECT "")
target_link_libraries(testingResources PUBLIC GTest::gtest GTest::gmock ablateLibrary PRIVATE chrestCompilerFlags)

target_sources(testingResources
        PRIVATE
        mpiTestFixture.cpp
        petscTestViewer.cpp
        convergenceTester.cpp
        testRunEnvironment.cpp
        temporaryPath.cpp
        integrationTest.cpp
        integrationRestartTest.cpp

        PUBLIC
        mpiTestFixture.hpp
        mpiTestParamFixture.hpp
        mpiTestEventListener.hpp
        petscTestFixture.hpp
        petscTestViewer.hpp
        petscTestErrorChecker.hpp
        convergenceTester.hpp
        temporaryPath.hpp
        testRunEnvironment.hpp
        integrationTest.hpp
        integrationRestartTest.hpp
        )

add_subdirectory(asserts)

# Allow public access to the header files in the directory
target_include_directories(testingResources PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# when specified add a compile definition for mpi
if (COMPILE_MPI_COMMAND)
    target_compile_definitions(testingResources PUBLIC COMPILE_MPI_COMMAND=${COMPILE_MPI_COMMAND})
endif ()