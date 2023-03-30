
# Setup the framework library
add_library(testingResources "")
target_link_libraries(testingResources PUBLIC GTest::gtest GTest::gmock ablateLibrary PRIVATE chrestCompilerFlags)

target_sources(testingResources
        PRIVATE
        MpiTestFixture.cpp
        PetscTestViewer.cpp
        convergenceTester.cpp
        testRunEnvironment.cpp
        temporaryPath.cpp

        PUBLIC
        MpiTestFixture.hpp
        MpiTestParamFixture.hpp
        MpiTestEventListener.hpp
        PetscTestFixture.hpp
        PetscTestViewer.hpp
        PetscTestErrorChecker.hpp
        convergenceTester.hpp
        temporaryPath.hpp
        testRunEnvironment.hpp
        )

# Allow public access to the header files in the directory
target_include_directories(testingResources PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# when specified add a compile definition for mpi
if (COMPILE_MPI_COMMAND)
    target_compile_definitions(testingResources PUBLIC COMPILE_MPI_COMMAND=${COMPILE_MPI_COMMAND})
endif ()

set_property(TARGET testingResources PROPERTY CXX_STANDARD 20)