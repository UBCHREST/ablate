
# Setup the framework library
add_library(testingResources "")
target_link_libraries(testingResources PUBLIC GTest::gtest GTest::gmock ablateLibrary)

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
