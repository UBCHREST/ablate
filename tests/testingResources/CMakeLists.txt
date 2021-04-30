
# Setup the framework library
add_library(testingResources "")
target_link_libraries(testingResources PUBLIC gtest gmock ablateCore)

target_sources(testingResources
        PUBLIC
        MpiTestFixture.hpp
        MpiTestFixture.cpp
        MpiTestParamFixture.hpp
        MpiTestEventListener.hpp
        )

# Allow public access to the header files in the directory
target_include_directories(testingResources PUBLIC ${CMAKE_CURRENT_LIST_DIR})
