# Create a unit test library that can be used by this unitTest executable as well as others
add_library(ablateUnitTestLibrary OBJECT "")
target_link_libraries(ablateUnitTestLibrary PUBLIC GTest::gtest GTest::gmock testingResources CHREST::cppParserTestLibrary PRIVATE chrestCompilerFlags)

add_subdirectory(parameters)
add_subdirectory(parser)
add_subdirectory(mathFunctions)
add_subdirectory(environment)
add_subdirectory(finiteVolume)
add_subdirectory(finiteElement)
add_subdirectory(particles)
add_subdirectory(eos)
add_subdirectory(domain)
add_subdirectory(utilities)
add_subdirectory(monitors)
add_subdirectory(io)
add_subdirectory(boundarySolver)
add_subdirectory(radiation)
add_subdirectory(levelSet)

# Allow public access to the header files in the directory
target_include_directories(ablateUnitTestLibrary PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Create the main unit test executable
add_executable(unitTests "")
target_link_libraries(unitTests PUBLIC ablateUnitTestLibrary testingResources PRIVATE chrestCompilerFlags)

target_sources(unitTests
        PRIVATE
        main.cpp
)

gtest_discover_tests(unitTests
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES "VS_DEBUGGER_WORKING_DIRECTORY;\"${PROJECT_DIR}\";LABELS;unitTests"
        DISCOVERY_TIMEOUT 360
        )

## Copy all input/output files needed for testing
copy_to_test_directory(${CMAKE_CURRENT_SOURCE_DIR}/outputs)
copy_to_test_directory(${CMAKE_CURRENT_SOURCE_DIR}/inputs)
