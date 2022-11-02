add_executable(unitTests "")
target_link_libraries(unitTests PUBLIC GTest::gtest GTest::gmock ablateLibrary testingResources CHREST::cppParserTestLibrary)

target_sources(unitTests
        PRIVATE
        main.cpp
        )

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
add_subdirectory(chemistry)
add_subdirectory(radiation)
add_subdirectory(solver)

gtest_discover_tests(unitTests
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES "VS_DEBUGGER_WORKING_DIRECTORY;\"${PROJECT_DIR}\";LABELS;unitTests"
        )

# Allow public access to the header files in the directory
target_include_directories(unitTests PUBLIC ${CMAKE_CURRENT_LIST_DIR})

## Copy all input/output files needed for testing
copy_to_test_directory(${CMAKE_CURRENT_SOURCE_DIR}/outputs)
copy_to_test_directory(${CMAKE_CURRENT_SOURCE_DIR}/inputs)