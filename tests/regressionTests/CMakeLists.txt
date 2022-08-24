add_executable(regressionTests "")
target_link_libraries(regressionTests PUBLIC GTest::gtest ablateLibrary testingResources)

target_sources(regressionTests
        PRIVATE
        tests.cpp
        )

add_subdirectory(runners)

gtest_discover_tests(regressionTests
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES "VS_DEBUGGER_WORKING_DIRECTORY;\"${PROJECT_DIR}\";LABELS;regressionTests"
        )

copy_to_test_directory(${CMAKE_CURRENT_SOURCE_DIR}/outputs/)
copy_to_test_directory(${CMAKE_CURRENT_SOURCE_DIR}/inputs/)