add_executable(regressionTests "")
target_link_libraries(regressionTests PUBLIC GTest::gtest ablateLibrary testingResources PRIVATE chrestCompilerFlags)

target_sources(regressionTests
        PRIVATE
        main.cpp
        )

gtest_discover_tests(regressionTests
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        # add a custom default timeout of three hours for the regression tests
        PROPERTIES "VS_DEBUGGER_WORKING_DIRECTORY;\"${PROJECT_DIR}\";TIMEOUT;21600;LABELS;regressionTests;"
        DISCOVERY_TIMEOUT 360
        )

copy_to_test_directory(${CMAKE_CURRENT_SOURCE_DIR}/inputs/)