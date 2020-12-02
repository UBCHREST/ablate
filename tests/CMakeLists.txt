add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")

add_executable(tests "")
target_link_libraries(tests gtest gmock gtest_main)

target_sources(tests
        PRIVATE
        demoTestA.cpp
        demotestB.cpp
        )
gtest_discover_tests(tests
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
        )