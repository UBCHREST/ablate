add_executable(integrationTests "")
target_link_libraries(integrationTests PUBLIC GTest::gtest ablateLibrary testingResources PRIVATE chrestCompilerFlags)

target_sources(integrationTests
        PRIVATE
        main.cpp
        )

add_subdirectory(runners)

gtest_discover_tests(integrationTests
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES "VS_DEBUGGER_WORKING_DIRECTORY;\"${PROJECT_DIR}\";LABELS;integrationTests"
        DISCOVERY_TIMEOUT 360
        )

copy_to_test_directory(${CMAKE_CURRENT_SOURCE_DIR}/outputs/)
copy_to_test_directory(${CMAKE_CURRENT_SOURCE_DIR}/inputs/)