add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")

add_executable(tests "")
target_link_libraries(tests gtest gmock frameworkLibrary)

target_sources(tests
        PRIVATE
        MpiTestFixture.h
        MpiTestFixture.cpp
        demoTestA.cpp
        demoTestB.cpp
        main.cpp
        )
gtest_discover_tests(tests
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
        )

add_custom_command(
        TARGET tests
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/outputs/
        $<TARGET_FILE_DIR:tests>/outputs
)