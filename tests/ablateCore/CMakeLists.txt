add_executable(coreTests "")
target_link_libraries(coreTests PUBLIC gtest gmock ablateCore testingResources)

target_sources(coreTests
        PRIVATE
        flowTests.cpp
        particleTests.cpp
        parametersTests.cpp
        main.cpp
        )

gtest_discover_tests(coreTests
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
        )

add_custom_command(
        TARGET coreTests
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/outputs/
        $<TARGET_FILE_DIR:coreTests>/outputs
)

