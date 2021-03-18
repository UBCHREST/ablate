add_executable(coreTests "")
target_link_libraries(coreTests PUBLIC gtest gmock ablateCore testingResources muparser)

target_sources(coreTests
        PRIVATE
        lowMachFlowTests.cpp
        incompressibleFlowFunctionSourceTests.cpp
        incompressibleFlowDynamicSourceTests.cpp
        incompressibleFlowParameterTests.cpp
        particleTests.cpp
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

# add testing support specific to ablateCore
add_subdirectory(support)

# Allow private access to the header files in the directory
target_include_directories(coreTests PRIVATE ${CMAKE_CURRENT_LIST_DIR})
