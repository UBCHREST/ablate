add_executable(libraryTests "")
target_link_libraries(libraryTests PUBLIC gtest gmock ablateLibrary testingResources)

target_sources(libraryTests
        PRIVATE
        main.cpp
        )

add_subdirectory(parameters)
add_subdirectory(parser)
add_subdirectory(mathFunctions)
add_subdirectory(environment)
add_subdirectory(flow)
add_subdirectory(particles)
add_subdirectory(eos)
add_subdirectory(mesh)
add_subdirectory(utilities)

gtest_discover_tests(libraryTests
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
        )

# Allow public access to the header files in the directory
target_include_directories(libraryTests PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Copy all input/output files needed for testing
add_custom_command(
        TARGET libraryTests
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/outputs/
        $<TARGET_FILE_DIR:libraryTests>/outputs
)

add_custom_command(
        TARGET libraryTests
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/inputs/
        $<TARGET_FILE_DIR:libraryTests>/inputs
)