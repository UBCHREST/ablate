add_executable(libraryTests "")
target_link_libraries(libraryTests PUBLIC gtest gmock ablateLibrary testingResources)

target_sources(libraryTests
        PRIVATE
        main.cpp
        )

add_subdirectory(parameters)
add_subdirectory(parser)

gtest_discover_tests(libraryTests
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
        )

# Allow public access to the header files in the directory
target_include_directories(libraryTests PUBLIC ${CMAKE_CURRENT_LIST_DIR})
