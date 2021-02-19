# Setup the ablate core
add_library(ablateLibrary SHARED "")

# Allow public access to the header files in the directory
target_include_directories(ablateLibrary PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Include the code in any subdirectory
add_subdirectory(flow)
add_subdirectory(utilities)
add_subdirectory(mesh)
add_subdirectory(parameters)
add_subdirectory(solve)
add_subdirectory(parser)

# Keep a simple main statement
target_link_libraries(ablateLibrary PUBLIC ablateCore)
