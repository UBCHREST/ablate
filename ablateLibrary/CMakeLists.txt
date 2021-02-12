# Setup the ablate core
add_library(ablateLibrary "")

# Include code
target_sources(ablateLibrary
        PUBLIC
        domain.cpp
        domain.h
        )

# Allow public access to the header files in the directory
target_include_directories(ablateLibrary PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Include the code in any subdirectory
add_subdirectory(flow)

# Keep a simple main statement
target_link_libraries(ablateLibrary PUBLIC ablateCore)