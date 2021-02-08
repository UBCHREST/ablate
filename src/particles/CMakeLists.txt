
# Include code
target_sources(ablateLibrary
        PUBLIC
        particles.c
        particles.h
        )

# Allow public access to the header files in the directory
target_include_directories(ablateLibrary PUBLIC ${CMAKE_CURRENT_LIST_DIR})
