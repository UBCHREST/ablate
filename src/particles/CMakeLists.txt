# Include code
target_sources(ablateLibrary
        PUBLIC
        particles.c
        particles.h
        particleTracer.c
        particleTracer.h
        )

# Include the code in any subdirectory
add_subdirectory(initializers)

# Allow public access to the header files in the directory
target_include_directories(ablateLibrary PUBLIC ${CMAKE_CURRENT_LIST_DIR})
