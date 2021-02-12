# Include code
target_sources(ablateCore
        PUBLIC
        particles.c
        particles.h
        particleTracer.c
        particleTracer.h
        )

# Include the code in any subdirectory
add_subdirectory(initializers)

# Allow public access to the header files in the directory
target_include_directories(ablateCore PUBLIC ${CMAKE_CURRENT_LIST_DIR})
