# Include code
target_sources(ablateLibrary
        PUBLIC
        particleInitializer.c
        particleInitializer.h
        particleCellInitializer.c
        particleCellInitializer.h
        particleBoxInitializer.c
        particleBoxInitializer.h
        )

# Allow public access to the header files in the directory
target_include_directories(ablateLibrary PUBLIC ${CMAKE_CURRENT_LIST_DIR})
