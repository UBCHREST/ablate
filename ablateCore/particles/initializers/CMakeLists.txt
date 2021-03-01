# Include code
target_sources(ablateCore
        PUBLIC
        particleInitializer.c
        particleInitializer.h
        particleCellInitializer.c
        particleCellInitializer.h
        particleBoxInitializer.c
        particleBoxInitializer.h
        )

# Allow public access to the header files in the directory
target_include_directories(ablateCore PUBLIC ${CMAKE_CURRENT_LIST_DIR})
