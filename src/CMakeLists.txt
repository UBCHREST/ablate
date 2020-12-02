# Include PETSc and MPI
target_include_directories(frameworkLibrary PUBLIC ${PETSc_INCLUDE_DIRS} ${PETSc_INCLUDE_DIRS})
target_link_libraries(frameworkLibrary PUBLIC ${PETSc_LIBRARIES} ${MPI_LIBRARIES})
target_link_directories (frameworkLibrary PUBLIC ${MPI_LIBRARY_DIRS} ${MPI_LIBRARY_DIRS})

# Include code
target_sources(frameworkLibrary
        PRIVATE
        demo.c
        demo.h
        )

# Allow public access to the header files in the directory
target_include_directories(frameworkLibrary PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Tag the version file
configure_file (
        "version.h.in"
        "version.h"
)