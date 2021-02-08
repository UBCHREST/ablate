# Include PETSc and MPI
target_include_directories(ablateLibrary PUBLIC ${PETSc_INCLUDE_DIRS})
target_link_libraries(ablateLibrary PUBLIC ${PETSc_LIBRARIES})
target_link_directories (ablateLibrary PUBLIC ${PETSc_LIBRARY_DIRS})

# Include code
target_sources(ablateLibrary
        PUBLIC
        mesh.c
        mesh.h
        parameters.c
        parameters.h
        )

# Allow public access to the header files in the directory
target_include_directories(ablateLibrary PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Include the code in any subdirectory
add_subdirectory(flow)
add_subdirectory(particles)

# Tag the version file
configure_file (
        "version.h.in"
        "version.h"
)