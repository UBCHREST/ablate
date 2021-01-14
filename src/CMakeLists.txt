# Include PETSc and MPI
target_include_directories(ablateLibrary PUBLIC ${PETSc_INCLUDE_DIRS})
target_link_libraries(ablateLibrary PUBLIC ${PETSc_LIBRARIES})
target_link_directories (ablateLibrary PUBLIC ${PETSc_LIBRARY_DIRS})

# Include code
target_sources(ablateLibrary
        PUBLIC
        lowMachFlow.c
        lowMachFlow.h
        mesh.c
        mesh.h
        )

# Allow public access to the header files in the directory
target_include_directories(ablateLibrary PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Tag the version file
configure_file (
        "version.h.in"
        "version.h"
)