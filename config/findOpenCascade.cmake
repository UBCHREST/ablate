
### EGADS ###
FIND_LIBRARY(EGADS_LIBRARY
        NAMES
        egads
        HINTS
        ${CMAKE_FIND_ROOT_PATH}
        ${PETSc_LIBRARY_DIRS}
        PATHS
        ${CMAKE_FIND_ROOT_PATH}
        ${PETSc_LIBRARY_DIRS}
        )
if(NOT EGADS_LIBRARY)
    message(FATAL_ERROR "Cannot find EGADS library.  Please reconfigure PETSc with --download-egads flag." )
else()
    add_library(EGADS::EGADS UNKNOWN IMPORTED)
    set_target_properties(
            EGADS::EGADS
            PROPERTIES
            IMPORTED_LOCATION ${EGADS_LIBRARY})
endif()
target_link_libraries(ablateLibrary PUBLIC EGADS::EGADS)

###  OpenCASCADE ###
# OpenCASCADE should be built with petsc
FIND_PACKAGE(OpenCASCADE CONFIG REQUIRED HINTS "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/cmake/opencascade/")
target_link_libraries(ablateLibrary PUBLIC ${OpenCASCADE_LIBRARIES})
target_link_directories(ablateLibrary PUBLIC ${OpenCASCADE_LIBRARY_DIR})

### Load in threads ###
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(ablateLibrary PUBLIC Threads::Threads)

