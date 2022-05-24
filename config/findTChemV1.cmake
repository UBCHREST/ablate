
### TCHEM ###
# TCHEM should be built with petsc using the --download-tchem flag
FIND_LIBRARY(TCHEMV1_LIBRARY
        NAMES
        tchem
        HINTS
        ${CMAKE_FIND_ROOT_PATH}
        ${PETSc_LIBRARY_DIRS}
        PATHS
        ${CMAKE_FIND_ROOT_PATH}
        ${PETSc_LIBRARY_DIRS}
        )
if(NOT TCHEMV1_LIBRARY)
    message(FATAL_ERROR "Cannot find TChemV1 library.  Please reconfigure PETSc with --download-tchem flag." )
else()
    add_library(TChemV1::TChemV1 UNKNOWN IMPORTED)
    set_target_properties(
            TChemV1::TChemV1
            PROPERTIES
            IMPORTED_LOCATION ${TCHEMV1_LIBRARY})
endif()