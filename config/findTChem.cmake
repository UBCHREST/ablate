
### TCHEM ###
# TCHEM should be built with petsc using the --download-tchem flag
FIND_LIBRARY(TCHEM_LIBRARY
        NAMES
        tchem
        HINTS
        ${CMAKE_FIND_ROOT_PATH}
        ${PETSc_LIBRARY_DIRS}
        PATHS
        ${CMAKE_FIND_ROOT_PATH}
        ${PETSc_LIBRARY_DIRS}
        )
if(NOT TCHEM_LIBRARY)
    message(FATAL_ERROR "Cannot find TChem library.  Please reconfigure PETSc with --download-tchem flag." )
else()
    add_library(TChem::TChem UNKNOWN IMPORTED)
    set_target_properties(
            TChem::TChem
            PROPERTIES
            IMPORTED_LOCATION ${TCHEM_LIBRARY})
endif()