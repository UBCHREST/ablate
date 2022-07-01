### XDMF Generation ###
# install the library for generating xdmf files
set(DISABLE_XDMFGENERATOR_TESTS ON CACHE BOOL "" FORCE)
if(NOT(DEFINED HDF5_ROOT|CACHE{HDF5_ROOT}|ENV{HDF5_ROOT}))
    message(VERBOSE "HDF5_ROOT not set.  Assuming HDF5_ROOT is in $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/")
    set(HDF5_ROOT "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/" STRING)
endif()

# Get the xdmfGeneratorLibrary dependency
IF(TARGET CHREST::xdmfGeneratorLibrary)
    message(STATUS "Found CHREST::xdmfGeneratorLibrary target")
ELSE()
    FetchContent_Declare(
            xdmfGeneratorLibrary
            GIT_REPOSITORY https://github.com/mmcgurn/petscXdmfGenerator.git
            GIT_TAG mcgurn/cmake-cleanup
    )
    FetchContent_MakeAvailable(xdmfGeneratorLibrary)
    # Put the library into CHREST namespace
    add_library(CHREST::xdmfGeneratorLibrary ALIAS xdmfGeneratorLibrary)
ENDIF()

