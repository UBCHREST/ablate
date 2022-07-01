### XDMF Generation ###
# install the library for generating xdmf files
set(DISABLE_PETSCXDMFGENERATOR_TESTS ON CACHE BOOL "" FORCE)
if(NOT(DEFINED HDF5_ROOT|CACHE{HDF5_ROOT}|ENV{HDF5_ROOT}))
    message(VERBOSE "HDF5_ROOT not set.  Assuming HDF5_ROOT is in $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/")
    set(HDF5_ROOT "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/" STRING)
endif()
FetchContent_Declare(
        petscXdmfGenerator
        GIT_REPOSITORY https://github.com/UBCHREST/petscXdmfGenerator.git
        GIT_TAG v0.1.1
)
FetchContent_MakeAvailable(petscXdmfGenerator)