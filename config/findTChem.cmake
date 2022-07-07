


# check to see if the Tines_DIR was specified, if not build
if (Tines::tines)
    message(STATUS "Found Tines::tines target")
elseif (NOT (DEFINED Tines_DIR|CACHE{Tines_DIR}|ENV{Tines_DIR}))
    message(STATUS "Tines_DIR not set.  Downloading and building Tines...")

    # Tines would like a blas/lapack library
    # Check if this was provided
    OPTION(OPENBLAS_INSTALL_PATH "Path to OpenBLAS installation for Tines")
    OPTION(LAPACKE_INSTALL_PATH "Path to LAPACKE installation for Tines")
    OPTION(TINES_ENABLE_MKL "Flag to enable MKL for Tines" OFF)
    if (NOT (OPENBLAS_INSTALL_PATH OR LAPACKE_INSTALL_PATH OR TINES_ENABLE_MKL))
        include(config/findBlasLapack.cmake)
        find_petsc_blas_lapack(OPENBLAS_INSTALL_PATH LAPACKE_INSTALL_PATH TINES_ENABLE_MKL)
    endif ()

    # tines expects the yaml-cpp target to be at yaml
    add_library(yaml ALIAS yaml-cpp)
    set(TINES_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(tines
            GIT_REPOSITORY https://github.com/UBCHREST/Tines.git
            GIT_TAG main
            SOURCE_SUBDIR src
            )
    FetchContent_MakeAvailable(tines)
    # make sure it is in the right name space
    if (TARGET tines)
        add_library(Tines::tines ALIAS tines)
    endif ()
elseif ()
    find_package(Tines REQUIRED)
endif ()

# check to see if the TChem_DIR was specified, if not build
if (NOT (DEFINED TChem_DIR|CACHE{TChem_DIR}|ENV{TChem_DIR}))
    message(STATUS "TChem_DIR not set.  Downloading and building TChem...")
    set(TCHEM_ENABLE_MAIN OFF CACHE BOOL "" FORCE)
    set(TCHEM_ENABLE_EXAMPLE OFF CACHE BOOL "" FORCE)
    set(TCHEM_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(tchem
            GIT_REPOSITORY https://github.com/UBCHREST/TChem.git
            GIT_TAG main
            SOURCE_SUBDIR src
            )
    FetchContent_MakeAvailable(tchem)
    # make sure it is in the right name space
    if (TARGET tchem)
        add_library(TChem::tchem ALIAS tchem)
    endif ()
elseif ()
    find_package(TChem REQUIRED)
endif ()