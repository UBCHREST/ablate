


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

    # check for env var
    if (DEFINED ENV{OPENBLAS_INSTALL_PATH} AND NOT OPENBLAS_INSTALL_PATH)
        SET(OPENBLAS_INSTALL_PATH "$ENV{OPENBLAS_INSTALL_PATH}")
    endif ()
    if (DEFINED ENV{LAPACKE_INSTALL_PATH} AND NOT LAPACKE_INSTALL_PATH)
        SET(LAPACKE_INSTALL_PATH "$ENV{LAPACKE_INSTALL_PATH}")
    endif ()

    # if not defined, check for default petsc values
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

    # add tines
    install(TARGETS tines
            EXPORT ablateTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            )
    # check for openblas
    if (OPENBLAS_INSTALL_PATH)
        install(TARGETS openblas
                EXPORT ablateTargets
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
                INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
                )
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

    install(TARGETS tchem
            EXPORT ablateTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            )

elseif ()
    find_package(TChem REQUIRED)
endif ()

