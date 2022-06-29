# Check to see if kokkos was built in with

# check to see if the Kokkos_DIR was specified, if not build
if (NOT (DEFINED Kokkos_DIR|CACHE{Kokkos_DIR}|ENV{Kokkos_DIR}))
    message(STATUS "Kokkos_DIR not set.  Downloading and building Kokkos...")
    FetchContent_Declare(kokkos
            GIT_REPOSITORY https://github.com/kokkos/kokkos.git
            GIT_TAG 3.6.01   # it's much better to use a specific Git revision or Git tag for reproducibility
            )
    FetchContent_MakeAvailable(kokkos)
elseif ()
    find_package(Kokkos REQUIRED)
endif ()

# check to see if the Tines_DIR was specified, if not build
if (NOT (DEFINED Tines_DIR|CACHE{Tines_DIR}|ENV{Tines_DIR}))
    message(STATUS "Tines_DIR not set.  Downloading and building Tines...")
    FetchContent_Declare(tines
            GIT_REPOSITORY https://github.com/UBCHREST/Tines.git
            GIT_TAG main
            SOURCE_SUBDIR src
            )
    FetchContent_MakeAvailable(tines)
elseif ()
    find_package(Tines REQUIRED)
endif ()

# check to see if the TChem_DIR was specified, if not build
#if (NOT (DEFINED TChem_DIR|CACHE{TChem_DIR}|ENV{TChem_DIR}))
#    message(STATUS "TChem_DIR not set.  Downloading and building TChem...")
#    FetchContent_Declare(tchem
#            GIT_REPOSITORY https://github.com/sandialabs/TChem.git
#            GIT_TAG main
#            SOURCE_SUBDIR src
#
#            )
#    FetchContent_MakeAvailable(tchem)
#elseif ()
    find_package(TChem REQUIRED)
#endif ()