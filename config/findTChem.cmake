# Check to see if kokkos was built in with
# check to see if the Tines_DIR was specified, if not build
if (NOT (DEFINED Tines_DIR|CACHE{Tines_DIR}|ENV{Tines_DIR}))
    message(STATUS "Tines_DIR not set.  Downloading and building Tines...")
    FetchContent_Declare(tines
            GIT_REPOSITORY https://github.com/UBCHREST/Tines.git
            GIT_TAG main
            SOURCE_SUBDIR src
            )
    FetchContent_MakeAvailable(tines)
    # make sure it is in the right name space
    if(TARGET tines)
        add_library(Tines::tines ALIAS tines)
    endif()
elseif ()
    find_package(Tines REQUIRED)
endif ()



# check to see if the TChem_DIR was specified, if not build
if (NOT (DEFINED TChem_DIR|CACHE{TChem_DIR}|ENV{TChem_DIR}))
    message(STATUS "TChem_DIR not set.  Downloading and building TChem...")
    set(TCHEM_ENABLE_MAIN OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(tchem
            GIT_REPOSITORY https://github.com/UBCHREST/TChem.git
            GIT_TAG main
            SOURCE_SUBDIR src

            )
    FetchContent_MakeAvailable(tchem)

    FetchContent_MakeAvailable(tchem)
    # make sure it is in the right name space
    if(TARGET tchem)
        add_library(TChem::tchem ALIAS tchem)
    endif()
elseif ()
    find_package(TChem REQUIRED)
endif ()