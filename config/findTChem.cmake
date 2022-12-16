# check to see if the Tines_DIR was specified, if not build
if (Tines::tines)
    message(STATUS "Found Tines::tines target")
elseif (NOT (DEFINED Tines_DIR|CACHE{Tines_DIR}|ENV{Tines_DIR}))
    message(STATUS "Tines_DIR not set.  Downloading and building Tines...")

    # tines expects the yaml-cpp target to be at yaml
    add_library(yaml ALIAS yaml-cpp)
    set(TINES_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(TINES_DENSE_LINEAR_ALGEBRA_WARNING OFF CACHE BOOL "" FORCE)
    set(TINES_CUDA_WARNING OFF CACHE BOOL "" FORCE)
    set(TINES_SUNDIALS_WARNING OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(tines
            GIT_REPOSITORY https://github.com/UBCHREST/Tines.git
            GIT_TAG main
            SOURCE_SUBDIR src
            )
    FetchContent_MakeAvailable(tines)

    # prevent error checks on included header files
    set_include_directories_as_system(tines)

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
    set(TCHEM_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(tchem
            GIT_REPOSITORY https://github.com/UBCHREST/TChem.git
            GIT_TAG main
            SOURCE_SUBDIR src
            )
    FetchContent_MakeAvailable(tchem)

    # prevent error checks on included header files
    set_include_directories_as_system(tchem)

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

