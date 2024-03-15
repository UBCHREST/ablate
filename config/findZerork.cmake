
# check to see if the ZERORK_DIR was specified, if not build
if (NOT (DEFINED ENV{ZERORK_DIR}))
    message(STATUS "ZERORK_DIR not set.  Downloading and building zerork...")

    FetchContent_Declare(zerork
            GIT_REPOSITORY https://github.com/LLNL/zero-rk.git
            GIT_TAG dabf3257c1598104099b08048d95099200fc795f
            )
    FetchContent_MakeAvailable(zerork)

    set_include_directories_as_system(zerork)

    install(TARGETS zerork_cfd_plugin ckconverter zerork_vectormath zerorkutilities zerork spify
            EXPORT ablateTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            )


elseif (DEFINED ENV{ZERORK_DIR})
    message(STATUS "Found ZERORK_DIR, using prebuilt zerork")


    add_library(zerork INTERFACE IMPORTED GLOBAL)
    target_include_directories(zerork INTERFACE "$ENV{ZERORK_DIR}/include")
    target_link_libraries(zerork INTERFACE "$ENV{ZERORK_DIR}/lib/libzerork_cfd_plugin.so")

    if (TARGET zerork)
        add_library(zerork_cfd_plugin ALIAS zerork)
    endif ()

elseif ()
    find_package(zerork REQUIRED)
endif ()

