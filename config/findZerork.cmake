
# check to see if the ZERORK_DIR was specified, if not build
if (NOT (DEFINED ENV{ZERORK_DIR}))
    message(STATUS "ZERORK_DIR not set.  Downloading and building zerork...")


    # CPU build
    if (NOT (DEFINED ENV{ABLATE_GPU}))
        message(STATUS "Builing zerork for CPUs.")

        FetchContent_Declare(zerork
            GIT_REPOSITORY https://github.com/LLNL/zero-rk.git
            GIT_TAG 3cf4001ed06f05aa6c98b78230e253a89545ddda  #git main branch for both cpu and CUDA
            )
        FetchContent_MakeAvailable(zerork)

        set_include_directories_as_system(zerork)

        install(TARGETS zerork_cfd_plugin zerork_vectormath ckconverter zerorkutilities zerork spify
            EXPORT ablateTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            )

        add_library(ZERORK::zerork_cfd_plugin ALIAS zerork_cfd_plugin)

    #Nvidia build
    elseif ($ENV{ABLATE_GPU} STREQUAL "CUDA")
        message(STATUS "Builing zerork for Nvidia GPUs with cuda.")

        FetchContent_Declare(zerork
                GIT_REPOSITORY https://github.com/LLNL/zero-rk.git
                GIT_TAG e972680adbbcb8deb4cfbe8bea8832a8c4124c7c  #git main branch for both cpu and CUDA
        )
        FetchContent_MakeAvailable(zerork)

        set_include_directories_as_system(zerork)

        install(TARGETS zerork_cfd_plugin_gpu zerork_cfd_plugin ckconverter zerork_vectormath zerorkutilities zerork zerork_cuda spify
            EXPORT ablateTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )

        add_library(ZERORK::zerork_cfd_plugin ALIAS zerork_cfd_plugin_gpu)

    elseif($ENV{ABLATE_GPU} STREQUAL "ROCM")
        message(STATUS "Builing zerork for AMD GPUs and hip.")

        FetchContent_Declare(zerork
            GIT_REPOSITORY https://github.com/LLNL/zero-rk.git
            GIT_TAG af50489b8ac6a24b025cf49f6e16399bdb5a8342  #points to zerork hip branch
            )
        FetchContent_MakeAvailable(zerork)

        set_include_directories_as_system(zerork)

        install(TARGETS zerork_cfd_plugin_gpu zerork_cfd_plugin ckconverter zerork_vectormath zerorkutilities zerork zerork_cuda spify
            EXPORT ablateTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            )

        add_library(ZERORK::zerork_cfd_plugin ALIAS zerork_cfd_plugin_gpu)

    endif()

elseif (DEFINED ENV{ZERORK_DIR})
    message(STATUS "Found ZERORK_DIR, using prebuilt zerork")


    add_library(zerork_cfd_plugin INTERFACE IMPORTED GLOBAL)
    target_include_directories(zerork_cfd_plugin INTERFACE "$ENV{ZERORK_DIR}/include")
    target_link_libraries(zerork_cfd_plugin INTERFACE "$ENV{ZERORK_DIR}/lib/libzerork_cfd_plugin.so")

    add_library(ZERORK::zerork_cfd_plugin ALIAS zerork_cfd_plugin)

endif ()

