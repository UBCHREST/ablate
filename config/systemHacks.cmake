# Include system specific hacks
if("${APPLE}" AND (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "arm64") AND (${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang"))
    # check for homebrew gfortran and get path for libstdc++.dylib
    execute_process(COMMAND gfortran --print-file-name=libstdc++.dylib OUTPUT_VARIABLE LIBSTDCPP_PATH)

    # convert to an absolute path and get the directory
    get_filename_component(LIBSTDCPP_PATH ${LIBSTDCPP_PATH} ABSOLUTE)
    get_filename_component(LIBSTDCPP_PATH ${LIBSTDCPP_PATH} DIRECTORY)

    target_link_directories(ablateLibrary PUBLIC ${LIBSTDCPP_PATH})
endif()