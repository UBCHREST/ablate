### Tensorflow ###
pkg_check_modules(TensorFlow QUIET tensorflow)
if(TensorFlow_FOUND)
    target_link_directories(${TensorFlow_LIBRARY_DIRS})
    include_directories(${TensorFlow_INCLUDE_DIRS})
    add_compile_definitions(${TensorFlow_CFLAGS_OTHER})
    message(STATUS "Tensorflow library has been found using package modules")
    target_link_libraries(ablateLibrary PUBLIC TensorFlow)
    target_compile_definitions(ablateLibrary PUBLIC WITH_TENSORFLOW)
elseif(DEFINED ENV{TENSORFLOW_DIR} AND (NOT $ENV{TENSORFLOW_DIR} STREQUAL ""))
    if(NOT DEFINED TENSORFLOW_DIR)
        set(TENSORFLOW_DIR $ENV{TENSORFLOW_DIR})
    endif()

    # manually specify the tensor flow directory with TENSORFLOW_DIR
    FIND_LIBRARY(TensorFlowLibrary
            NAMES
            tensorflow
            HINTS
            ${TENSORFLOW_DIR}
            ${TENSORFLOW_DIR}/lib
            )
    if(NOT TensorFlowLibrary)
        message(FATAL_ERROR "Cannot find TensorFlow library at " ${TENSORFLOW_DIR} )
    else()
        add_library(TensorFlow UNKNOWN IMPORTED)
        set_target_properties(
                TensorFlow
                PROPERTIES
                IMPORTED_LOCATION ${TensorFlowLibrary})

        # Build a list of include directories
        list(APPEND TENSORFLOW_INCLUDE_DIRS ${TENSORFLOW_DIR}/include)
        list(APPEND TENSORFLOW_INCLUDE_DIRS ${TENSORFLOW_DIR}/include/external/local_tsl/)

        set_target_properties(
                TensorFlow
                PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${TENSORFLOW_INCLUDE_DIRS}" )
        message(STATUS "Tensorflow library has been found at the specified TENSORFLOW_DIR location " ${TENSORFLOW_DIR})
        target_compile_definitions(ablateLibrary PUBLIC WITH_TENSORFLOW)
        target_link_libraries(ablateLibrary PUBLIC TensorFlow)
    endif()
else()
    message(STATUS "Tensorflow could not be located and will be skipped. It can be specified with TENSORFLOW_DIR CMAKE variable or env variable ")
endif()
