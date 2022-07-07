# march over each target property to update path to includes
FUNCTION(update_header_paths_for_install target buildRoot installRoot)
    get_target_property(interfaceListOrg ${target} INTERFACE_SOURCES)
    foreach (interfaceItem ${interfaceListOrg})
        # Replace this hard code value with relative values
        file(RELATIVE_PATH relativeHeaderPath ${buildRoot} ${interfaceItem})
        # add the lists back
        list(APPEND interfaceListUpdated "$<BUILD_INTERFACE:${buildRoot}/${relativeHeaderPath}>")
        list(APPEND interfaceListUpdated "$<INSTALL_INTERFACE:${installRoot}/${relativeHeaderPath}>")
    endforeach ()
    set_property(TARGET ${target} PROPERTY INTERFACE_SOURCES ${interfaceListUpdated})
ENDFUNCTION()

# find petsc argument
FUNCTION(get_petsc_argument regexArgument argument)
    file(STRINGS "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/petsc/conf/configure-hash" configureLines REGEX "${regexArgument}")
    if (configureLines)
        string(REGEX MATCH ${regexArgument} foundArgument ${configureLines})
        if (${CMAKE_MATCH_COUNT} GREATER 0)
            set(${argument} ${CMAKE_MATCH_1} PARENT_SCOPE)
            return()
        endif ()
    endif ()
ENDFUNCTION()

FUNCTION(check_petsc_argument regexArgument found)
    file(STRINGS "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/petsc/conf/configure-hash" configureLines REGEX "${regexArgument}")
    if (configureLines)
        set(${found} TRUE PARENT_SCOPE)
    else ()
        set(${found} FALSE PARENT_SCOPE)
    endif()

ENDFUNCTION()