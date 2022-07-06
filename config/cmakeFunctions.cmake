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
