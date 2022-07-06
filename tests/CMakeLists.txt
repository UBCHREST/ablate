# Define a function to allow copying directories.  This function will automatically update files when they are changed
FUNCTION(copy_to_test_directory DIRECTORY)
    FILE(GLOB_RECURSE files ${DIRECTORY}/*)
    FOREACH(absFile ${files})
        file(RELATIVE_PATH relFile ${CMAKE_CURRENT_SOURCE_DIR} ${absFile})
        configure_file(${absFile} ${relFile} COPYONLY)
    ENDFOREACH()
ENDFUNCTION()

# Add each testing library
add_subdirectory(testingResources)
add_subdirectory(ablateLibrary)
add_subdirectory(integrationTests)

