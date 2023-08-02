# add a custom command to print the current version
add_custom_target(
        print-version
        COMMAND echo ${CMAKE_PROJECT_VERSION}
        USES_TERMINAL
)