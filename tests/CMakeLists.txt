# Download google test
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main
)

FetchContent_MakeAvailable(googletest)

# Add each testing library
add_subdirectory(testingResources)
add_subdirectory(ablateLibrary)
add_subdirectory(integrationTests)

