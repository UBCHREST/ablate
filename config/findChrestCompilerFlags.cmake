# load in the build/compiler standards
FetchContent_Declare(
        chrestCompilerFlags
        GIT_REPOSITORY https://github.com/UBCHREST/chrestCompilerFlags.git
        GIT_TAG main
)
FetchContent_MakeAvailable(chrestCompilerFlags)
