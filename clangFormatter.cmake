# Download the clang format script
include(FetchContent)
FetchContent_Declare(
        run-clang-format
        GIT_REPOSITORY https://github.com/Sarcasm/run-clang-format.git
        GIT_TAG 39081c9c42768ab5e8321127a7494ad1647c6a2f
)
FetchContent_MakeAvailable(run-clang-format)

message(clang ${run-clang-format_SOURCE_DIR})

# Add a target to allow linting check
find_program(CLANG_FORMAT "clang-format")
find_package (Python)
if(CLANG_FORMAT AND Python_Interpreter_FOUND)
    add_custom_target(
            format-check
            COMMAND ${Python_EXECUTABLE} ${run-clang-format_SOURCE_DIR}/run-clang-format.py
            --style=file
            -r
            --extensions=cpp,hpp,cc,hh,c++,h++,cxx,hxx
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/tests
            COMMAND ${PROJECT_SOURCE_DIR}/extern/petscFormat/petscFormatTest.sh
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            USES_TERMINAL
    )
endif()