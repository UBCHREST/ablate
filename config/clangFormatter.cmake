# Download the clang format script
include(FetchContent)
FetchContent_Declare(
        run-clang-format
        GIT_REPOSITORY https://github.com/Sarcasm/run-clang-format.git
        GIT_TAG 39081c9c42768ab5e8321127a7494ad1647c6a2f
)
FetchContent_MakeAvailable(run-clang-format)

# Add a target to allow linting check
find_program(CLANG_FORMAT "clang-format")
find_package (Python)

# determine the version
EXECUTE_PROCESS(COMMAND ${CLANG_FORMAT} --version OUTPUT_VARIABLE CLANG_FORMAT_INFO)

# Special case for version 16
if(${CLANG_FORMAT_INFO} MATCHES " 16.")
    if(CLANG_FORMAT AND Python_Interpreter_FOUND)
        add_custom_target(
                format-check
                COMMAND ${Python_EXECUTABLE} ${run-clang-format_SOURCE_DIR}/run-clang-format.py
                --style=file:config/.clang-format.v16
                -r
                --extensions=cpp,hpp,cc,hh,c++,h++,cxx,hxx,c,h
                ${PROJECT_SOURCE_DIR}/src
                ${PROJECT_SOURCE_DIR}/tests
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                USES_TERMINAL
        )
    endif()
elseif ()
    if(CLANG_FORMAT AND Python_Interpreter_FOUND)
        add_custom_target(
                format-check
                COMMAND ${Python_EXECUTABLE} ${run-clang-format_SOURCE_DIR}/run-clang-format.py
                --style=file
                -r
                --extensions=cpp,hpp,cc,hh,c++,h++,cxx,hxx,c,h
                ${PROJECT_SOURCE_DIR}/src
                ${PROJECT_SOURCE_DIR}/tests
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                USES_TERMINAL
        )
    endif()
endif ()