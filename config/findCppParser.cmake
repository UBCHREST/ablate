# Get the CPP dependency
IF(TARGET CHREST::cppParserLibrary)
    message("Found CHREST::cppParserLibrary target")
ELSE()
    FetchContent_Declare(
            cppParserLibrary
            GIT_REPOSITORY https://github.com/mmcgurn/CppParser.git
            GIT_TAG mcgurn/cmake-cleanup
    )
    FetchContent_MakeAvailable(cppParserLibrary)
    add_library(CHREST::cppParserLibrary ALIAS cppParserLibrary)
    add_library(CHREST::cppParserTestLibrary ALIAS cppParserTestLibrary)
ENDIF()

