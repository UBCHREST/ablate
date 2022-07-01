### MuParser ###
# install muparser for reading text equations
set(ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
set(ENABLE_SAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
        mu-parser
        GIT_REPOSITORY https://github.com/beltoforion/muparser.git
        GIT_TAG v2.3.2
)
FetchContent_MakeAvailable(mu-parser)
FetchContent_GetProperties(mu-parser)
target_include_directories(muparser SYSTEM PUBLIC ${mu-parser_SOURCE_DIR}/include)