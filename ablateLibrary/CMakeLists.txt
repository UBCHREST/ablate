# Setup any ablateLibrary specific dependencies
### YAML CPP ###
# turn off yaml-cpp options
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG yaml-cpp-0.6.3
)
FetchContent_MakeAvailable(yaml-cpp)

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

### XDMF Generation ###
# install the library for generating xdmf files
set(DISABLE_PETSCXDMFGENERATOR_TESTS ON CACHE BOOL "" FORCE)
FetchContent_Declare(
        petscXdmfGenerator
        GIT_REPOSITORY https://github.com/UBCHREST/petscXdmfGenerator.git
        GIT_TAG v0.0.9
)
FetchContent_MakeAvailable(petscXdmfGenerator)

### TCHEM ###
# TCHEM should be built with petsc using the --download-tchem flag
FIND_LIBRARY(TCHEM_LIBRARY
        NAMES
            tchem
        HINTS
            ${CMAKE_FIND_ROOT_PATH}
            ${PETSc_LIBRARY_DIRS}
        PATHS
            ${CMAKE_FIND_ROOT_PATH}
            ${PETSc_LIBRARY_DIRS}
        )
if(NOT TCHEM_LIBRARY)
    message(FATAL_ERROR "Cannot find TChem library.  Please reconfigure PETSc with --download-tchem flag." )
else()
    add_library(TChem::TChem UNKNOWN IMPORTED)
    set_target_properties(
            TChem::TChem
            PROPERTIES
                IMPORTED_LOCATION ${TCHEM_LIBRARY})
endif()

### Ablate Library Setup ###
add_library(ablateLibrary SHARED "")

# Allow public access to the header files in the directory
target_include_directories(ablateLibrary PUBLIC ${CMAKE_CURRENT_LIST_DIR})

target_sources(ablateLibrary
        PUBLIC
        builder.cpp
        builder.hpp
        )

# Include the code in any subdirectory
add_subdirectory(finiteVolume)
add_subdirectory(utilities)
add_subdirectory(domain)
add_subdirectory(parameters)
add_subdirectory(solver)
add_subdirectory(parser)
add_subdirectory(mathFunctions)
add_subdirectory(particles)
add_subdirectory(monitors)
add_subdirectory(eos)
add_subdirectory(environment)
add_subdirectory(io)

# Add link to required libraries to the ablate library
target_link_libraries(ablateLibrary PUBLIC ablateCore yaml-cpp muparser petscXdmfGeneratorLibrary TChem::TChem)

# Add required warnings for the ablate library
ablate_default_target_compile_options_cxx(ablateLibrary)