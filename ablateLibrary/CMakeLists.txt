# Setup any ablateLibrary specific dependencies
# Get the CPP dependency
FetchContent_Declare(
        cppParser
        GIT_REPOSITORY https://github.com/UBCHREST/CppParser.git
        GIT_TAG v0.1.2
)
FetchContent_MakeAvailable(cppParser)

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
        GIT_TAG v0.0.10
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

### EGADS ###

# Get all propreties that cmake supports
if(NOT CMAKE_PROPERTY_LIST)
    execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)

    # Convert command output into a CMake list
    string(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    string(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
endif()

function(print_properties)
    message("CMAKE_PROPERTY_LIST = ${CMAKE_PROPERTY_LIST}")
endfunction()

function(print_target_properties target)
    if(NOT TARGET ${target})
        message(STATUS "There is no target named '${target}'")
        return()
    endif()

    foreach(property ${CMAKE_PROPERTY_LIST})
        string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" property ${property})

        # Fix https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
        if(property STREQUAL "LOCATION" OR property MATCHES "^LOCATION_" OR property MATCHES "_LOCATION$")
            continue()
        endif()

        get_property(was_set TARGET ${target} PROPERTY ${property} SET)
        if(was_set)
            get_target_property(value ${target} ${property})
            message("${target} ${property} = ${value}")
        endif()
    endforeach()
endfunction()


# EGADS should be built with petsc using the --download-egads flag
FIND_PACKAGE(opencascade)
FIND_LIBRARY(EGADS_LIBRARY
        NAMES
        egads
        HINTS
        ${CMAKE_FIND_ROOT_PATH}
        ${PETSc_LIBRARY_DIRS}
        PATHS
        ${CMAKE_FIND_ROOT_PATH}
        ${PETSc_LIBRARY_DIRS}
        )
if(NOT TCHEM_LIBRARY)
    message(FATAL_ERROR "Cannot find EGADS library.  Please reconfigure PETSc with --download-egads flag." )
else()
    message("EGADS_LIBRARY" ${EGADS_LIBRARY})
    add_library(EGADS::EGADS UNKNOWN IMPORTED)
    set_target_properties(
            EGADS::EGADS
            PROPERTIES
            IMPORTED_LOCATION ${EGADS_LIBRARY})
endif()

# Load in threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

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
add_subdirectory(finiteElement)
add_subdirectory(boundarySolver)
add_subdirectory(utilities)
add_subdirectory(domain)
add_subdirectory(parameters)
add_subdirectory(solver)
add_subdirectory(mathFunctions)
add_subdirectory(particles)
add_subdirectory(monitors)
add_subdirectory(eos)
add_subdirectory(environment)
add_subdirectory(io)

# Add link to required libraries to the ablate library
target_link_libraries(ablateLibrary PUBLIC ablateCore cppParserLibrary muparser petscXdmfGeneratorLibrary TChem::TChem EGADS::EGADS Threads::Threads  opencascade)

# Add required warnings for the ablate library
ablate_default_target_compile_options_cxx(ablateLibrary)