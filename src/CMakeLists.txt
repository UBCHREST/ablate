# Setup any ablateLibrary specific dependencies
### Ablate Library Setup ###
add_library(ablateLibrary SHARED "")

# Include PETSc and MPI
target_link_libraries(ablateLibrary PUBLIC PkgConfig::PETSc)

# Get the CPP dependency
FetchContent_Declare(
        cppParser
        GIT_REPOSITORY https://github.com/UBCHREST/CppParser.git
        GIT_TAG v0.1.9
)
FetchContent_MakeAvailable(cppParser)
target_link_libraries(ablateLibrary PUBLIC cppParserLibrary)

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
target_link_libraries(ablateLibrary PUBLIC muparser)

### XDMF Generation ###
# install the library for generating xdmf files
set(DISABLE_PETSCXDMFGENERATOR_TESTS ON CACHE BOOL "" FORCE)
FetchContent_Declare(
        petscXdmfGenerator
        GIT_REPOSITORY https://github.com/UBCHREST/petscXdmfGenerator.git
        GIT_TAG v0.1.1
)
FetchContent_MakeAvailable(petscXdmfGenerator)
target_link_libraries(ablateLibrary PUBLIC petscXdmfGeneratorLibrary)

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
target_link_libraries(ablateLibrary PUBLIC TChem::TChem)

### EGADS ###
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
if(NOT EGADS_LIBRARY)
    message(FATAL_ERROR "Cannot find EGADS library.  Please reconfigure PETSc with --download-egads flag." )
else()
    add_library(EGADS::EGADS UNKNOWN IMPORTED)
    set_target_properties(
            EGADS::EGADS
            PROPERTIES
            IMPORTED_LOCATION ${EGADS_LIBRARY})
endif()
target_link_libraries(ablateLibrary PUBLIC EGADS::EGADS)

###  OpenCASCADE ###
# OpenCASCADE should be built with petsc
FIND_PACKAGE(OpenCASCADE CONFIG REQUIRED HINTS "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/cmake/opencascade/")
target_link_libraries(ablateLibrary PUBLIC ${OpenCASCADE_LIBRARIES})
target_link_directories(ablateLibrary PUBLIC ${OpenCASCADE_LIBRARY_DIR})

### Load in threads ###
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(ablateLibrary PUBLIC Threads::Threads)

### Tensorflow ###
pkg_check_modules(TensorFlow QUIET tensorflow)
if(TensorFlow_FOUND)
    target_link_directories(${TensorFlow_LIBRARY_DIRS})
    include_directories(${TensorFlow_INCLUDE_DIRS})
    add_compile_definitions(${TensorFlow_CFLAGS_OTHER})
    message(STATUS "Tensorflow library has been found using package modules")
    target_link_libraries(ablateLibrary PUBLIC TensorFlow)
    target_compile_definitions(ablateLibrary PUBLIC WITH_TENSORFLOW)
elseif(DEFINED ENV{TENSORFLOW_DIR} AND (NOT $ENV{TENSORFLOW_DIR} STREQUAL ""))
    if(NOT DEFINED TENSORFLOW_DIR)
        set(TENSORFLOW_DIR $ENV{TENSORFLOW_DIR})
    endif()

    # manually specify the tensor flow directory with TENSORFLOW_DIR
    FIND_LIBRARY(TensorFlowLibrary
            NAMES
            tensorflow
            HINTS
            ${TENSORFLOW_DIR}
            ${TENSORFLOW_DIR}/lib
            )
    if(NOT TensorFlowLibrary)
        message(FATAL_ERROR "Cannot find TensorFlow library at " ${TENSORFLOW_DIR} )
    else()
        add_library(TensorFlow UNKNOWN IMPORTED)
        set_target_properties(
                TensorFlow
                PROPERTIES
                IMPORTED_LOCATION ${TensorFlowLibrary})
        set_target_properties(
                TensorFlow
                PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES ${TENSORFLOW_DIR}/include)
        message(STATUS "Tensorflow library has been found at the specified TENSORFLOW_DIR location " ${TENSORFLOW_DIR})
        target_compile_definitions(ablateLibrary PUBLIC WITH_TENSORFLOW)
        target_link_libraries(ablateLibrary PUBLIC TensorFlow)
    endif()
else()
    message(STATUS "Tensorflow could not be located and will be skipped. It can be specified with TENSORFLOW_DIR CMAKE variable or env variable ")
endif()


# Allow public access to the header files in the directory
target_include_directories(ablateLibrary PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Tag the version file
configure_file (
        "version.h.in"
        "version.h"
)
target_include_directories(ablateLibrary PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

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
add_subdirectory(chemistry)

# Add required warnings for the ablate library
ablate_default_target_compile_options_cxx(ablateLibrary)