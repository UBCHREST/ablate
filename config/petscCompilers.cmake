find_package(PkgConfig REQUIRED)

function(configure_flags PETSC_FLAGS PETSC_FLAGS_OUT)
    # remove any flag that will not work with ablate
    list(FILTER PETSC_FLAGS EXCLUDE REGEX "-fvisibility")
    string(REPLACE ";" " " PETSC_FLAGS_STR "${PETSC_FLAGS}")
    set(${PETSC_FLAGS_OUT} "${PETSC_FLAGS_STR}" PARENT_SCOPE)
endfunction()

# Check if a C compiler is explicitly stated
if (NOT DEFINED CMAKE_C_COMPILER)
    # Set the compilers based upon the PETSc package
    pkg_get_variable(PETSC_C_COMPILER PETSc ccompiler)
    set(CMAKE_C_COMPILER ${PETSC_C_COMPILER})

    pkg_get_variable(PETSC_C_FLAGS PETSc cflags_extra)
    configure_flags("${PETSC_C_FLAGS}" PETSC_FLAGS_OUT)
    set(CMAKE_C_FLAGS_INIT ${PETSC_C_FLAGS_STR})

    message("Using found PETSc C Compiler/Flags: ${PETSC_C_COMPILER} ${PETSC_FLAGS_OUT}\n")
endif ()

# Check if a CXX compiler is explicitly stated
if (NOT DEFINED CMAKE_CXX_COMPILER)
    # Set the compilers based upon the PETSc package
    pkg_get_variable(PETSC_CXX_COMPILER PETSc cxxcompiler)
    set(CMAKE_CXX_COMPILER ${PETSC_CXX_COMPILER})

    pkg_get_variable(PETSC_CXX_FLAGS PETSc cxxflags_extra)
    configure_flags("${PETSC_CXX_FLAGS}" PETSC_FLAGS_OUT)
    set(CMAKE_CXX_FLAGS_INIT ${PETSC_FLAGS_OUT})

    message("Using found PETSc c++ Compiler/Flags: ${PETSC_CXX_COMPILER} ${PETSC_FLAGS_OUT}\n")
endif ()
