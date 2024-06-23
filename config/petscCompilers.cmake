find_package(PkgConfig REQUIRED)

function(configure_flags PETSC_FLAGS PETSC_FLAGS_OUT)
    # remove any flag that will not work with ablate
    list(FILTER PETSC_FLAGS EXCLUDE REGEX "-fvisibility")

    # check for the fsanitize and add to overall link
    if ("-fsanitize=address" IN_LIST PETSC_FLAGS)
        add_link_options(-fsanitize=address)
        message(STATUS "Adding -fsanitize=address link option")
    endif ()

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

    message(STATUS "Using found PETSc C Compiler/Flags: ${PETSC_C_COMPILER} ${PETSC_FLAGS_OUT}\n")
endif ()

# Check if a CXX compiler is explicitly stated
if (NOT DEFINED CMAKE_CXX_COMPILER)
    # Set the compilers based upon the PETSc package
    pkg_get_variable(PETSC_CXX_COMPILER PETSc cxxcompiler)
    set(CMAKE_CXX_COMPILER ${PETSC_CXX_COMPILER})

    pkg_get_variable(PETSC_CXX_FLAGS PETSc cxxflags_extra)
    configure_flags("${PETSC_CXX_FLAGS}" PETSC_FLAGS_OUT)

    # allow adding custom CXX_ADDITIONAL_FLAGS
    if (DEFINED ENV{CXX_ADDITIONAL_FLAGS})
        string(APPEND PETSC_FLAGS_OUT " " $ENV{CXX_ADDITIONAL_FLAGS})
    endif ()

    set(CMAKE_CXX_FLAGS_INIT ${PETSC_FLAGS_OUT})

    message(STATUS "Using found PETSc c++ Compiler/Flags: ${PETSC_CXX_COMPILER} ${PETSC_FLAGS_OUT}\n")
endif ()


# Check if a Fortran compiler is explicitly stated
if (NOT DEFINED CMAKE_Fortran_COMPILER)
    # Set the compilers based upon the PETSc package
    pkg_get_variable(PETSC_Fortran_COMPILER PETSc fcompiler)
    set(CMAKE_Fortran_COMPILER ${PETSC_Fortran_COMPILER})

    pkg_get_variable(PETSC_Fortran_FLAGS PETSc fflags_extra)
    configure_flags("${PETSC_Fortran_FLAGS}" PETSC_FLAGS_OUT)

    set(CMAKE_Fortran_FLAGS_INIT ${PETSC_FLAGS_OUT})

    message(STATUS "Using found PETSc Fortran Compiler/Flags: ${CMAKE_Fortran_COMPILER} ${PETSC_FLAGS_OUT}\n")
endif ()


# Check if a CUDA compiler is explicitly stated
if (NOT DEFINED CMAKE_CUDA_COMPILER)
    # Set the compilers based upon the PETSc package
    pkg_get_variable(CMAKE_CUDA_COMPILER PETSc cudacompiler)
    set(CMAKE_CUDA_COMPILER ${CMAKE_CUDA_COMPILER})

    pkg_get_variable(CMAKE_CUDA_HOST_COMPILER PETSc cuda_cxx)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CUDA_HOST_COMPILER})

    pkg_get_variable(PETSC_CUDA_FLAGS PETSc cudaflags_extra)
    configure_flags("${PETSC_CUDA_FLAGS}" PETSC_FLAGS_OUT)
    set(PETSC_CUDA_FLAGS_INIT ${PETSC_FLAGS_OUT})

    if (DEFINED CMAKE_CUDA_COMPILER)
        message("Using found PETSc CUDA Compiler/Flags: ${CMAKE_CUDA_COMPILER} ${PETSC_FLAGS_OUT}\n")
    endif ()
endif ()