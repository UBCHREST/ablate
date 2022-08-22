# check to see if the Kokkos_DIR was specified, if not check petsc build for kokkos
if (NOT (DEFINED Kokkos_DIR|CACHE{Kokkos_DIR}|ENV{Kokkos_DIR}))
    # assume that kokkos was built by petsc
    find_package(Kokkos PATHS $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/cmake/Kokkos/)
    if (NOT Kokkos_FOUND)
        message(FATAL_ERROR "Kokkos library not found.  Build PETSc with the --download-kokkos or specify custom built Kokkos_DIR ")
    endif ()

    # prevent error checks on included header files
    set_include_directories_as_system(Kokkos::kokkos)
elseif ()
    find_package(Kokkos REQUIRED)
endif ()