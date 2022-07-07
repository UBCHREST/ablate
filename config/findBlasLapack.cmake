# march over each target property to update path to includes
FUNCTION(find_petsc_blas_lapack OPENBLAS_INSTALL_PATH LAPACKE_INSTALL_PATH TINES_ENABLE_MKL)
    # use petsc to guess the the library for either openblas or MKL
    get_petsc_argument("--with-blaslapack-dir=(.*)" blaslapackDir)

    # check if the specified library is either openblas or mkl
    message(STATUS "Found PETSc blaslapack-dir " ${blaslapackDir})
    if (blaslapackDir)
        set(BLA_VENDOR OpenBLAS)
        find_package(BLAS NO_DEFAULT_PATH QUIET PATHS ${blaslapackDir})
        if (BLAS_FOUND)
            set(OPENBLAS_INSTALL_PATH ${blaslapackDir} PARENT_SCOPE)
            message(STATUS Using OPENBLAS_INSTALL_PATH ${OPENBLAS_INSTALL_PATH})
            return()
        ENDIF ()

        # check for intel
        set(intelBlasList "Intel;Intel10_32;Intel10_64lp;Intel10_64lp_seq;Intel10_64ilp;Intel10_64ilp_seq;Intel10_64_dyn")

        foreach (intelBlas ${intelBlasList})
            set(BLA_VENDOR ${intelBlas})
            find_package(BLAS QUIET)

            if (BLAS_FOUND)
                set(TINES_ENABLE_MKL TRUE PARENT_SCOPE)
                message(STATUS Set TINES_ENABLE_MKL ${TINES_ENABLE_MKL})
                return()
            endif ()
        endforeach ()
    endif ()

    # check to see if petsc was used to download openblas
    check_petsc_argument("--download-openblas" openBlasDownload)
    if (openBlasDownload)
        # set the open blas to the petsc directory

        return()
    endif ()
ENDFUNCTION()