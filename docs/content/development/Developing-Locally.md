---
layout: default
title: Developing Locally
parent: Development Guides
---

## Install System Specific Prerequisites
### macOS
1. Install [Xcode](https://apps.apple.com/us/app/xcode/id497799835) using the Mac AppStore
1. Using the terminal, install the command line tools and accept the license
    ```bash
    sudo xcode-select --install 
    sudo xcodebuild -license
    ```
1. It is recommended to use a package manager to install other prerequisites.  The following directions will assume [Homebrew](https://brew.sh) has been installed, but alternative methods could be used.
1. Install prerequisites ```brew install cmake autoconf automake libtool libpng gfortran pkg-config clang-format```

### Ubuntu
Using the terminal, install the required prerequisites
    ```bash
    sudo apt-get update
    sudo apt-get install build-essential gfortran git cmake autoconf automake git python3-distutils libpng-dev libtool clang-format pkg-config
    ```

### Windows
It is recommended that development on Windows uses the [Windows Subsystem for Linux 2 (WSL2)](https://docs.microsoft.com/en-us/windows/wsl/install-win10).
1. Follow Microsoft's instructions for installing [WSL2](https://docs.microsoft.com/en-us/windows/wsl/install-win10) and Ubuntu.
1. Open the Ubuntu Terminal window.  The news [Windows Terminal](https://www.microsoft.com/en-us/p/windows-terminal/9n0dx20hk701?activetab=pivot:overviewtab) can be used to access Ubuntu.
1. Using the terminal, install the required prerequisites
    ```bash
    sudo apt-get update
    sudo apt-get install build-essential gfortran git cmake autoconf automake git python3-distutils libpng-dev libtool clang-format pkg-config
    ```

## Install PETSc
The framework requires that PETSc be configured and build with certain options.  Detailed instructions for installing PETSc are available at [https://www.mcs.anl.gov/petsc/documentation/installation.html](https://www.mcs.anl.gov/petsc/documentation/installation.html), but an abbreviated guide is provided for convenience.
1. Clone the PETSc ```git clone -b release https://gitlab.com/petsc/petsc.git petsc```
	- To clone a specific version or commit ```git checkout v3.14.2``` 
1. Configure PETSc with at least the following options from the petsc directory
    ```bash
    ./configure  --download-mpich --download-fblaslapack \
        --download-mpich --download-fblaslapack --download-ctetgen \
 	    --download-egads --download-fftw --download-hdf5 --download-metis \
 	    --download-ml --download-mumps --download-netcdf --download-p4est \
 	    --download-parmetis --download-pnetcdf --download-scalapack \
 	    --download-slepc --download-suitesparse --download-superlu_dist \
 	    --download-triangle --with-slepc --download-zlib --with-libpng
    ```
1. Determine the PETSC_DIR and PETSC_ARCH values from the output of the configure command.  Look for output similar to:
    ```bash
    ...	
    xxx=========================================================================xxx
     Configure stage complete. Now build PETSc libraries with:
      make PETSC_DIR=/Users/path/petsc PETSC_ARCH=arch-darwin-c-debug all
    xxx=========================================================================xxx
    ```
    Add the following environment variables where PETSC_DIR and PETSC_ARCH are replaced with specified values from the configure command.
    ```bash
    export PETSC_DIR="/path/to/petsc-install"
    export PETSC_ARCH="petsc arch name"
    export PKG_CONFIG_PATH="${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig:$PKG_CONFIG_PATH"
    
    # Include the bin directory to access mpi commands
    export PATH="${PETSC_DIR}/${PETSC_ARCH}/bin:$PATH"
    ```
1. Use the configured build system to compile PETSc ```make all check```

## Configuring CLion
CLion is a C/C++ IDE that uses cmake files for configuration. These directions outline the steps to running the framework with CLion.
1. Download and Install [CLion](https://www.jetbrains.com/clion/). For Windows users, follow the [CLion/WSL Instructions](https://www.jetbrains.com/help/clion/how-to-use-wsl-development-environment-in-clion.html) from JetBrains to setup CLion to work with WSL.
1. Open CLion and select *Get From VCS* from the welcome window.
1. Select Git from the *Version Control* drop down and enter the repository url.  It is recommend that you clone your forked version of the repository as outlined in this [guide](https://gist.github.com/Chaser324/ce0505fbed06b947d962).
1. Build and run all tests using the *All CTest* configuration.
  ![Clion All CTest configuration location](assets/clion_ctest_configuration.png)
1. To run the format check in CLion, select the *format-check* build configuration and *build* (hammer icon).
