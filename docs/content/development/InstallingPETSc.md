---
layout: default
title: Installing PETSc Locally
parent: Development Guides
nav_order: 1
---

For the most recent development issues, notes, and workarounds for building PETSc for ABLATE check the [ABLATE repo wiki](https://github.com/UBCHREST/ablate/wiki).

## Install System Specific Prerequisites
### macOS
1. Install macOS Development Tools.  This can be done one of two ways:
   - Install [Xcode](https://apps.apple.com/us/app/xcode/id497799835) using the Mac AppStore
   - Using the terminal, install the command line tools and accept the license
       ```bash
       sudo xcode-select --install 
       ```
2. It is recommended to use a package manager to install other prerequisites.  The following directions will assume [Homebrew](https://brew.sh) has been installed, but alternative methods could be used.
3. Install prerequisites
  ```bash
  brew install cmake autoconf automake libtool libpng gfortran pkg-config clang-format
  ```

### Ubuntu
Using the terminal, install the required prerequisites
  ```bash
  sudo apt-get update
  sudo apt-get install build-essential gfortran git cmake autoconf automake git python3-distutils libpng-dev libtool clang-format pkg-config
  ```

### Windows
It is recommended that development on Windows uses the [Windows Subsystem for Linux 2 (WSL2)](https://docs.microsoft.com/en-us/windows/wsl/install-win10).
1. Follow Microsoft's instructions for installing [WSL2](https://docs.microsoft.com/en-us/windows/wsl/install-win10) and Ubuntu.
1. Open the Ubuntu Terminal window.  The [Windows Terminal](https://www.microsoft.com/en-us/p/windows-terminal/9n0dx20hk701?activetab=pivot:overviewtab) can be used to access Ubuntu.
1. Using the terminal, install the required prerequisites
    ```bash
    sudo apt-get update
    sudo apt-get install build-essential gfortran git cmake autoconf automake git python3-distutils libpng-dev libtool clang-format pkg-config
    ```

## Install PETSc
PETSc can be built in two primary configurations, optimized/release and debug.  In short, the debug build makes it easier to debug but is slower.  The release/optimized build is faster to execute.  Microsoft provides a more detailed overview of the [differences](https://docs.microsoft.com/en-us/visualstudio/debugger/how-to-set-debug-and-release-configurations). The framework requires that PETSc be configured and built with additional options/modules.  Detailed instructions for installing PETSc are available at [petsc.org](https://petsc.org/release/install/), but an abbreviated guide is provided for convenience.  Check [petsc.org](https://petsc.org/release/install/) for additional configuration and compiler flags specific to your system.  
1. Clone PETSc ```git clone https://gitlab.com/petsc/petsc.git ```
    - To checkout a specific version or commit ```git checkout release``` 
2. Configure PETSc to be built in both debug and optimized configurations
   1. Configure PETSc with the following options from the petsc directory to build the debug configuration.  Run the following make command.
       ```bash
       # Configure debug PETSc
       ./configure PETSC_ARCH=arch-ablate-debug --download-mpich --download-fblaslapack --download-ctetgen \
            --download-egads --download-fftw --download-hdf5 --download-metis \
            --download-ml --download-mumps --download-netcdf --download-p4est \
            --download-parmetis --download-pnetcdf --download-scalapack \
            --download-slepc --download-suitesparse --download-superlu_dist \
            --download-triangle --with-slepc --download-zlib --with-libpng --download-tchem
    
       # Follow the on screen directions to make PETSc
       ```

   2. Configure PETSc with the following options from the petsc directory to build the release configuration.  Run the following make command.
       ```bash   
       # Configure opt PETSc
       ./configure PETSC_ARCH=arch-ablate-opt --download-mpich --download-fblaslapack --download-ctetgen \
            --download-egads --download-fftw --download-hdf5 --download-metis \
            --download-ml --download-mumps --download-netcdf --download-p4est \
            --download-parmetis --download-pnetcdf --download-scalapack \
            --download-slepc --download-suitesparse --download-superlu_dist \
            --download-triangle --with-slepc --download-zlib --with-libpng --download-tchem --with-debugging=0 
    
       # Follow the on screen directions to make PETSc
       ```
3. Set up the environmental variables so that ABLATE can locate PETSc. The PETSC_DIR path should be the path to the downloaded PETSc files.  This value is reported in the output of the configure command.
    ```bash
    # Add the following environment variables where PETSC_DIR and PETSC_ARCH are replaced with specified values from the configure command.  On macOS this means putting the following in the ~/.zshrc or ~/.bashrc hidden file (depending on version).  On most Linux versions add the following to the ~/.bashrc file.

    export PETSC_DIR="/path/to/petsc-install"
    export PETSC_ARCH="arch-ablate-debug" # arch-ablate-debug or arch-ablate-opt
    export PKG_CONFIG_PATH="${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig:$PKG_CONFIG_PATH"
    
    # Include the bin directory to access mpi commands
    export PATH="${PETSC_DIR}/${PETSC_ARCH}/bin:$PATH"
    ```
   Specify either the arch-ablate-opt or arch-ablate-debug for PETSC_ARCH depending on what you are running.  If you are developing new capabilities you may want to specify debug.  If you are running large simulations specify opt. This can be changed at any time (may require restarting the terminal/IDE/CLion).