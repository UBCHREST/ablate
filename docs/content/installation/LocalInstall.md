---
layout: default
title: Local Install
parent: Installation
nav_order: 1
---

## Install ABLATE Dependencies
Keeping all dependency build steps updated can be difficult as they change often and are OS/platform specific. To allow the most up-to-date information to be available dependency build steps are maintained in the [ABLATE wiki](https://github.com/UBCHREST/ablate/wiki).  Please follow the step-by-step guides for your OS/platform on the ABLATE wiki to install the required dependencies before following this page.  These directions are for local install where PETSc built mpi and the PETSC_DIR environmental variable was set according to the [ABLATE wiki](https://github.com/UBCHREST/ablate/wiki).  The latest version of ABLATE has been tested against PETSc {% include generated/petscVersion.html %}.

## Downloading and Building with CLion (with installed dependencies)
CLion is a C/C++ IDE that uses cmake files for configuration. These directions outline the steps to running the framework with CLion.
1. Download and Install [CLion](https://www.jetbrains.com/clion/).
2. Open CLion and select *Get From VCS* from the welcome window and either
   - (recommended) Select GitHub and Login/Authorize access. Then follow on-screen instructions to clone your [fork of ABLATE]({{ site.baseurl}}{%link content/development/GitOverview.md  %}#forking-ablate).
   - Select Git from the *Version Control* dropdown and enter your [ABLATE fork url]({{ site.baseurl}}{%link content/development/GitOverview.md  %}#forking-ablate).
3. Enable the ```local-ablate-debug``` and ```local-ablate-opt``` build profiles.
   - If not opened by default, open the  Settings / Preferences > Build, Execution, Deployment > CMake preference window from the menu bar.
   - Select the ```local-ablate-debug``` and click the "Enable profile". Repeat for the ```local-ablate-opt``` and apply/close the window.
     ![clion cmake profiles](assets/clion_cmake_profiles.png)
   - Select the ```local-ablate-opt``` or ```local-ablate-debug``` build profile under the build toolbar.  In short, the debug build makes it easier to debug but is slower.  The release/optimized build is faster to execute.
     ![clion cmake select build profile](assets/clion_cmake_select_build_profile.png)
   - Disable any other profile
4. Build and run all tests using the *All CTest* configuration.
   ![Clion All CTest configuration location](assets/clion_ctest_configuration.png)
5. If you are new to CLion it is recommended that you read through the [CLion Quick Start Guide](https://www.jetbrains.com/help/clion/clion-quick-start-guide.html).

## Downloading and Building with the Command Line
1. Clone your [ABLATE fork]({{ site.baseurl}}{%link content/development/GitOverview.md  %}#forking-ablate) onto your local machine. It is recommended that you [setup passwordless ssh](https://docs.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account) for accessing GitHub.
   ```bash
   git clone git@github.com:USERNAME/ablate.git
   ```
2. Move into the ablate directory
    ```bash
    cd ablate
    ```
3. Configure and build ABLATE. Both the debug and optimized versions are built. If you are developing new capabilities you may want to specify debug.  If you are running large simulations specify opt.
    ```bash
    # debug mode
    cmake  --preset=local-ablate-debug
    cmake --build --preset=local-ablate-debug -j

    # optimized
    cmake  --preset=local-ablate-opt
    cmake --build --preset=local-ablate-opt -j
    ```
4. Run all the ABLATE tests
    ```bash
    # debug mode
    ctest --preset=local-ablate-debug

    # optimized
    ctest --preset=local-ablate-opt
    ```
5. Run ablate with an input file
    ```bash
    # debug mode
    $PETSC_DIR/arch-ablate-debug/bin/mpirun -n 2 cmake-build-debug-ablate/ablate --input /path/to/input/file

    # optimized
    $PETSC_DIR/arch-ablate-opt/bin/mpirun -n 2 cmake-build-opt-ablate/ablate --input /path/to/input/file
    ```