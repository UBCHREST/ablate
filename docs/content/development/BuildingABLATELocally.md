---
layout: default
title: Building ABLATE Locally
parent: Development Guides
nav_order: 3
---

These directions outline the steps for downloading, building, and testing ABLATE on your local computer.  These directions assume you have installed PETSc and setup the environmental variables outlined in [Installing PETSc for ABLATE]({{ site.baseurl}}{%link content/development/InstallingPETSc.md  %}).  There are a variety of ways to build and interact with ABLATE including the command line and integrated development environments (IDEs). This document will cover building with the command line (WSL on Windows) and [CLion](https://www.jetbrains.com/clion/).

## Note:
For the most recent development issues, notes, and workarounds for building ABLATE check the [ABLATE repo wiki](https://github.com/UBCHREST/ablate/wiki).

## Downloading and Building with the Command Line
1. Clone your [ABLATE fork]({{ site.baseurl}}{%link content/development/UsingGitWithABLATE.md  %}#forking-ablate) onto your local machine. It is recommended that you [setup passwordless ssh](https://docs.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account) for accessing GitHub.
   ```bash
   git clone git@github.com:USERNAME/ablate.git
   ```
1. Create debug and release build directories
    ```bash
    mkdir debug
    mkdir release
    ```
1. Configure and build ABLATE
    ```bash
    # debug mode
    cmake -DCMAKE_BUILD_TYPE=Debug -B debug -S ablate
    make -C debug

    # release
    cmake -DCMAKE_BUILD_TYPE=Release -B release -S ablate
    make -C release
    ```
1. Run all of the ABLATE tests
   ```bash
   cd release
   ctest
   ```

## Downloading and Building with CLion
CLion is a C/C++ IDE that uses cmake files for configuration. These directions outline the steps to running the framework with CLion.
1. Download and Install [CLion](https://www.jetbrains.com/clion/). For Windows users, follow the [CLion/WSL Instructions](https://www.jetbrains.com/help/clion/how-to-use-wsl-development-environment-in-clion.html) from JetBrains to setup CLion to work with WSL.
1. Open CLion and select *Get From VCS* from the welcome window and either
   - (recommended) Select GitHub and Login/Authorize access. Then follow on screen instructions to clone your [fork of ABLATE]({{ site.baseurl}}{%link content/development/UsingGitWithABLATE.md  %}#forking-ablate).
   - Select Git from the *Version Control* drop down and enter your [ABLATE fork url]({{ site.baseurl}}{%link content/development/UsingGitWithABLATE.md  %}#forking-ablate).
1. Build and run all tests using the *All CTest* configuration.
   ![Clion All CTest configuration location](assets/clion_ctest_configuration.png)
1. If you are new to CLion it is recommended that you read through the [CLion Quick Start Guide](https://www.jetbrains.com/help/clion/clion-quick-start-guide.html).