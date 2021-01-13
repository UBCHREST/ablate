---
layout: default
title: Contributing
nav_order: 1
---

This is a step-by-step overview of how to contribute code to the CHREST framework.
1. Create a fork a the project and branch for the issue/feature you will be working on. Detailed instructions are also [available](https://gist.github.com/Chaser324/ce0505fbed06b947d962).
1. Modify or create any required files.  Any new files should be added to the CMakeLists.txt file in the associated folder.
1. Update any documentation within the docs folder.
1. Update or create any tests.  The [GoogleTest](https://github.com/google/googletest) framework is used to control and monitor tests.
1. Make sure that all code meets formatting requirements [Google Style Guide](https://google.github.io/styleguide/) for c++ and [PETSc Style and Usage Guide](https://docs.petsc.org/en/latest/developers/style/) for C.  
    ```bash
    # To run a format check from build directory
    make format-check
    ```
1. Update the CHREST version in the CMakeLists.txt file in the root of the project following [semantic versioning](https://semver.org/).
1. Issue a pull request and reference the associated issue.
 