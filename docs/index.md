---
layout: about
title: About
nav_order: 1
---
# **ABLATE** **A**blative **B**oundary **L**ayers **A**t **T**he **E**xascale
**Version {% include generated/version.html %}**

ABLATE is a [UB CHREST](https://www.buffalo.edu/chrest.html) project focused on leveraging advances in both exascale computing and machine learning to better understand the turbulent mixing and fuel entrainment in the combustion environment that is critical to the operation of hybrid rocket motors.

The ABLATE documentation that should contain:
* building/running the framework
* physics overview and assumptions
* development guides

## Getting Started
New to Ablate? Get up and running with the [Getting Started Guide.]({{ site.baseurl}}{%link content/GettingStarted.md  %})

## Repository Layout
### ablateCore
AblateCore builds a c library of reusable functions and data structures for solving flow and particle based system.  Use of this library requires knowledge of PETSc. 

### ablateLibrary
AblateLibrary is a c++ library used to setup and run simulations.  It is built upon ablateCore and provides a level of abstraction that simplifies setting up complex simulations.

### tests
The test directory holds four sub-test projects, each focused on a different aspect of testing.

- tests/testingResources: testingResources build a library used by the other testing projects.  This includes helper classes for running mpi based tests.
- tests/ablateCore: includes tests for the core functionality.  These tests resemble what another client may use from ablateCore.
- tests/ablateLibrary: includes unit level tests for the ablate library classes.
- tests/integrationTests: integration level tests built upon the ablateLibrary parser.

### docs
The Markdown documents for ablate.  These are build with [Jekyll](jekyllrb.com) and published automatically upon merge. Follow the [Quickstart](https://jekyllrb.com/docs/) steps to preview your changes locally.  

## Acknowledgements
This research is funded by the United States Department of Energy’s (DoE) National Nuclear Security Administration
(NNSA) under the Predictive Science Academic Alliance Program III (PSAAP III) at the University at Buffalo, under
contract number DE-NA000396