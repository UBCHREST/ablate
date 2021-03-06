---
layout: about
---
**ABLATE** **A**blative **B**oundary **L**ayers **A**t **T**he **E**xascale

ABLATE is a [UB CHREST](https://www.buffalo.edu/chrest.html) project focused on leveraging advances in both exascale computing and machine learning to better understand the turbulent mixing and fuel entrainment in the combustion environment that is critical to the operation of hybrid rocket motors.

The ABLATE documentation that should contain:
* building/running the framework
* physics overview and assumptions
* development guides

# Repository Layout
## ablateCore
AblateCore builds a c library of reusable functions and datastructures for solving flow and particle based system.

## ablateLibrary
AblateLibrary is a c++ library used to setup and run simulations.  It is built upon ablateCore and provides a level of abstraction that simplifies setting up complex simulations.

## tests
The test directory holds four sub-test projects, each focused on a different aspect of testing.

- tests/testingResources: testingResources build a library used by the other projects.  This includes helper classes for running mpi based tests.
- tests/ablateCore: includes tests for the core functionality.  These tests resemble what another client may use from ablateCore.
- tests/ablateLibrary: includes unit level tests for the ablate library classes.
- tests/integrationTests: integration level tests built upon the ablateLibrary parser.

## docs
The Markdown docs for ablate.  These are build with [Jekyll](jekyllrb.com) and published automatically upon merge.