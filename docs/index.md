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
AblateLibrary is a c++ library used to setup and run simulations.  It is built upon ablateCore and provides a level of abstraction that simplifies setting up complex simulations.  Details of the code structure can be found in [Source Documentation]({{ site.baseurl}}{%link content/development/SourceDocumentation.md %}).

### tests
The test directory holds four sub-test projects, each focused on a different aspect of testing.

- tests/testingResources: testingResources build a library used by the other testing projects.  This includes helper classes for running mpi based tests.
- tests/ablateCore: includes tests for the core functionality.  These tests resemble what another client may use from ablateCore.
- tests/ablateLibrary: includes unit level tests for the ablate library classes.
- tests/integrationTests: integration level tests built upon the ablateLibrary parser.

### docs
The Markdown documents for ablate.  These are build with [Jekyll](jekyllrb.com) and published automatically upon merge. Follow the [Quickstart](https://jekyllrb.com/docs/) steps to preview your changes locally.  

## Status
Status of the automated build/development/deployment pipelines for ABLATE and associated dependencies.

| Workflow                            | Description                                                                                                                                                                                                                   | Status                                                                                                                                                                                                              |
|-------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PETSc Docker Dependency             | Builds a docker image with the latest version of [PETSc main branch](https://petsc.org/main/) used for ABLATE automated testing. The latest version tested against ABLATE is PETSc {% include generated/petscVersion.html %}. | [![PETSc Docker Dependency](https://github.com/UBCHREST/petsc-docker/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/UBCHREST/petsc-docker/actions/workflows/docker-publish.yml)                |
| TensorFlow Docker Dependency        | Builds a docker image with the latest version of [TensorFlow](https://www.tensorflow.org) used for ABLATE automated testing.                                                                                                  | [![TensorFlow Docker Dependency](https://github.com/UBCHREST/tensorflow-docker/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/UBCHREST/tensorflow-docker/actions/workflows/docker-publish.yml) |
| ABLATE Docker Dependency            | Combines the required build dependencies for ABLATE into a single image for automated testing and local development.                                                                                                          | [![ABLATE Docker Dependency](https://github.com/UBCHREST/ablate/actions/workflows/DependencyDockerBuild.yaml/badge.svg)](https://github.com/UBCHREST/ablate/actions/workflows/DependencyDockerBuild.yaml)           |
| ABLATE Post-Merge Workflow          | Tags the post-merge commit and generates part of the automated ABLATE.dev documentation.                                                                                                                                      | [![ABLATE Post-Merge Workflow](https://github.com/UBCHREST/ablate/actions/workflows/MergeWorkflow.yml/badge.svg)](https://github.com/UBCHREST/ablate/actions/workflows/MergeWorkflow.yml)                           |
| [ABLATE.dev](ABLATE.dev) Deployment | Deploys the latest documentation to [ABLATE.dev](ABLATE.dev)                                                                                                                                                                  | [![ABLATE.dev Deployment](https://github.com/UBCHREST/ablate/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/UBCHREST/ablate/actions/workflows/pages/pages-build-deployment)          |

## Acknowledgements
This research is funded by the United States Department of Energy’s (DoE) National Nuclear Security Administration (NNSA) under the Predictive Science Academic Alliance Program III (PSAAP III) at the University at Buffalo, under contract number DE-NA000396
