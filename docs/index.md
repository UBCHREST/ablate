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

### src
AblateLibrary is a c++ library used to set up and run simulations.  It is built provides a level of abstraction that simplifies setting up complex simulations.  Details of the code structure can be found in [Source Documentation]({{ site.baseurl}}{%link content/development/SourceDocumentation.md %}).

### tests
The test directory holds four sub-test projects, each focused on a different aspect of testing.

- tests/testingResources: testingResources build a library used by the other testing projects.  This includes helper classes for running mpi based tests
- tests/unitTests: includes unit tests for the core ABLATE functionality
- tests/integrationTests: integration level tests built upon the ablateLibrary parser and runs during every pull request process
- tests/regressionTests: larger tests that may take hours to solve but serve as real examples of using ABLATE.  These are not automatically run as part of the pull request process, instead run weekly to check functionality

### docs
The Markdown documents for ablate.  These are build with [Jekyll](jekyllrb.com) and published automatically upon merge. Follow the [Quickstart](https://jekyllrb.com/docs/) steps to preview your changes locally.  

## Status
Status of the automated build/development/deployment pipelines for ABLATE and associated dependencies.

| Workflow                            | Description                                                                                                                                                                                                                   | Status                                                                                                                                                                                                              |
|-------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PETSc Docker Dependency             | Builds a docker image with the latest version of [PETSc main branch](https://petsc.org/main/) used for ABLATE automated testing. The latest version tested against ABLATE is PETSc {% include generated/petscVersion.html %}. | [![PETSc Docker Dependency](https://github.com/UBCHREST/petsc-docker/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/UBCHREST/petsc-docker/actions/workflows/docker-publish.yml)                |
| TensorFlow Docker Dependency        | Builds a docker image with the latest version of [TensorFlow](https://www.tensorflow.org) used for ABLATE automated testing.                                                                                                  | [![TensorFlow Docker Dependency](https://github.com/UBCHREST/tensorflow-docker/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/UBCHREST/tensorflow-docker/actions/workflows/docker-publish.yml) |
| ABLATE Docker Dependency            | Combines the required build dependencies for ABLATE into a single image for automated testing and local development.                                                                                                          | [![ABLATE Docker Dependency](https://github.com/UBCHREST/ablate/actions/workflows/DependencyDockerBuild.yaml/badge.svg)](https://github.com/UBCHREST/ablate/actions/workflows/DependencyDockerBuild.yaml)           |
| ABLATE Tag Version Workflow         | Tags the post-merge commit                                                                                                                                                                                                    | [![Tag Version Workflow](https://github.com/UBCHREST/ablate/actions/workflows/TagWorkflow.yml/badge.svg)](https://github.com/UBCHREST/ablate/actions/workflows/TagWorkflow.yml)                                     |
| [ABLATE.dev](ABLATE.dev) Deployment | Deploys the latest documentation to [ABLATE.dev](ABLATE.dev)                                                                                                                                                                  | [![Deploy ABLATE.dev workflow](https://github.com/UBCHREST/ablate/actions/workflows/PublishDoc.yaml/badge.svg)](https://github.com/UBCHREST/ablate/actions/workflows/PublishDoc.yaml)                               |

## Acknowledgements
This research is funded by the United States Department of Energy’s (DoE) National Nuclear Security Administration (NNSA) under the Predictive Science Academic Alliance Program III (PSAAP III) at the University at Buffalo, under contract number DE-NA000396
