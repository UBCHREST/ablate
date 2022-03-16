---
layout: default
title: Building ABLATE with Docker
parent: Development Guides
nav_order: 4
---

If PETSc and other dependencies (tensorflow) cannot be installed for local development a docker container can be used with CLion to run/test locally.  There will be overhead in the computional cost using this method.

1. Download and install [Docker](https://www.docker.com) or docker desktop.
2. Download and install [CLion](https://www.jetbrains.com/clion/download/).
3. Build a local Docker image that contains the required depencicies.  This step will download other images from GitHub and assembles them.
   ```bash
   # From the root of the ablate directory
   docker build -t ablate-dep-image --build-arg PETSC_BUILD_ARCH='arch-opt' -f DockerDependencyFile .
   ```
4. Set the toolchain in CLion using the [instructions](https://www.jetbrains.com/help/clion/clion-toolchains-in-docker.html).  Select the 'ablate-dep-image:latest' image in the UI.
