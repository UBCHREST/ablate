---
layout: default
title: Getting Started
nav_order: 2
---
**Welcome to ABLATE!**  This guide will provide a step-by-step guide to downloading, building, and running ABLATE.  There are two primary ways of interacting with ABLATE:

- (simple) running ABLATE using a text input file (.yaml)
- (advanced) writing your own client library

This Guide will step you through both scenarios. This Guide assumes that you are building and running ABLATE on your local machine. Additional directions are available for specific computing environments. 

1. **Installing PETSc** Regardless of how you run ABLATE the first step is installing [PETSc](https://www.mcs.anl.gov/petsc/).  PETSc is a library for the scalable solution of scientific applications modeled by partial differential equations and is required by ABLATE.  The [Installing PETSc for ABLATE]({{ site.baseurl}}{%link content/development/InstallingPETSc.md  %}) provides an overview of installing PETSc.
1. **Forking ABLATE** If you plan on making any contributions to ABLATE you must first [Fork ABLATE]({{ site.baseurl}}{%link content/development/UsingGitWithABLATE.md  %}#forking-ablate).  This creates a version of ABLATE that you can modify, commit, and test.  When complete, you can create a Pull Request to bring your changes back to the main branch. 
1. **Building ABLATE** Inorder to run ABLATE using a text input file or make contributions you will need to [download and build ablate]({{ site.baseurl}}{%link content/development/BuildingABLATELocally.md  %}).
1. **Running ABLATE with an Input File** ABLATE includes a yaml parser for setting up and configuring simulations.  [Step-by-Step instructions]({{ site.baseurl}}{%link content/parser/RunningSimulations.md  %}) will guide you through running the sample cases.
1. **Using ABLATE in your program** You can use ABLATE within your own application.  This is often useful as a first step in developing new features for ABLATE or trying to gain further understanding of the library.  Once developed and tested these capabilities can be integrated back to ABLATE with full testing and parser support.  A example [client repository]({{ site.baseurl}}{%link content/development/ClientLibrary.md  %}) if available to help getting started.