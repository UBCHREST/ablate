---
layout: default
title: Getting Started
nav_order: 2
---

## Welcome to ABLATE!
This guide will provide a step-by-step guide to downloading, building, and running ABLATE.  There are two primary ways of interacting with ABLATE:

- (simple) running ABLATE using a text input file (.yaml)
- (advanced) writing your own client library

This guide will step you through both scenarios. This guide assumes that you are building and running ABLATE on your local machine. Additional directions are available for specific computing environments. 

## Running ABLATE Simulations
1. **Installing ABLATE** Regardless of how you run ABLATE the first step is installing [ABLATE]({{ site.baseurl}}{%link content/installation/index.md  %}).
2. **Running ABLATE with an Input File** ABLATE includes a yaml parser for setting up and configuring simulations.  [Step-by-Step instructions]({{ site.baseurl}}{%link content/simulations/index.md  %}) will walk you through running the sample cases.
3. **Using ABLATE in your program** You can use ABLATE within your own application.  This is often useful as a first step in developing new features for ABLATE or trying to gain further understanding of the library.  Once developed and tested these capabilities can be integrated back to ABLATE with full testing and parser support.  A example [client repository]({{ site.baseurl}}{%link content/development/ClientLibrary.md  %}) if available to help to get started.

## Contributing to ABLATE
1. **[Coding ABLATE Guide]({{ site.baseurl}}{%link content/development/CodingAblate.md  %})** This guide outlines a step-by-step procedure for getting started with ABLATE assuming little to no c/c++ experience.
2. **Forking ABLATE** If you plan on making any contributions to ABLATE you must first [Fork ABLATE]({{ site.baseurl}}{%link content/development/GitOverview.md  %}#forking-ablate).  This creates a version of ABLATE that you can modify, commit, and test.
3. **Installing ABLATE** Regardless of how you run ABLATE the first step is installing [ABLATE]({{ site.baseurl}}{%link content/installation/index.md  %}).
4. **Running ABLATE with an Input File** ABLATE includes a yaml parser for setting up and configuring simulations.  [Step-by-Step instructions]({{ site.baseurl}}{%link content/simulations/index.md  %}) will walk you through running the sample cases.
5. **Using ABLATE in your program** You can use ABLATE within your own application.  This is often useful as a first step in developing new features for ABLATE or trying to gain further understanding of the library.  Once developed and tested these capabilities can be integrated back to ABLATE with full testing and parser support.  A example [client repository]({{ site.baseurl}}{%link content/development/ClientLibrary.md  %}) if available to help to get started.
6. **Developing Code** Follow the guidelines on how to write code in ABLATE.
1. **[Testing Code]({{ site.baseurl}}{%link content/development/Testing.md  %})** Testing is essential for any high-quality software product and should be integrated at an early stage of development.
1. **[Contributing Code]({{ site.baseurl}}{%link content/development/Contributing.md  %})** How to get your changes back into ABLATE to allow others people to use.
