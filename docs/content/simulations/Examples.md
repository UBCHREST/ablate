---
layout: default
title: Examples Input Files
nav_order: 2
parent: Running Simulations
has_children: false
---

# Integration Test Examples

ABLATE example input files are maintained as part of the main code repository and are primarily used for [integration testing]({{ site.baseurl}}{%link content/development/Testing.md  %}#integration-tests).  A consequence of this is that most of the examples use very small domains/few timesteps to insure they run very quickly.  Despite this they are designed to illustrate a wide variety of ABLATE functions allowing custom input file to be composed. The following input files currently function with ABLATE and can be downloaded directly using the folder icon.  Some examples may require additional files that can be found on [GitHub](https://github.com/UBCHREST/ablate/tree/main/tests/integrationTests/inputs).

{% include_relative integrationExamples/_exampleList.md %}


# Regression Test Examples
The following examples are larger/longer simulations and more closely resemble real work simulations. These [regression tests]({{ site.baseurl}}{%link content/development/Testing.md  %}#regression-tests) are run regularly instead of as part of the pull request processes because of the increased simulation time.

{% include_relative regressionExamples/_exampleList.md %}