---
layout: default
title: Parser
nav_order: 4
---
## Registrar

The ablateLibrary uses a factory/parser paradigm that allows you to register that your class meets an interface requirement.  When registering your class you must describe how to build your class (the arguments, their names, and a brief description).  To access the registrar you must  ```#include "parser/registrar.hpp"```.
 
For instance, in hte following example the ParsedFunction implements MathFunction where it takes a single string argument for the formula.

```c++
REGISTERDEFAULT(ablate::mathFunctions::MathFunction, ablate::mathFunctions::ParsedFunction, "a string based function to be parsed with muparser",
ARG(std::string, "formula", "the formula that may accept x, y, z, t"));
```

In order to view all registered classes run the ablate main statement with the ```-parserHelp``` command flag.

## Parser
At this point in time there is only a single YAML based implementation of a parser/factory. In this implementation arguments are passed as dictionary objects and list available.  When a class must be specified in YAML (no default specified) this must be done with a YAML tag.  For instance, in the following example the first particle in the list specified as a tracer particle initialized using a BoxInitializer.

```yaml
particles:
  - !ablate::particles::Tracer
    ndims: 3
    arguments:
      ts_max_steps: 2
      ts_dt: 0.05
      ts_convergence_estimate: ""
      ts_monitor_cancel: ""
    initializer: !ablate::particles::initializers::BoxInitializer
      arguments:
        particle_lower: 0.25,0.25,.25
        particle_upper: 0.75,0.75,.75
        Npb: 30
        convest_num_refine: 1
    exactSolution:
      formula: "t + x + y"

```

Example Inputs
- [Incompressible Flow 2D.nb]({{site.url}}{{site.baseurl}}/content/parser/inputs/incompressibleFlow.yaml)
- [Tracer Particles 3D.nb]({{site.url}}{{site.baseurl}}/content/parser/inputs/particleTracer3D.yaml)