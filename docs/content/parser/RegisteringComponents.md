---
layout: default
title: Registering Components
parent: Input Files
nav_order: 1
---
## Registering Your Class

The ablateLibrary uses a factory/parser paradigm that allows you to register that your class meets an interface requirement.  When registering your class you must describe how to build your class (the arguments, their names, and a brief description).  To access the registrar you must  ```#include "parser/registrar.hpp"```.

For instance, in the following example the ParsedFunction implements MathFunction where it takes a single string argument for the formula.

```c++
REGISTERDEFAULT(ablate::mathFunctions::MathFunction, ablate::mathFunctions::ParsedFunction, "a string based function to be parsed with muparser",
ARG(std::string, "formula", "the formula that may accept x, y, z, t"));
```

In order to view all registered classes run the ablate main statement with the ```--help``` command flag.
