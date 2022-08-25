---
layout: default
title: Adding Components
parent: Code Development
nav_order: 20
---
## Registering Your Class with Parser

Ablate uses a factory/parser paradigm that allows you to register that your class if it meets an interface requirement.  The cppParser library (https://github.com/UBCHREST/CppParser) serves as a c++ dependency injection/parsing framework that will dynamically create instances of registered classes.  When registering your class you must describe how to build your class (the arguments, their names, and a brief description).  Full details about using the library can be found with within the [library](https://github.com/UBCHREST/CppParser) and a series of macros have been defined to help simplify registering in [registrar.hpp](https://github.com/UBCHREST/CppParser/blob/main/src/registrar.hpp).

```c++
#include "registrar.hpp"
REGISTER_DEFAULT(ablate::mathFunctions::MathFunction, ablate::mathFunctions::SimpleFormula, "a string based function to be parsed with muparser",
ARG(std::string, "formula", "the formula that may accept x, y, z, t"));
```

In order to view all registered classes run ablate with the ```--help``` command flag.
