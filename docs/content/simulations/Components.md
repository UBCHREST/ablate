---
layout: default
title: Component List
nav_order: 3
parent: Running Simulations
has_children: true
---

ABLATE is composed of components that can be dynamically specified if they meet a set of requirements or "interface".  Any component listed under an interface can be used.  For instance; SimpleFormula, ConstantValue, and LinearTable are all valid MathFunctions that can be used whenever a MathFunction is required.  Some interfaces have a default implementation (indicated with a *) that will be used if a specific type is not specified in the input file. An example of this is SimpleFormula for MathFunction.  Specific component types can be specified using the `!className` or `!ablate::mathFunctions::ConstantValue` convection.