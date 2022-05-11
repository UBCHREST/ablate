---
layout: default
title: Component List
nav_order: 3
parent: Running Simulations
has_children: true
---

ABLATE is composed of components that can be dynamically specified if they meet set of requirements or "interface".  Any component listed under an interface can be used.  For instance; SimpleFormula, ConstantValue, and LinearTable are all valid MathFunctions that can be used whenever a MathFunction is required.  Some interfaces have a default implementation (such as SimpleFormula for MathFunction) that will be used if a specific type is not specified in the input file.  Specific component types can be specified using the `!className` or `!ablate::mathFunctions::ConstantValue` convection.