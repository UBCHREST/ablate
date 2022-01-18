---
layout: default
title: Boundary Conditions
parent: Compressible Flow Formulation
grand_parent: Flow Formulations
nav_order: 4
---

More complex boundary conditions can be implemented using the boundary solver.  The boundary solver allows for boundary cells/elements to be solved as part of the overall time integration and specified arbitrarily throughout the domain.  The solver can utilize field and gradient information to compute source terms in the boundary solver domain.  The most relevant boundary conditions for compressible flow are the Local One-Dimensional Inviscid (LODI) boundary conditions.  In this approach one-dimensional problems are used to infer values for the wave amplitude variations in the multi-dimensional flow [1].  Full derivation details are available in Poinsot and Lelef[1].  Implemented boundaries include ["open"](https://github.com/UBCHREST/ablate/blob/main/ablateLibrary/boundarySolver/lodi/openBoundary.hpp), ["wall"](https://github.com/UBCHREST/ablate/blob/main/ablateLibrary/boundarySolver/lodi/isothermalWall.hpp), and ["inlet"](https://github.com/UBCHREST/ablate/blob/main/ablateLibrary/boundarySolver/lodi/inlet.hpp).

## References
 - [1] Poinsot, T. J &, and S. K. Lelef. "Boundary conditions for direct simulations of compressible viscous flows." Journal of computational physics 101.1 (1992): 104-129.