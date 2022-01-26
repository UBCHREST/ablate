---
layout: default
title: Stiffened Gas
parent: EOS Formulations 
nav_order: 2
---

The stiffened gas equation of state models liquids using assumptions of an ideal gas with respect to a reference
pressure, assuming that $$C_p$$ and $$C_v$$ are constant. For the current implementation this results in an EOS that is
independent of the gas species. The EOS can be described with $$\gamma, C_p$$ and $$p^0$$.

| Input     | Default Value             | Argument |
| ----------|---------------------------|----------|
| $$\gamma$$| 1.932                     | -gamma   |
| $$C_p$$   | 8095.08 J/kg/K            | -Cp      |
| $$p^0$$   | $$1.1645 \times 10^9$$ Pa | -p0      |

### Decode State

The decode state function computes the required values to compute fluxes from the Euler conserved variables.

#### Internal Energy

$$\begin{eqnarray} e = e_t - KE \end{eqnarray}$$

#### Pressure

$$\begin{eqnarray} p = (\gamma - 1.0) \rho e - \gamma p^0 \end{eqnarray}$$

#### Speed of Sound

$$\begin{eqnarray} a = \sqrt{\gamma (p+p^0)/\rho} \end{eqnarray}$$

### Temperature

The temperature function computes T from $$e_t$$, $$\rho\vector{u}$$, and $$\rho$$.

$$\begin{eqnarray} T = (e - \frac{p^0}{\rho}) \frac{\gamma}{C_p} \end{eqnarray}$$

## Reference

- Chang, C. H. and Liou, M. S. (2007), "A robust and accurate approach to computing compressible multiphase flow:
  Stratified flow model and AUSM+up scheme." Journal of Computational Physics, 225, 840-873.
