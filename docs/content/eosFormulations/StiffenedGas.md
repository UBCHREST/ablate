---
layout: default
title: Stiffened Gas
parent: EOS Formulations
nav_order: 2
---

The stiffened gas equation of state models liquids using assumptions of an ideal gas with respect to a reference state, assuming that $$C_p$$ and $$C_v$$ are constant.  For the current implementation this results in an EOS that is independent of the gas species.  The EOS can be described with $$\gamma, Cv, p^0, T^0$$ and $$e^0$$.

| Input     | Default Value | Argument |
| ----------|---------------|----------|
| $$\gamma$$| 2.4           | -gamma   |
| $$Cv$$    | 3.03 kJ/kg/K  | -Cv      |
| $$p^0$$   | $$10^7$$ Pa   | -p0      |
| $$T^0$$   | 311.1 C       | -T0      |
| $$e^0$$   | 1393 kJ/kg/K  | -e0      |

### Decode State
The decode state function computes the required values to compute fluxes from the Euler conserved variables.

#### Internal Energy
$$\begin{eqnarray}
e = e_t - KE
\end{eqnarray}$$

#### Pressure
$$\begin{eqnarray}
p = (\gamma - 1.0) \rho e - \gamma p^0
\end{eqnarray}$$

#### Speed of Sound
$$\begin{eqnarray}
a = \sqrt{\gamma (p+p^0)/\rho}
\end{eqnarray}$$

### Temperature
The temperature function computes T from $$e_t$$, $$\rho\vector{u}$$, and $$\rho$$.

$$\begin{eqnarray}
T = \frac{e - e^0}{C_v} + T^0
\end{eqnarray}$$

## Reference
- Jibben, Z., Velechovsky, J., Masser, T., Francois, M. M. (2019). "Modeling surface tension in compressible flow on an adaptively refined mesh." Computers and Mathematics with Applications, 78, 504-516.

