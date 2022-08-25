---
layout: default
title: Perfect Gas
parent: Chemistry
nav_order: 1
---

The perfect gas ("perfectGas") assumptions of an ideal gas simplify the equations assuming that $$C_p$$ and $$C_v$$ are constant.  For the current implementation this results in an EOS that is independent of the gas species.  The EOS can be described with only $$\gamma$$ and $$R$$.

| Input      | Definition           | Default Value |  Argument |
| ---------- | -------------------- | ------------- | --------- |
| $$\gamma$$ | $$\gamma = C_p/C_v$$ | 1.4           | -gamma    |
| $$R$$      | $$R = C_p - C_v$$    | 287.0         | -Rgas     |

### Decode State
The decode state function computes the required values to compute fluxes from the Euler conserved variables.

#### Internal Energy
$$\begin{eqnarray}
e = e_t - KE
\end{eqnarray}$$

#### Pressure
$$\begin{eqnarray}
p = (\gamma - 1.0) \rho e
\end{eqnarray}$$

#### Speed of Sound
$$\begin{eqnarray}
a = \sqrt{\gamma p/\rho}
\end{eqnarray}$$

### Temperature
The temperature function computes T from $$e_t$$, $$\rho\vector{u}$$, and $$\rho$$.

$$\begin{eqnarray}
T = \frac{e}{C_v} \\
C_v = \frac{R}{\gamma -1}
\end{eqnarray}$$