---
layout: default
title: Non-Dimensional Terms
parent: Flow Formulations
nav_order: 1
---

The low mach number flow formulation is based upon the work of J. Principe and R. Codina reproduced here in dimensionless form. For simplicity:

- the non-dimensional form is written without any additional super/subscript
- the dimensional (tradition with units) values uses a * (e.g. $$T^*$$)
- and the characteristic scales are denoted with a o (e.g. $$T_o$$)

| Dimensionless Form    | Name | Definition | 
| ----------- | ----------- | -----------|
| $$\rho$$    | Density     | $$\rho = \frac{\rho^*}{\rho_o}$$ |
| $$p$$   | Mechanical Pressure | $$p = \frac{p^*}{p_o}$$ |
| $$p^{th}$$   | Thermodynamic Pressure | $$p^{th} = \frac{p^{th*}}{p_o}$$ |
| $$\boldsymbol{u}$$ | Velocity | $$\boldsymbol{u} = \frac{\boldsymbol{u}^*}{u_o}$$ |
| $$\mu$$ | Viscosity | $$\mu = \frac{\mu^*}{\mu_o}$$ |
| $$k$$ | Thermal Conductivity | $$k = \frac{k^*}{k_o}$$ |
| $$Q$$ | External Heating | $$Q = \frac{Q^*}{Q_o}$$ |
| $$T$$ | Absolute Temperature | $$T = \frac{T^*}{T_o}$$ |
| $$C_p$$ | Specific Heat Capacity | $$C_p = \frac{C_p^*}{ {C_p}_o}$$ |
| $$\beta$$ | Thermal expansion coefficient | $$\beta = \beta^* T_o$$ |
| $$\boldsymbol{x}$$ | Spacial Location | $$\boldsymbol{x} = \frac{\boldsymbol{x}^*}{l_o}$$ |

$$p_o$$, $$\rho_o$$, and $$T_o$$ are assumed to be related by the equation of state.

The non-dimensional quantities are defined as:

| Symbol      | Name | Definition |
| ----------- | ----------- |----------- |
| $$S$$       | Strouhal     | $$\frac{l_o}{u_o t_o}$$ |
| $$R$$       | Reynolds     | $$\frac{\rho_o u_o l_o}{\mu_o}$$ |
| $$F$$       | Froude       | $$\frac{u_o}{\sqrt{g_o l_o}}$$|
| $$P$$       | Péclet       | $$\frac{rho_o {C_p}_o u_o l_o}{k_o}$$|
| $$H$$       | heat release     | $$\frac{t_o Q_o}{\rho_o {C_p}_o T_o}$$ |
| $$\Gamma$$  | *depends upon state equation* | $$\frac{p_o}{\rho_o {C_p}_o T_o}$$  |

## References
- Principe, J., & Codina, R. (2009). Mathematical models for thermally coupled low speed flows. Advances in Theoretical and Applied Mechanics, 2(2), 93-112.
