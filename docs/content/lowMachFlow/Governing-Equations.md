---
layout: default
title: Governing Equations
parent: Low Mach Flow Formulation
nav_order: 1
---

### Governing Equations
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
| $$T$$ | Absolute Temperature | $$Q = \frac{Q^*}{Q_o}$$ |
| $$C_p$$ | Specific Heat Capacity | $$C_p = \frac{C_p^*}{ {C_p}_o}$$ |
| $$\beta$$ | Thermal expansion coefficient | $$\beta = \beta^* T_o$$ |
| $$\boldsymbol{x}$$ | Spacial Location | $$\boldsymbol{x} = \frac{\boldsymbol{x}^*}{l_o}$$ |

$$p_o$$, $$\rho_o$$, and $$T_o$$ are assumed to be related by the equation of state.

$$\begin{eqnarray}
S\frac{\partial \rho}{\partial t} + \nabla \cdot \left(\rho \boldsymbol{u} \right ) = 0 \\
\rho \left(S\frac{\partial \boldsymbol{u}}{\partial t} + \boldsymbol{u} \cdot \nabla \boldsymbol{u}) \right ) + \nabla p - \frac{1}{R}\nabla \cdot \left(2 \mu \boldsymbol{\epsilon'}{}(\boldsymbol{u})\right ) = -\frac{1}{F}\rho\hat{\boldsymbol{z}} \\
\rho c_p \left(S \frac{\partial T}{\partial t} + \boldsymbol{u} \cdot \nabla T \right ) - \Gamma \beta T S \frac{d p^{th}}{dt} - \frac{1}{P} \nabla \cdot \left (k \nabla T \right ) = HSQ
\end{eqnarray}$$

where:

$$\begin{eqnarray}
\boldsymbol{\epsilon'}{}(\boldsymbol{u}) = \boldsymbol{\epsilon}{}(\boldsymbol{u}) - \frac{1}{3}\left(\nabla \cdot \boldsymbol{u} \right )\boldsymbol{I} \\
\boldsymbol{\epsilon}{}(\boldsymbol{u}) = \frac{1}{2} \left(\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T \right )
\end{eqnarray}$$

where the non-dimensional quantities were defined:

| Symbol      | Name | Definition |
| ----------- | ----------- |----------- |
| $$S$$       | Strouhal     | $$\frac{l_o}{u_o t_o}$$ |
| $$R$$       | Reynolds     | $$\frac{\rho_o u_o l_o}{\mu_o}$$ |
| $$F$$       | Froude       | $$\frac{u_o}{\sqrt{g_o l_o}}$$|
| $$P$$       | Péclet       | $$\frac{rho_o {C_p}_o u_o l_o}{k_o}$$|
| $$H$$       | heat release     | $$\frac{t_o Q_o}{\rho_o {C_p}_o T_o}$$ |
| $$\Gamma$$  | *depends upon state equation* | $$\frac{p_o}{\rho_o {C_p}_o T_o}$$  |

The ideal gas equation of state is assumed where $$\rho$$ is a function of $$p^{th}$$ and $$T$$ only, $$\rho^*=\frac{p^{th*}}{RT^*}$$. From the relationship between $$p_o$$, $$\rho_o$$, and $$T_o$$ we know $$ \rho_o=\frac{p_o}{R_o}$$. Combining this with the the definition of the non-dimensional quantities

$$ \rho^*=\frac{p^{th*}}{RT^*} \Rightarrow  \rho\rho_o=\frac{p^{th}p_o}{RTT_o} \Rightarrow p^{th}=\rho T $$

## References
 - Principe, J., & Codina, R. (2009). Mathematical models for thermally coupled low speed flows. Advances in Theoretical and Applied Mechanics, 2(2), 93-112.
