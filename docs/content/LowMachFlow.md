---
layout: default
title: Low Mach Flow Formulation
nav_order: 2
---

### Governing Equations
The low mach number flow formulation is based upon the work of J. Principe and R. Codina[1] reproduced here in dimensionless form. For simplicity:

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



### Weak Formulation
The non-dimensionalized weak form is provided (Ref [2]) for each governing equation where $$\left < f , g \right >  \equiv \int_{\Omega}^{} f g d\Omega$$ and $$\left < f , g \right >_\Gamma  \equiv \int_{\Gamma}^{} f g d\Gamma $$ notation is introduced.
#### Test Function $$q$$

$$
\left < q, S \frac{\partial \rho}{\partial t} \right > + \left < q, \nabla \cdot \left(\rho \boldsymbol{u} \right ) \right > = 0 $$

#### Test Function $$v$$

$$
\left < \boldsymbol{v}, \rho S \frac{\partial \boldsymbol{u}}{\partial t} \right > + \left < \boldsymbol{v}, \rho \boldsymbol{u} \cdot \nabla \boldsymbol{u} \right > + \left < \frac{2\mu\boldsymbol{\epsilon'}{}(\boldsymbol{u})}{R}, \nabla^S \boldsymbol{v}  \right > - \left < p, \nabla \cdot \boldsymbol{v} \right > = \left < -\frac{1}{F^2}\rho \boldsymbol{\hat{z}}, \boldsymbol{v} \right > + \left < \boldsymbol{t_n}, \boldsymbol{v} \right >_\Gamma\\

\boldsymbol{t_n} = \left( -p\boldsymbol{I}+ \frac{2\mu\boldsymbol{\epsilon'}{}(\boldsymbol{u})}{R} \right) \cdot \boldsymbol{n}
$$

#### Test Function $$w$$

$$
\left< \rho c_p S \frac{\partial T}{\partial t} , w \right> + \left< \rho c_p \boldsymbol{u} \cdot \nabla T, w \right> + \left<\frac{k}{P}\nabla T, w \right> - \left<\Gamma \beta S T \frac{dp^{th}}{dt}, w \right> = \left<HSQ, w \right> + \left<q_n, w \right>_\Gamma \\

q_n = \frac{k}{p} \nabla T \cdot \boldsymbol{n}
$$

## References
1. Principe, J., & Codina, R. (2009). Mathematical models for thermally coupled low speed flows. Advances in Theoretical and Applied Mechanics, 2(2), 93-112.
2. Avila, M., Principe, J., & Codina, R. (2011). A finite element dynamical nonlinear subscale approximation for the low Mach number flow equations. Journal of Computational Physics, 230(22), 7988-8009.