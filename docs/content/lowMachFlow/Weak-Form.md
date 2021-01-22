---
layout: default
title: Weak Form
parent: Low Mach Flow Formulation
nav_order: 2
---

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
\left< \rho c_p S \frac{\partial T}{\partial t} , w \right> + \left< \rho c_p \boldsymbol{u} \cdot \nabla T, w \right> + \left<\frac{k}{P}\nabla T, \nabla w \right> - \left<\Gamma \beta S T \frac{dp^{th}}{dt}, w \right> = \left<HSQ, w \right> + \left<q_n, w \right>_\Gamma \\

q_n = \frac{k}{p} \nabla T \cdot \boldsymbol{n}
$$

## References
- Avila, M., Principe, J., & Codina, R. (2011). A finite element dynamical nonlinear subscale approximation for the low Mach number flow equations. Journal of Computational Physics, 230(22), 7988-8009.
