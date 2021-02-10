---
layout: default
title: Governing Equations
parent: Incompressible Flow Formulation
grand_parent: Flow Formulations
nav_order: 3
---

### Governing Equations
The incompressible low mach flow formulation is based upon the work of J. Principe and R. Codina reproduced here in dimensionless form.  The definitions for the dimensionless terms are provided in [Non-Dimensional Terms]({{ site.baseurl}}{%link content/formulations/NonDimensional.md %}).

$$\begin{eqnarray}
\nabla \cdot \boldsymbol{u} = 0 \\
\rho \left(S\frac{\partial \boldsymbol{u}}{\partial t} + \boldsymbol{u} \cdot \nabla \boldsymbol{u} \right ) + \nabla p - \frac{1}{R}\nabla \cdot \left(2 \mu \boldsymbol{\epsilon}{}(\boldsymbol{u})\right ) = 0\\
\rho c_p \left(S \frac{\partial T}{\partial t} + \boldsymbol{u} \cdot \nabla T \right ) - \Gamma \beta T S \frac{d p^{th}}{dt} - \frac{1}{P} \nabla \cdot \left (k \nabla T \right ) = HSQ
\end{eqnarray}$$

where:

$$\begin{eqnarray}
\boldsymbol{\epsilon}{}(\boldsymbol{u}) = \frac{1}{2} \left(\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T \right )
\end{eqnarray}$$

### Weak Formulation
The non-dimensionalized weak form is provided for each governing equation where $$\left < f , g \right >  \equiv \int_{\Omega}^{} f g d\Omega$$ and $$\left < f , g \right >_\Gamma  \equiv \int_{\Gamma}^{} f g d\Gamma $$ notation is introduced.
#### Test Function $$q$$

$$ \left < q, \nabla \cdot  \boldsymbol{u} \right > = 0 $$

#### Test Function $$v$$

$$
\left < \boldsymbol{v}, \rho S \frac{\partial \boldsymbol{u}}{\partial t} \right > + \left < \boldsymbol{v}, \rho \boldsymbol{u} \cdot \nabla \boldsymbol{u} \right > + \left < \frac{2\mu\boldsymbol{\epsilon}{}(\boldsymbol{u})}{R}, \nabla^S \boldsymbol{v}  \right > - \left < p, \nabla \cdot \boldsymbol{v} \right > =  \left < \boldsymbol{t_n}, \boldsymbol{v} \right >_\Gamma\\

\boldsymbol{t_n} = \left( -p\boldsymbol{I}+ \frac{2\mu\boldsymbol{\epsilon}{}(\boldsymbol{u})}{R} \right) \cdot \boldsymbol{n}
$$

#### Test Function $$w$$

$$
\left< \rho c_p S \frac{\partial T}{\partial t} , w \right> + \left< \rho c_p \boldsymbol{u} \cdot \nabla T, w \right> + \left<\frac{k}{P}\nabla T, \nabla w \right> - \left<\Gamma \beta S T \frac{dp^{th}}{dt}, w \right> = \left<HSQ, w \right> + \left<q_n, w \right>_\Gamma \\

q_n = \frac{k}{p} \nabla T \cdot \boldsymbol{n}
$$

## References
 - Avila, M., Principe, J., & Codina, R. (2011). A finite element dynamical nonlinear subscale approximation for the low Mach number flow equations. Journal of Computational Physics, 230(22), 7988-8009.
 - Principe, J., & Codina, R. (2009). Mathematical models for thermally coupled low speed flows. Advances in Theoretical and Applied Mechanics, 2(2), 93-112.
