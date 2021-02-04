---
layout: default
title: Governing Equations
parent: Low Mach Flow Formulation
grand_parent: Flow Formulations
nav_order: 1
---

### Governing Equations
The low mach number flow formulation is based upon the work of J. Principe and R. Codina reproduced here in dimensionless form (see [Non-Dimensional Terms]({{ site.baseurl}}{%link content/formulations/NonDimensional.md %}).).

$$\begin{eqnarray}
S\frac{\partial \rho}{\partial t} + \nabla \cdot \left(\rho \boldsymbol{u} \right ) = 0 \\
\rho \left(S\frac{\partial \boldsymbol{u}}{\partial t} + \boldsymbol{u} \cdot \nabla \boldsymbol{u}) \right ) + \nabla p - \frac{1}{R}\nabla \cdot \left(2 \mu \boldsymbol{\epsilon'}{}(\boldsymbol{u})\right ) = -\frac{1}{F^2}\rho\hat{\boldsymbol{z}} \\
\rho c_p \left(S \frac{\partial T}{\partial t} + \boldsymbol{u} \cdot \nabla T \right ) - \Gamma \beta T S \frac{d p^{th}}{dt} - \frac{1}{P} \nabla \cdot \left (k \nabla T \right ) = HSQ
\end{eqnarray}$$

where:

$$\begin{eqnarray}
\boldsymbol{\epsilon'}{}(\boldsymbol{u}) = \boldsymbol{\epsilon}{}(\boldsymbol{u}) - \frac{1}{3}\left(\nabla \cdot \boldsymbol{u} \right )\boldsymbol{I} \\
\boldsymbol{\epsilon}{}(\boldsymbol{u}) = \frac{1}{2} \left(\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T \right )
\end{eqnarray}$$


The ideal gas equation of state is assumed where $$\rho$$ is a function of $$p^{th}$$ and $$T$$ only, $$\rho^*=\frac{p^{th*}}{RT^*}$$. From the relationship between $$p_o$$, $$\rho_o$$, and $$T_o$$ we know $$ \rho_o=\frac{p_o}{R_o}$$. Combining this with the the definition of the non-dimensional quantities

$$ \rho^*=\frac{p^{th*}}{RT^*} \Rightarrow  \rho\rho_o=\frac{p^{th}p_o}{RTT_o} \Rightarrow p^{th}=\rho T $$

## References
 - Principe, J., & Codina, R. (2009). Mathematical models for thermally coupled low speed flows. Advances in Theoretical and Applied Mechanics, 2(2), 93-112.
