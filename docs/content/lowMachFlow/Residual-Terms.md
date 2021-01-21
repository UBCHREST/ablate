---
layout: default
title: Residual Terms
parent: Low Mach Flow Formulation
nav_order: 3
---
Provides the term to evaluate and multiply by the test function/test gradient function at each quadrature location.

### Q Test Function

$$
F_q = \int_\Omega q \left( S \frac{\partial \rho}{\partial t} + \nabla \cdot \left(\rho \boldsymbol{u} \right)\right)  d\Omega
$$

We know that $$\rho = \frac{p^{th}}{T}$$.

$$
 S\frac{\partial \rho}{\partial t} \Rightarrow \frac{1}{T} \frac{\partial p^{th}}{\partial t}-\frac{p^{th}}{T^2}\frac{\partial T}{\partial t}
$$

$$
\nabla \cdot \left(\rho \boldsymbol{u} \right) \Rightarrow
\nabla \cdot \left(\frac{p^{th}}{T} \boldsymbol{u} \right) \Rightarrow
p^{th} \left( \frac{1}{T} \nabla \cdot \boldsymbol{u} + \boldsymbol{u} \cdot \nabla \frac{1}{T} \right) \Rightarrow
p^{th} \left( \frac{1}{T} \nabla \cdot \boldsymbol{u} - \frac{1}{T^2}\boldsymbol{u} \cdot \nabla T \right)
$$

Combining the terms and assuming that $$\frac{\partial p^{th}}{\partial t} = 0 $$ results in

$$
F_q = q \left(-\frac{p^{th}}{T^2}\frac{\partial T}{\partial t} + \frac{p^{th}}{T} \nabla \cdot \boldsymbol{u} - \frac{p^{th}}{T^2}\boldsymbol{u} \cdot \nabla T \right)  d\Omega
$$

### Jacobians


### $$\boldsymbol{V}$$ Test Function

$$
 v_i \rho S \frac{u_i}{t} \Rightarrow v_i \frac{S p^{th}}{T} \frac{u_i}{t}
$$


### Jacobians
$$ \int_\Omega \phi g_0(u, u_t, \nabla u, x, t) \psi + \phi {\vec g}_1(u, u_t, \nabla u, x, t) \nabla \psi + \nabla\phi \cdot {\vec g}_2(u, u_t, \nabla u, x, t) \psi + \nabla\phi \cdot {\overleftrightarrow g}_3(u, u_t, \nabla u, x, t) \cdot \nabla \psi $$

### $$ q - \boldsymbol{u} $$:

$$  g_o = - \frac{p^{th}}{T^2}\nabla T $$

$$  \boldsymbol{g_1} = - \frac{p^{th}}{T^2}\nabla T $$
