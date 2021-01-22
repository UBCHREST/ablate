---
layout: default
title: Residual Terms
parent: Low Mach Flow Formulation
nav_order: 3
---
Provides the term to evaluate and multiply by the test function/test gradient function at each quadrature location.  The Residual and Jacobians are in terms of

### Q Test Function

$$
F_q = \int_\Omega q \left( S \frac{\partial \rho}{\partial t} + \nabla \cdot \left(\rho \boldsymbol{u} \right)\right)  d\Omega
$$

We know that $$\rho = \frac{p^{th}}{T}$$.

$$
 S\frac{\partial \rho}{\partial t} \Rightarrow \frac{S}{T} \frac{\partial p^{th}}{\partial t}-\frac{Sp^{th}}{T^2}\frac{\partial T}{\partial t}
$$

$$
\nabla \cdot \left(\rho \boldsymbol{u} \right) \Rightarrow
\nabla \cdot \left(\frac{p^{th}}{T} \boldsymbol{u} \right) \Rightarrow
p^{th} \left( \frac{1}{T} \nabla \cdot \boldsymbol{u} + \boldsymbol{u} \cdot \nabla \frac{1}{T} \right) \Rightarrow
p^{th} \left( \frac{1}{T} \nabla \cdot \boldsymbol{u} - \frac{1}{T^2}\boldsymbol{u} \cdot \nabla T \right)
$$

Combining the terms and assuming that $$\frac{\partial p^{th}}{\partial t} = 0 $$ results in

$$
F_q = q \left(-\frac{Sp^{th}}{T^2}\frac{\partial T}{\partial t} + \frac{p^{th}}{T} \nabla \cdot \boldsymbol{u} - \frac{p^{th}}{T^2}\boldsymbol{u} \cdot \nabla T \right)  d\Omega
$$

### Jacobians

$$
F_{q,i} = \int \phi_i S \frac{\partial}{\partial t}\frac{p^{th}}{T} + \phi_i \nabla \cdot \left(\frac{p^{th}}{T}\boldsymbol{u} \right) \\
\\
\frac{F_{q,i}}{\partial c_{T,j}} = \int \phi_i S \frac{\partial}{\partial c_{T,j}}\frac{\partial}{\partial t}\frac{p^{th}}{T} + \phi_i \frac{\partial}{\partial c_{T,j}}\nabla \cdot \left(\frac{p^{th}}{T}\boldsymbol{u} \right)
$$

$$\begin{eqnarray}
\int \phi_i S \frac{\partial}{\partial c_{T,j}}\frac{\partial}{\partial t}\frac{p^{th}}{T} &=& \int \phi_i S p^{th} \frac{\partial}{\partial c_{T,j}}\frac{\partial}{\partial t}\frac{1}{T} \\
&=& \int \frac{-\phi_i S  p^{th}}{T^2} \frac{\partial}{\partial c_{T,j}}\frac{\partial T}{\partial t} \\
&=& \int \frac{-\phi_i S  p^{th}}{T^2} \frac{\partial}{\partial t}\frac{\partial T}{\partial c_{T,j}} \\
&=& \int \frac{-\phi_i S  p^{th}}{T^2} \frac{\partial \psi_{T,j}}{\partial t}
\end{eqnarray}$$

$$\begin{eqnarray}
\int \phi_i \frac{\partial}{\partial c_{T,j}}\nabla \cdot \left(\frac{p^{th}}{T}\boldsymbol{u} \right) 
&=& \int \phi_i p^{th} \frac{\partial}{\partial c_{T,j}} \left( \frac{1}{T}\nabla \cdot \boldsymbol{u} + \boldsymbol{u}\cdot \nabla \frac{1}{T} \right) \\
&=& \int \phi_i p^{th} \left( \frac{\partial \frac{1}{T}}{\partial c_{T,j}} \nabla \cdot \boldsymbol{u} + \boldsymbol{u}\cdot \frac{\partial}{\partial c_{T,j}} \nabla \frac{1}{T} \right) \\
&=& \int \phi_i p^{th} \left( - \frac{1}{T^2} \frac{\partial T}{\partial c_{T,j}}  \nabla \cdot \boldsymbol{u} + \boldsymbol{u}\cdot \frac{\partial}{\partial c_{T,j}} \nabla \frac{1}{T} \right) \\
&=& \int \phi_i p^{th} \left( - \frac{1}{T^2} \frac{\partial T}{\partial c_{T,j}}  \nabla \cdot \boldsymbol{u} - \boldsymbol{u}\cdot \frac{\partial}{\partial c_{T,j}} \frac{\nabla T}{T^2} \right) \\
&=& \int \phi_i p^{th} \left( - \frac{1}{T^2} \frac{\partial T}{\partial c_{T,j}}  \nabla \cdot \boldsymbol{u} - \boldsymbol{u}\cdot \left(\frac{1}{T^2} \frac{\partial \nabla T}{\partial c_{T,j}} +  \frac{\partial \frac{1}{T^2}}{\partial c_{T,j}} \nabla T \right)\right) \\
&=& \int \phi_i p^{th} \left( - \frac{1}{T^2} \frac{\partial T}{\partial c_{T,j}}  \nabla \cdot \boldsymbol{u} - \boldsymbol{u}\cdot \left(\frac{1}{T^2} \frac{\partial \nabla T}{\partial c_{T,j}} - \frac{2}{T^3}  \frac{\partial T}{\partial c_{T,j}} \nabla T \right)\right) \\
&=& \int \phi_i p^{th} \left( - \frac{1}{T^2} \frac{\partial T}{\partial c_{T,j}}  \nabla \cdot \boldsymbol{u} - \boldsymbol{u}\cdot \left(\frac{1}{T^2} \frac{\partial \nabla T}{\partial c_{T,j}} - \frac{2}{T^3}  \frac{\partial T}{\partial c_{T,j}} \nabla T \right)\right) \\
&=& \int \frac{\phi_i p^{th}}{T^2} \left( - \psi_{T,j}  \nabla \cdot \boldsymbol{u} + \boldsymbol{u}\cdot \left(\frac{2}{T} \psi_{T,j} \nabla T - \nabla \psi_{T,j}\right) \right) \\
\end{eqnarray}$$

$$
\therefore \frac{F_{q,i}}{\partial c_{T,j}} = \int \frac{-\phi_i S  p^{th}}{T^2} \frac{\partial \psi_{T,j}}{\partial t} + \int \frac{\phi_i p^{th}}{T^2} \left( - \psi_{T,j}  \nabla \cdot \boldsymbol{u} + \boldsymbol{u}\cdot \left(\frac{2}{T} \psi_{T,j} \nabla T - \nabla \psi_{T,j}\right) \right)
$$

$$\begin{eqnarray}
\frac{F_{q,i}}{\partial c_{u_c,j}}
&=& \phi_i \frac{\partial}{\partial c_{u_c,j}}\nabla \cdot \left(\frac{p^{th}}{T}\boldsymbol{u} \right) \\
&=&  \phi_i \frac{\partial}{\partial c_{u_c,j}} \left(\rho \nabla \cdot \boldsymbol{u} + \boldsymbol{u} \cdot \nabla \rho \right) \\
&=&  \phi_i \left(\rho \frac{\partial \nabla \cdot \boldsymbol{u}}{\partial c_{u_c,j}} + \frac{\partial \boldsymbol{u}}{\partial c_{u_c,j}} \cdot \nabla \rho \right) \\
&=&  \phi_i \rho \frac{\partial \psi_{u_c,j}}{\partial x_c}  + \phi_i \psi_{u_c,j} \hat{e}_j \cdot \nabla \rho \\
&=&  \phi_i \rho \frac{\partial \psi_{u_c,j}}{\partial x_c}  + \phi_i \psi_{u_c,j} \frac{\partial rho}{\partial x_c} \\
&=&  \phi_i \frac{p^{th}}{T} \frac{\partial \psi_{u_c,j}}{\partial x_c}  + \phi_i \psi_{u_c,j} p^{th} \frac{\partial \frac{1}{T}}{\partial x_c} \\
&=&  \phi_i \frac{p^{th}}{T} \frac{\partial \psi_{u_c,j}}{\partial x_c}  - \phi_i \psi_{u_c,j} \frac{p^{th}}{T^2} \frac{\partial T}{\partial x_c} 
\end{eqnarray}$$


[comment]: <> (### $$\boldsymbol{V}$$ Test Function)

[comment]: <> ($$)

[comment]: <> ( v_i \rho S \frac{u_i}{t} \Rightarrow v_i \frac{S p^{th}}{T} \frac{u_i}{t})

[comment]: <> ($$)


[comment]: <> (### Jacobians)

[comment]: <> ($$ \int_\Omega \phi g_0&#40;u, u_t, \nabla u, x, t&#41; \psi + \phi {\vec g}_1&#40;u, u_t, \nabla u, x, t&#41; \nabla \psi + \nabla\phi \cdot {\vec g}_2&#40;u, u_t, \nabla u, x, t&#41; \psi + \nabla\phi \cdot {\overleftrightarrow g}_3&#40;u, u_t, \nabla u, x, t&#41; \cdot \nabla \psi $$)

[comment]: <> (### $$ q - \boldsymbol{u} $$:)

[comment]: <> ($$  g_o = - \frac{p^{th}}{T^2}\nabla T $$)

[comment]: <> ($$  \boldsymbol{g_1} = - \frac{p^{th}}{T^2}\nabla T $$)
