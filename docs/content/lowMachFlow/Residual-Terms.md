---
layout: default
title: Residual Terms
parent: Low Mach Flow Formulation
nav_order: 3
---
Provides the term to evaluate and multiply by the test function/test gradient function at each quadrature location.  The Residual and Jacobians are in terms of

### Q Test Function

$$
F_q = \int_\Omega q \left( S \frac{\partial \rho}{\partial t} + \nabla \cdot \left(\rho \boldsymbol{u} \right)\right)  d\Omega = 0
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
F_q = \int_\Omega q \left(-\frac{Sp^{th}}{T^2}\frac{\partial T}{\partial t} + \frac{p^{th}}{T} \nabla \cdot \boldsymbol{u} - \frac{p^{th}}{T^2}\boldsymbol{u} \cdot \nabla T \right)  d\Omega
$$

#### Jacobians

$$
F_{q,i} = \int \phi_i S \frac{\partial}{\partial t}\frac{p^{th}}{T} + \phi_i \nabla \cdot \left(\frac{p^{th}}{T}\boldsymbol{u} \right) \\
$$

$$\begin{eqnarray}
\frac{F_{q,i}}{\partial c_{T,j}}
&=& \int \phi_i \frac{\partial}{\partial c_{T,j}}\nabla \cdot \left(\frac{p^{th}}{T}\boldsymbol{u} \right)\\
&=& \int \phi_i p^{th} \frac{\partial}{\partial c_{T,j}} \left( \frac{1}{T}\nabla \cdot \boldsymbol{u} + \boldsymbol{u}\cdot \nabla \frac{1}{T} \right) \\
&=& \int \phi_i p^{th} \left( \frac{\partial \frac{1}{T}}{\partial c_{T,j}} \nabla \cdot \boldsymbol{u} + \boldsymbol{u}\cdot \frac{\partial}{\partial c_{T,j}} \nabla \frac{1}{T} \right) \\
&=& \int \phi_i p^{th} \left( - \frac{1}{T^2} \frac{\partial T}{\partial c_{T,j}}  \nabla \cdot \boldsymbol{u} + \boldsymbol{u}\cdot \frac{\partial}{\partial c_{T,j}} \nabla \frac{1}{T} \right) \\
&=& \int \phi_i p^{th} \left( - \frac{1}{T^2} \frac{\partial T}{\partial c_{T,j}}  \nabla \cdot \boldsymbol{u} - \boldsymbol{u}\cdot \frac{\partial}{\partial c_{T,j}} \frac{\nabla T}{T^2} \right) \\
&=& \int \phi_i p^{th} \left( - \frac{1}{T^2} \frac{\partial T}{\partial c_{T,j}}  \nabla \cdot \boldsymbol{u} - \boldsymbol{u}\cdot \left(\frac{1}{T^2} \frac{\partial \nabla T}{\partial c_{T,j}} +  \frac{\partial \frac{1}{T^2}}{\partial c_{T,j}} \nabla T \right)\right) \\
&=& \int \phi_i p^{th} \left( - \frac{1}{T^2} \frac{\partial T}{\partial c_{T,j}}  \nabla \cdot \boldsymbol{u} - \boldsymbol{u}\cdot \left(\frac{1}{T^2} \frac{\partial \nabla T}{\partial c_{T,j}} - \frac{2}{T^3}  \frac{\partial T}{\partial c_{T,j}} \nabla T \right)\right) \\
&=& \int \frac{\phi_i p^{th}}{T^2} \left( - \psi_{T,j}  \nabla \cdot \boldsymbol{u} + \boldsymbol{u}\cdot \left(\frac{2}{T} \psi_{T,j} \nabla T - \nabla \psi_{T,j}\right) \right) \\
\end{eqnarray}$$

$$\begin{eqnarray}
\frac{F_{q,i}}{\partial c_{\frac{\partial T}{\partial t},j}} &=& \int \phi_i S \frac{\partial}{\partial c_{T,j}}\frac{\partial}{\partial t}\frac{p^{th}}{T} \\
&=& \int \phi_i S p^{th} \frac{\partial}{\partial c_{T,j}}\frac{\partial}{\partial t}\frac{1}{T} \\
&=& \int \frac{-\phi_i S  p^{th}}{T^2} \frac{\partial}{\partial c_{T,j}}\frac{\partial T}{\partial t} \\
&=& \int \frac{-\phi_i S  p^{th}}{T^2} \psi_j \\
\end{eqnarray}$$

$$\begin{eqnarray}
\frac{F_{q,i}}{\partial c_{u_c,j}}
&=& \int \phi_i \frac{\partial}{\partial c_{u_c,j}}\nabla \cdot \left(\frac{p^{th}}{T}\boldsymbol{u} \right) \\
&=& \int \phi_i \frac{\partial}{\partial c_{u_c,j}} \left(\rho \nabla \cdot \boldsymbol{u} + \boldsymbol{u} \cdot \nabla \rho \right) \\
&=& \int \phi_i \left(\rho \frac{\partial \nabla \cdot \boldsymbol{u}}{\partial c_{u_c,j}} + \frac{\partial \boldsymbol{u}}{\partial c_{u_c,j}} \cdot \nabla \rho \right) \\
&=& \int \phi_i \rho \frac{\partial \psi_{u_c,j}}{\partial x_c}  + \phi_i \psi_{u_c,j} \hat{e}_j \cdot \nabla \rho \\
&=& \int \phi_i \rho \frac{\partial \psi_{u_c,j}}{\partial x_c}  + \phi_i \psi_{u_c,j} \frac{\partial \rho}{\partial x_c} \\
&=& \int \phi_i \frac{p^{th}}{T} \frac{\partial \psi_{u_c,j}}{\partial x_c}  + \phi_i \psi_{u_c,j} p^{th} \frac{\partial \frac{1}{T}}{\partial x_c} \\
&=& \int \phi_i \frac{p^{th}}{T} \frac{\partial \psi_{u_c,j}}{\partial x_c}  - \phi_i \psi_{u_c,j} \frac{p^{th}}{T^2} \frac{\partial T}{\partial x_c} 
\end{eqnarray}$$

### W Test Function

$$
\require{cancel}
\begin{eqnarray}
F_w &=& \int_\Omega w \rho C_p S \frac{\partial T}{\partial t} + w \rho C_p \boldsymbol{u} \cdot \nabla T + \nabla w \cdot \frac{k}{P} \nabla T - \cancelto{0}{w \Gamma \beta S T \frac{\partial p^{th}}{\partial t}} - wHSQ - \int_\Gamma w q_n = 0 \\
&=& \int_\Omega w \frac{C_p S p^{th}}{T} \frac{\partial T}{\partial t} + \ w \frac{C_p p^{th}}{T} \boldsymbol{u} \cdot \nabla T + \nabla w \cdot \frac{k}{P} \nabla T - wHSQ - \int_\Gamma w q_n
\end{eqnarray}$$

#### Jacobians

$$\begin{eqnarray}
\require{enclose}
F_{w,i} &=& \int_\Omega \phi_i \frac{C_p S p^{th}}{T} \frac{\partial T}{\partial t} + \phi_i \frac{C_p p^{th}}{T} \boldsymbol{u} \cdot \nabla T + \nabla \phi_i \cdot \frac{k}{P} \nabla T - wHSQ - \int_\Gamma \phi_i q_n \\
&=& \int_\Omega \enclose{circle}{1} + \enclose{circle}{2} + \enclose{circle}{3} - wHSQ - \int_\Gamma \phi_i q_n
\end{eqnarray}$$

$$\begin{eqnarray}
\frac{\partial F_{w,i}}{\partial c_{u_c,j}} &=& \frac{\partial}{\partial c_{u_c,j}} \phi_i \frac{C_p p^{th}}{T} \boldsymbol{u} \cdot \nabla T \\
&=&  \phi_i \frac{C_p p^{th}}{T} \frac{\partial \boldsymbol{u}}{\partial c_{u_c,j}} \cdot \nabla T \\
&=&  \phi_i \frac{C_p p^{th}}{T} \psi_{u_c,j} \hat{e}_c \cdot \nabla T \\
&=&  \phi_i \frac{C_p p^{th}}{T} \psi_{u_c,j} \frac{\partial T}{\partial x_c}
\end{eqnarray}$$

$$\begin{eqnarray}
\frac{\partial F_{w,i}}{\partial c_{T,j}} &=& \frac{\partial \enclose{circle}{1}}{\partial c_{T,j}} + \frac{\partial \enclose{circle}{2}}{\partial c_{T,j}} + \frac{\partial \enclose{circle}{3}}{\partial c_{T,j}}\\
\end{eqnarray}$$

$$\begin{eqnarray}
\frac{\partial \enclose{circle}{1}}{\partial c_{T,j}} &=& \frac{\partial }{\partial c_{T,j}} \phi_i \frac{C_p S p^{th}}{T} \frac{\partial T}{\partial t} \\
&=& \psi_i C_p S p^{th} \frac{\partial }{\partial c_{T,j}} \left(\frac{1}{T} \frac{\partial T}{\partial t} \right) \\
&=& \psi_i C_p S p^{th} \left(\frac{\partial T}{\partial t} \frac{\partial }{\partial c_{T,j}} \frac{1}{T} \right) \\
&=& -\psi_i C_p S p^{th} \left(\frac{\partial T}{\partial t} \frac{1}{T^2} \frac{\partial T}{\partial c_{T,j}} \right) \\
&=& -\psi_i C_p S p^{th} \frac{\partial T}{\partial t} \frac{1}{T^2} \psi_{T,j}
\end{eqnarray}$$

$$\begin{eqnarray}
\frac{\partial \enclose{circle}{2}}{\partial c_{T,j}} &=& \frac{\partial }{\partial c_{T,j}} \phi_i \frac{C_p p^{th}}{T} \boldsymbol{u} \cdot \nabla T \\
&=& \psi_i C_p p^{th} \boldsymbol{u} \cdot  \frac{\partial}{\partial c_{T,j}}\left(\frac{\nabla T}{T} \right) \\
&=& \psi_i C_p p^{th} \boldsymbol{u} \cdot  \left( \frac{1}{T} \frac{\partial \nabla T}{\partial c_{T,j}} + \nabla T \frac{\partial \frac{1}{T}}{\partial c_{T,j}}  \right) \\
&=& \psi_i C_p p^{th} \boldsymbol{u} \cdot  \left( \frac{1}{T} \frac{\partial \nabla T}{\partial c_{T,j}} - \frac{\nabla T}{T^2} \frac{\partial T}{\partial c_{T,j}} \right) \\
&=& \psi_i C_p p^{th} \boldsymbol{u} \cdot  \left( \frac{1}{T} \nabla \psi_{T,j} - \frac{\nabla T}{T^2} \psi_{T,j} \right)
\end{eqnarray}$$

$$\begin{eqnarray}
\frac{\partial \enclose{circle}{3}}{\partial c_{T,j}} &=& \frac{\partial \enclose{circle}{3}}{\partial c_{T,j}} \nabla \phi_i \cdot \frac{k}{P} \nabla T  \\
&=& \frac{k}{P} \nabla \phi_i \cdot  \frac{\partial \nabla T}{\partial c_{T,j}} \\
&=& \frac{k}{P} \nabla \phi_i \cdot  \nabla \psi_{T,j}
\end{eqnarray}$$

$$
\therefore \frac{\partial F_{w,i}}{\partial c_{T,j}} = - \phi_i C_p S p^{th} \frac{\partial T}{\partial t} \frac{1}{T^2} \psi_{T,j} + \phi_i C_p p^{th} \boldsymbol{u} \cdot  \left( \frac{1}{T} \nabla \psi_{T,j} - \frac{\nabla T}{T^2} \psi_{T,j} \right) + \frac{k}{P} \nabla \phi_i \cdot  \nabla \psi_{T,j}
$$

$$\begin{eqnarray}
\frac{\partial F_{w,i}}{\partial c_{\frac{\partial T}{\partial t},j}} &=& \frac{\partial }{\partial c_{\frac{\partial T}{\partial t},j}} \phi_i \frac{C_p S p^{th}}{T} \frac{\partial T}{\partial t} \\
&=& \psi_i C_p S p^{th} \frac{\partial }{\partial c_{\frac{\partial T}{\partial t},j}} \left(\frac{1}{T} \frac{\partial T}{\partial t} \right) \\
&=& \psi_i C_p S p^{th} \left(\frac{1}{T} \frac{\partial }{\partial c_{\frac{\partial T}{\partial t},j}}  \frac{\partial T}{\partial t} \right) \\
&=& \psi_i C_p S p^{th} \left(\frac{1}{T} \psi_{j} \right) \\
&=& \psi_i C_p S p^{th} \frac{1}{T} \psi_{j}\\
\end{eqnarray}$$

### V Test Function

$$
\require{cancel}
\begin{eqnarray}
F_\boldsymbol{v} &=& \int_\Omega \boldsymbol{v} \cdot \rho S \frac{\partial \boldsymbol{u}}{\partial t} + \boldsymbol{v} \cdot \rho \boldsymbol{u} \cdot \nabla \boldsymbol{u} + \nabla^S \boldsymbol{v} \cdot \frac{2 \mu}{R} \boldsymbol{\epsilon}'(\boldsymbol{u}) - p \nabla \cdot \boldsymbol{v} + \frac{\rho \hat{\boldsymbol{z}}}{F^2} \cdot \boldsymbol{v} - \int_\Gamma \boldsymbol{t_n} \cdot \boldsymbol{v} = 0
\end{eqnarray}$$

#### Jacobians

$$\begin{eqnarray}
F_{\boldsymbol{v}_i} &=& \int_\Omega \boldsymbol{\phi_i} \cdot \rho S \frac{\partial \boldsymbol{u}}{\partial t} + \boldsymbol{\phi_i} \cdot \rho \boldsymbol{u} \cdot \nabla \boldsymbol{u} + \nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \boldsymbol{\epsilon}'(\boldsymbol{u}) - p \nabla \cdot \boldsymbol{\phi_i} + \frac{\rho \hat{\boldsymbol{z}}}{F^2} \cdot \boldsymbol{\phi_i} - \int_\Gamma \boldsymbol{t_n} \cdot \boldsymbol{\phi_i} = 0
\end{eqnarray}$$

$$\begin{eqnarray}
\frac{\partial F_{\boldsymbol{v}_i}}{\partial c_{p,j}} &=& \int - \frac{\partial}{\partial c_{p,j}} p \nabla \cdot \boldsymbol{\phi_i} \\
&=&\int - \frac{\partial p}{\partial c_{p,j}}  \nabla \cdot \boldsymbol{\phi_i} \\
&=&\int - \psi_{p,j} \nabla \cdot \boldsymbol{\phi_i} \\
\end{eqnarray}$$

$$\begin{eqnarray}
\frac{\partial F_{\boldsymbol{v}_i}}{\partial c_{T,j}} &=& \int \boldsymbol{\phi_i} \cdot \frac{\partial \rho}{\partial c_{T,j}} S \frac{\partial \boldsymbol{u}}{\partial t} + \boldsymbol{\phi_i} \cdot \frac{\partial \rho}{\partial c_{T,j}} \boldsymbol{u} \cdot \nabla \boldsymbol{u} + \frac{\partial \rho}{\partial c_{T,j}} \frac{\hat{\boldsymbol{z}}}{F^2} \cdot \boldsymbol{\phi_i} \\
&=& - \int \boldsymbol{\phi_i} \cdot \frac{p^{th}}{T^2} \frac{\partial T}{\partial c_{T,j}}  S \frac{\partial \boldsymbol{u}}{\partial t} - \boldsymbol{\phi_i} \cdot \frac{p^{th}}{T^2} \frac{\partial T}{\partial c_{T,j}} \boldsymbol{u} \cdot \nabla \boldsymbol{u} - \frac{p^{th}}{T^2} \frac{\partial T}{\partial c_{T,j}} \frac{\hat{\boldsymbol{z}}}{F^2} \cdot \boldsymbol{\phi_i} \\
&=& - \int \boldsymbol{\phi_i} \cdot \frac{p^{th}}{T^2} \psi_{T,j}  S \frac{\partial \boldsymbol{u}}{\partial t} - \boldsymbol{\phi_i} \cdot \frac{p^{th}}{T^2} \psi_{T,j} \boldsymbol{u} \cdot \nabla \boldsymbol{u} - \frac{p^{th}}{T^2} \psi_{T,j} \frac{\hat{\boldsymbol{z}}}{F^2} \cdot \boldsymbol{\phi_i}
\end{eqnarray}$$

$$\begin{eqnarray}
\frac{\partial F_{\boldsymbol{v}_i}}{\partial c_{u_c,j}} &=& \int_\Omega \frac{\partial}{\partial c_{u_c,j}}\boldsymbol{\phi_i} \cdot \rho \boldsymbol{u} \cdot \nabla \boldsymbol{u} + \frac{\partial }{\partial c_{u_c,j}}\nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \boldsymbol{\epsilon}'(\boldsymbol{u})
\end{eqnarray}$$

$$\begin{eqnarray}
\frac{\partial}{\partial c_{u_c,j}} \boldsymbol{\phi_i} \cdot \rho S \frac{\partial \boldsymbol{u}}{\partial t}
&=&\boldsymbol{\phi_i} \cdot \rho S  \frac{\partial}{\partial c_{u_c,j}} \frac{\partial \boldsymbol{u}}{\partial t}
\end{eqnarray}$$

$$\begin{eqnarray}
\frac{\partial}{\partial c_{u_c,j}}\boldsymbol{\phi_i} \cdot \rho \boldsymbol{u} \cdot \nabla \boldsymbol{u}
&=& \boldsymbol{\phi_i} \cdot \left(\rho \frac{\partial\boldsymbol{u}}{\partial c_{u_c,j}} \cdot \nabla \boldsymbol{u} + \rho \boldsymbol{u} \cdot \frac{\partial \nabla \boldsymbol{u}}{\partial c_{u_c,j}}  \right) \\
&=& \boldsymbol{\phi_i} \cdot \left(\rho \psi_j \hat{e}_c \cdot \hat{e}_l \frac{\partial u_k}{\partial x_l}\hat{e}_k + \rho \boldsymbol{u} \cdot \hat{e}_l \frac{\partial \psi_j}{\partial x_l}\hat{e}_c  \right) \\
&=& \boldsymbol{\phi_i} \cdot \left(\rho \psi_j \frac{\partial u_k}{\partial x_c}\hat{e}_k +  \rho u_c \nabla \psi_j  \right) \\
\end{eqnarray}$$

$$\begin{eqnarray}
\frac{\partial}{\partial c_{u_c,j}}\nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \boldsymbol{\epsilon}'(\boldsymbol{u})
&=& \nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \frac{\partial  \boldsymbol{\epsilon}'(\boldsymbol{u})}{\partial c_{u_c,j}} \\
&=& \nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \frac{\partial }{\partial c_{u_c,j}} \left( \frac{1}{2}\left(\nabla \boldsymbol{u} + \nabla\boldsymbol{u}^T \right) - \frac{1}{3}(\nabla \cdot \boldsymbol{u} \boldsymbol{I}) \right) \\
&=& \nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \left( \frac{1}{2}\left(\frac{\partial \nabla \boldsymbol{u}}{\partial c_{u_c,j}} + \frac{\partial \nabla \boldsymbol{u}^T}{\partial c_{u_c,j}} \right) - \frac{1}{3} \frac{\partial \nabla \cdot \boldsymbol{u}}{\partial c_{u_c,j}} \boldsymbol{I} \right) \\
&=& \nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \left( \frac{1}{2}\left( \hat{e}_l \frac{\partial \psi_j}{\partial x_l}\hat{e}_c + \hat{e}_c \frac{\partial \psi_j}{\partial x_l}\hat{e}_l \right) - \frac{1}{3} \frac{\partial \psi_j}{\partial x_c} \boldsymbol{I} \right) \\
\end{eqnarray}$$

$$
\therefore \frac{\partial F_{\boldsymbol{v}_i}}{\partial c_{u_c,j}} = \int \boldsymbol{\phi_i} \cdot \left(\rho \psi_j \frac{\partial u_k}{\partial x_c}\hat{e}_k + \rho u_c \nabla \psi_j  \right) +  \nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \left( \frac{1}{2}\left( \hat{e}_l \frac{\partial \psi_j}{\partial x_l}\hat{e}_c + \hat{e}_c \frac{\partial \psi_j}{\partial x_l}\hat{e}_l \right) - \frac{1}{3} \frac{\partial \psi_j}{\partial x_c} \boldsymbol{I} \right)
$$

$$\begin{eqnarray}
\frac{\partial F_{\boldsymbol{v}_i}}{\partial c_{\frac{\partial u_c}{\partial t},j}} 
&=& \int_\Omega \frac{\partial }{\partial c_{\frac{\partial u_c}{\partial t},j}} \boldsymbol{\phi_i} \cdot \rho S \frac{\partial \boldsymbol{u}}{\partial t} \\
&=& \int_\Omega \boldsymbol{\phi_i} \cdot \rho S \frac{\partial \frac{\partial \boldsymbol{u}}{\partial t}}{\partial c_{\frac{\partial u_c}{\partial t},j}} \\
&=& \int_\Omega \boldsymbol{\phi_i} \cdot \rho S \psi_j \\

\end{eqnarray}$$