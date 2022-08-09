---
layout: default
title: Residual Terms
parent: Incompressible Flow
grand_parent: Formulations
nav_order: 3
---
Provides the term to evaluate and multiply by the test function/test gradient function at each quadrature location.  The density is assumed to be unity for the implementation (i.e. equal to the reference density).  The Residual and Jacobians are in terms of

### Q Test Function

$$
F_q = \int_\Omega q \left( \nabla \cdot \boldsymbol{u} \right)  d\Omega = 0
$$

#### Jacobians

$$
F_{q,i} = \int \phi_i \nabla \cdot \boldsymbol{u}  \\
$$

$$\begin{eqnarray}
\frac{F_{q,i}}{\partial c_{u_c,j}}
&=& \int \phi_i \nabla \cdot \boldsymbol{u}  \\
&=& \int \phi_i \frac{\partial \psi_{u_c,j}}{\partial x_c}
\end{eqnarray}$$

### W Test Function

$$
\require{cancel}
\begin{eqnarray}
F_w &=& \int_\Omega w \rho C_p S \frac{\partial T}{\partial t} + w \rho C_p \boldsymbol{u} \cdot \nabla T + \nabla w \cdot \frac{k}{P} \nabla T - \cancelto{0}{w \Gamma \beta S T \frac{\partial p^{th}}{\partial t}} - wHSQ - \int_\Gamma w q_n = 0 \\
\end{eqnarray}$$

#### Jacobians

$$\begin{eqnarray}
\require{enclose}
F_{w,i} &=& \int_\Omega \phi_i C_p S \rho \frac{\partial T}{\partial t} + \phi_i \rho C_p \boldsymbol{u} \cdot \nabla T + \nabla \phi_i \cdot \frac{k}{P} \nabla T - wHSQ - \int_\Gamma \phi_i q_n \\
\end{eqnarray}$$

$$\begin{eqnarray}
\frac{\partial F_{w,i}}{\partial c_{u_c,j}} &=& \frac{\partial}{\partial c_{u_c,j}} \phi_i \rho C_p \boldsymbol{u} \cdot \nabla T \\
&=&  \phi_i  \rho C_p \frac{\partial \boldsymbol{u}}{\partial c_{u_c,j}} \cdot \nabla T \\
&=&  \phi_i  \rho C_p \psi_{u_c,j} \hat{e}_c \cdot \nabla T \\
&=&  \phi_i  \rho C_p \psi_{u_c,j} \frac{\partial T}{\partial x_c}
\end{eqnarray}$$

$$\begin{eqnarray}
\frac{\partial F_{w,i}}{\partial c_{T,j}} &=&  \frac{\partial }{\partial c_{T,j}}  \phi_i \rho C_p \boldsymbol{u} \cdot \nabla T  + \frac{\partial }{\partial c_{T,j}}  \nabla \phi_i \cdot \frac{k}{P} \nabla T \\
&=&    \phi_i \rho C_p \boldsymbol{u} \cdot  \frac{\partial \nabla T}{\partial c_{T,j}}   + \frac{\partial }{\partial c_{T,j}}  \nabla \phi_i \cdot \frac{k}{P} \frac{\partial \nabla T}{\partial c_{T,j}} \\
&=&    \phi_i \rho C_p \boldsymbol{u} \cdot  \nabla \psi_{T,j}   + \frac{k}{P} \nabla  \phi_i \cdot   \nabla \psi_{T,j} \\

\end{eqnarray}$$

$$\begin{eqnarray}
\frac{\partial F_{w,i}}{\partial c_{\frac{\partial T}{\partial t},j}} 
&=& \frac{\partial }{\partial c_{\frac{\partial T}{\partial t},j}} \phi_i \rho C_p S \frac{\partial T}{\partial t} \\
&=& \psi_i \rho C_p S \frac{\partial }{\partial c_{\frac{\partial T}{\partial t},j}}\frac{\partial T}{\partial t} \\
&=& \psi_i \rho C_p S \psi_j \\
\end{eqnarray}$$

### V Test Function

$$
\require{cancel}
\begin{eqnarray}
F_\boldsymbol{v} &=& \int_\Omega \boldsymbol{v} \cdot \rho S \frac{\partial \boldsymbol{u}}{\partial t} + \boldsymbol{v} \cdot \rho \boldsymbol{u} \cdot \nabla \boldsymbol{u} + \nabla^S \boldsymbol{v} \cdot \frac{2 \mu}{R} \boldsymbol{\epsilon}(\boldsymbol{u}) - p \nabla \cdot \boldsymbol{v} - \int_\Gamma \boldsymbol{t_n} \cdot \boldsymbol{v} = 0
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
\frac{\partial}{\partial c_{u_c,j}}\nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \boldsymbol{\epsilon}(\boldsymbol{u})
&=& \nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \frac{\partial  \boldsymbol{\epsilon}(\boldsymbol{u})}{\partial c_{u_c,j}} \\
&=& \nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \frac{\partial }{\partial c_{u_c,j}} \left( \frac{1}{2}\left(\nabla \boldsymbol{u} + \nabla\boldsymbol{u}^T \right) \right) \\
&=& \nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \left( \frac{1}{2}\left(\frac{\partial \nabla \boldsymbol{u}}{\partial c_{u_c,j}} + \frac{\partial \nabla \boldsymbol{u}^T}{\partial c_{u_c,j}} \right) \right) \\
&=& \nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \left( \frac{1}{2}\left( \hat{e}_l \frac{\partial \psi_j}{\partial x_l}\hat{e}_c + \hat{e}_c \frac{\partial \psi_j}{\partial x_l}\hat{e}_l \right) \right) \\
\end{eqnarray}$$

$$
\therefore \frac{\partial F_{\boldsymbol{v}_i}}{\partial c_{u_c,j}} = \int_\Omega \boldsymbol{\phi_i} \cdot \left(\rho \psi_j \frac{\partial u_k}{\partial x_c}\hat{e}_k +  \rho u_c \nabla \psi_j  \right) + \nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \left( \frac{1}{2}\left( \hat{e}_l \frac{\partial \psi_j}{\partial x_l}\hat{e}_c + \hat{e}_c \frac{\partial \psi_j}{\partial x_l}\hat{e}_l \right) \right)
$$

$$\begin{eqnarray}
\frac{\partial F_{\boldsymbol{v}_i}}{\partial c_{\frac{\partial u_c}{\partial t},j}} 
&=& \int_\Omega \frac{\partial }{\partial c_{\frac{\partial u_c}{\partial t},j}} \boldsymbol{\phi_i} \cdot \rho S \frac{\partial \boldsymbol{u}}{\partial t} \\
&=& \int_\Omega \boldsymbol{\phi_i} \cdot \rho S \frac{\partial \frac{\partial \boldsymbol{u}}{\partial t}}{\partial c_{\frac{\partial u_c}{\partial t},j}} \\
&=& \int_\Omega \boldsymbol{\phi_i} \cdot \rho S \psi_j \\

\end{eqnarray}$$