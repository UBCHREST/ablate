---
layout: default
title: Governing Equations
parent: Compressible Flow Formulation
grand_parent: Flow Formulations
nav_order: 1
---

### Governing Equations
The compressible flow formulation is solved using a finite volume formulation where the conserved values $$(\rho, \rho\boldsymbol{u}, \rho e_t,)$$ are computed for each volume.  Writen in terms of volumes and fluxes boundaries:

$$\begin{eqnarray}
\frac{\partial}{\partial t} \int_\Omega \vector{Q_t} d \Omega + \int_{\partial \Omega} \left (\vector{F_c} - \vector{F_v} \right) dS = \int_\Omega \vector{Q_s} d \Omega
\end{eqnarray}$$

Where the conservative variables in 3D are:

$$\begin{eqnarray}
\vector{Q_t} = \begin{bmatrix} \rho \\ \rho u \\ \rho v \\ \rho w \\ \rho e_t \end{bmatrix}
\end{eqnarray}$$

Within ABLATE the energy conservation equation is often stored after continuity to simplify indexing but presented in the traditional order in the documentation. The convective fluxes are

$$\begin{eqnarray}
\vector{F_c} = \begin{bmatrix} \rho u_n \\ \rho u u_n + n_x p \\ \rho v u_n + n_y p \\ rho w u_n + n_z p \\ \rho H u_n \end{bmatrix}
\end{eqnarray}$$

where $$\vector{n}$$ is the normal, $$u_n$$ is the normal velocity, and $$H = e + p/\rho + \|\vector{u}\|^2/2$$.  These terms define the Euler equations and can be recovered in ABLATE by setting diffusion coefficients to zero.  The viscous fluxes are defined as:

$$\begin{eqnarray}
\vector{F_v} = \begin{bmatrix}
    0 \\
    n_x \tau_{xx}  + n_y \tau_{xy} + n_z \tau_{xz} \\
    n_x \tau_{yx}  + n_y \tau_{yy} + n_z \tau_{yz} \\
    n_x \tau_{zx}  + n_y \tau_{zy} + n_z \tau_{zz} \\
    nx \Theta_x    + n_y \Theta_y  + n_z \Theta_z 
\end{bmatrix} 
\end{eqnarray}$$

where

$$\begin{eqnarray}
\Theta_x = u \tau_{xx} + v \tau_{xy} + w \tau_{xz} + k \frac{\partial T}{\partial x} \\
\Theta_y = u \tau_{yx} + v \tau_{yy} + w \tau_{yz} + k \frac{\partial T}{\partial y} \\
\Theta_z = u \tau_{zx} + v \tau_{zy} + w \tau_{zz} + k \frac{\partial T}{\partial z}
\end{eqnarray}$$

The vector of source terms is

$$\begin{eqnarray}
\vector{Q_s} = \begin{bmatrix}
    0 \\
    \rho f_x \\
    \rho f_y \\
    \rho f_z \\
    \rho \vector{f} \cdot \vector{u} + \dot{q_h}
    \end{bmatrix}
\end{eqnarray}$$

Additional models are need to complete the system including an Equation of State (EOS) viscosity model.

## References
 - Blazek, J. (2001). Computational fluid dynamics: Principles and applications.
 - Roy, Chris, Curt Ober, and Tom Smith. "Verification of a compressible CFD code using the method of manufactured solutions." 32nd AIAA Fluid Dynamics Conference and Exhibit. 2002.
