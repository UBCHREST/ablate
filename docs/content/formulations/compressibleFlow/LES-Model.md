---
layout: default
title: LES Model
parent: Compressible Flow 
grand_parent: Flow Formulations
nav_order: 5
---
## Large-eddy Simulation of Compressible Flows

### The filtered compressible Navier-Stokes equations can be expressed as (Hirch [1]):

$$\begin{equation}\frac{\partial \bar{\rho}}{\partial t} + \frac{\partial(\bar{\rho} \tilde{u}_{j})}{\partial x_{j}} = 0\end{equation}$$

$$\begin{equation}\frac{\partial (\bar{\rho} \tilde{u}_{i})}{\partial t} + \frac{\partial(\bar{\rho} \tilde{u}_{i}\tilde{u}_{j})}{\partial x_{j}} = -\frac{\partial \bar{p}}{\partial x_{i}} + \frac{\partial \mu \tilde{S}_{ij}}{\partial x_{j}} -
 \frac{\partial \tau_{ij}}{\partial x_{j}}\end{equation}$$

$$\begin{equation}\frac{\partial (\bar{\rho} \tilde{e})}{\partial t} + (\bar{\rho} \tilde{e}+\bar{p})\frac{\partial \tilde{u}_{j}}{\partial x_{j}} = \frac{\partial(\mu \tilde{S}_{ij}-\tau_{ij})\tilde{u}_{i}}{\partial x_{j}} + \frac{\partial }{\partial x_{j}}\left(\lambda\frac{\partial \tilde{T}}{\partial x_{j}}\right) - \frac{\partial{ q_{j}}}{\partial x_{j}}+ \bar{\dot{\omega}}_T\end{equation}$$

$$\begin{equation}\frac{\partial (\bar{\rho} \tilde{Y_i})}{\partial t} + (\bar{\rho} \tilde{Y_i})\frac{\partial \tilde{u}_{j}}{\partial x_{j}} = \frac{\partial }{\partial x_{j}}\left(\alpha \frac{\partial (\bar{\rho}\tilde{Y_i})}{\partial x_{j}}\right) - \frac{\partial{ J_{j}}}{\partial x_{j}}+ \bar{\dot{\omega}}_m    (i=1, ..., N-1)\end{equation}$$


Where $$\tau_{ij}=\bar{\rho}\widetilde{u_{i}u_{j}}-\bar{\rho}\tilde{u_i}\tilde{u_j}$$ is the subgrid-scale (SGS) stress tensor, $$q_{j}=\bar{\rho} C_p \widetilde{u_jT}-\bar{\rho} C_p \tilde{u_j}\tilde{T}$$, the SGS heat flux, representing the effect of the small scales of turbulence on the large ones, and $$J_j$$ is the species transport due to the SGS velocity fluctuations. The set of equations is closed by setting $$\tau_{ij} = -2 \mu_t \widetilde{S_{ij}}$$, $$q_j =-C_p \frac{\mu_t}{Pr_t} \frac{\partial \tilde{ T}}{\partial x_j}$$, and $$J_j =- \frac{\mu_t}{Sc_t} \frac{\partial \tilde{ Y_i}}{\partial x_j}$$, where $$\widetilde{S_{ij}}= \frac{1}{2} \left(\frac{\partial \widetilde{u_i}}{\partial x_j}\frac{\partial \widetilde{u_j}}{\partial x_i} \right) -\frac{1}{3} \frac{\partial \widetilde{u_k}}{\partial x_k} \delta_{ij}$$ and $$\mu_t$$ is the turbulent viscosity, and by the modified averaged Equation of State, $$\bar{P}= \bar{\rho} R \tilde{T}$$. $$Pr_t$$ and $$Sc_t$$ are assumed to be $$1$$ based on Reynolds analogy.
                
#### k-eq LES model
              
Here k-equation model is used to model the eddy viscosity. The turbulent viscosity is given by (Chai et al. [2]):
$$\mu_t = \bar{\rho} C_k \Delta k^{0.5}$$

To account for the effects of convection, diffusion, production and destruction on the SGS tensor, the transport equation is solved to determine turbulence kinetic 
energy $$k$$:


$$\begin{equation}\frac{\partial (\bar{\rho }k)}{\partial t} + \frac{\partial (\bar{\rho } \tilde{u}_j k)}{\partial x_{j}} -\frac{\partial}{\partial x_{j}} \left[\\
\left(\nu + \nu_t\right) \frac{\partial(\bar{\rho } k)}{\partial x_{j}} \right] =-\bar{\rho } \tau_{ij} .\widetilde{ S_{ij}} \; -\; C_{\epsilon} \frac{\bar{\rho } k^{3/2}}{\Delta},\end{equation}$$

where $$$\Delta$$ is the grid size and the default model coefficients are $$ C_k=0.094, C_\epsilon =1.048$$.

## References
- [1] Hirsch, C., Numerical computation of internal and external flows. Vol. 1 - Computational methods for inviscid and viscous
flows, 1990.
- [2] Chai, X., and Mahesh, K., “Dynamic k-Equation Model for Lage Eddy Simulation of Compressible Flows,” Journal of Fluid
Mechanics, Vol. 699, 2012, pp. 385–413
