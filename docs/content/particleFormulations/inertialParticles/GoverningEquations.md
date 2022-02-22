---
layout: default
title: Governing Equations for Inertial Particles
parent: Particle Formulations
nav_order: 1
---

### Inertial Particles
A one-way coupled non-reacting iso-thermal
Lagrangian tracking model is currently implemented. Inertial
droplets/particles are dispersed by the gaseous
phase, while their interaction/reaction with the background flow
is neglected. Buoyancy and drag, as the carrier phase forces
acting on the particles, are currently implemented. The position,
$$\boldsymbol{x}_p$$, and velocity, $$\boldsymbol{u}_p$$, of
particles with diameter $$d_p$$ and density of
$$\rho_p$$ are obtained as,

$$\begin{eqnarray}
\frac{d\boldsymbol{x}_p}{dt} = \boldsymbol{u}_p \\
\frac{d\boldsymbol{u}_p}{dt} =  \frac{f}{\tau_p}\left(\boldsymbol{u}_f - \boldsymbol{u}_p\right) + \boldsymbol{g}\left(1-\frac{\rho_f}{\rho_p}\right)
\end{eqnarray}$$

where, $$f=1+0.15Re_p^{0.687}$$ is the Schiller-Naumann
correction factor to Stokes drag expression (see the reference)
to account for the effect of finite particle Reynolds number
(denoted by $$Re_p=\rho_f|\boldsymbol{u}_f-\boldsymbol{u}_p|d_p/\mu$$).
In addition, $$\tau_p=\rho_pd^2_p/18\mu$$ is the particle
relaxation time obtained based on Stokes drag expression.
$$\mu$$ and $$\rho_f$$ are the carrier phase dynamic
viscosity and density, respectively.  
The upstream fluid velocity for each particle, $$\boldsymbol{u}_f$$,
is obtained by interpolating the surrounding fluid velocity to the
particle location.

In the limit of small $$Re_p$$ (thus $$f{=}1$$), analytical
solution for the particle velocity is obtained as,

$$\begin{eqnarray}
\boldsymbol{u}_p(t) = \boldsymbol{u}_{st}\left(1-\exp\left(\frac{-t}{\tau_p}\right)\right)
\end{eqnarray}$$

where,

$$\boldsymbol{u}_{st}=\tau_p\boldsymbol{g}(1-\frac{\rho_f}{\rho_p})$$ is the particle
terminal (settling) velocity.

## References
- Clift, R., Grace, J.R., & Weber, M.E. (2005). Bubbles, drops, and particles. Dover Publications.
