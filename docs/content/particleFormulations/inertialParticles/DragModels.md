---
layout: default
title: Drag Models
parent: Particle Formulations
nav_order: 2
---

### Linear Drag (Stokes' Law)

A particle drag model suitable for low Reynolds number flows is Stokes' Law. This drag model is linear in the relative velocity $$\vec{V}$$.

$$\vec{F}_d = -3 \pi d^2 \mu_f \vec{V}$$

### Quadratic Drag (High Reynolds Number Solid Spheres)

At higher Reynolds numbers a drag model that is quadratic in the relative velocity $$\vec{V}$$ is:

$$\vec{F}_d = -C_d \frac{\pi}{8} d^2 \cdot \frac{1}{2} \rho_f |\vec{V}| \vec{V}$$

The value $$C_d = 0.42$$ is the high Reynolds number limit of equation 8 of Loth.

This equation applies for solid spheres at high Reynolds numbers. Additional physics relevant to droplets are missing from this model.

## References

- E. Loth, “Quasi-steady shape and drag of deformable bubbles and drops,” International Journal of Multiphase Flow, vol. 34, no. 6, pp. 523–546, Jun. 2008, doi: 10.1016/j.ijmultiphaseflow.2007.08.010.
