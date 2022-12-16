---
layout: default
title: Numerical Methods
parent: Compressible Flow
grand_parent: Formulations
nav_order: 2
---

## Numerical Methods

### AUSM+up for All Speeds, Single Phase

This algorithm is used to calculate the convective fluxes $$F_c$$ in the Governing Equations.

$$M_{L/R} =  \frac{u_{L/R}}{a_{1/2}}$$

where $$a_{1/2}$$ is an average of $$a_L$$ and $$a_R$$

$$\bar{M}^2 = \frac{(u^2_L+u^2_R)}{2 a^2_{1/2}} \\
M_o^2=\min(1,\max(\bar{M}^2,M_\infty^2))\\ $$

where $$M_\infty \neq 0$$ and is a specified parameter describing the representative Mach number for the given problem.

$$ \begin{eqnarray} f_a = M_o(2-M_o)\\
\mathcal{M}_{(1)}^\pm (M) = \frac{1}{2} (M \pm |M|)\\
\mathcal{M}_{(2)}^\pm (M) = \pm \frac{1}{4}(M \pm 1)^2\\
\mathcal{M}_{(4)}^\pm (M)= \begin{cases}
\mathcal{M}_{(1)}^\pm & \text{if } |M| \geq 1 \\
\mathcal{M}_{(2)}^\pm (1 \mp 16\beta \mathcal{M}_{(2)}^\mp ) & \text{otherwise}
\end{cases}\\
M_{1/2}=\mathcal{M}_{(4)}^+(M_L)+\mathcal{M}_{(4)}^-(M_R)+M_p\\
M_p = -\frac{K_p}{f_a} \max(1-\sigma \bar{M}^2,0) \frac{p_R-p_L}{\rho_{1/2}a_{1/2}^2}
\end{eqnarray}$$

where $$\rho_{1/2}$$ is also an average of $$\rho_L$$ and $$\rho_R$$.

$$\begin{eqnarray}\dot{m} = a_{1/2}M_{1/2} \begin{cases}
\rho_L & \text{if } M_{1/2}>0\\
\rho_R & \text{otherwise}
\end{cases} \\
\mathcal{P}_{(5)}^\pm = \begin{cases}
\frac{1}{M} \mathcal{M}^\pm_{(1)} & \text{if } |M| \geq 1 \\
\mathcal{M}^\pm_{(2)}[(\pm 2 -M) \mp 16 \alpha_{coeff}M \mathcal{M}^\mp_{(2)}] & \text{otherwise}
\end{cases}\\
p_{1/2}= \mathcal{P}_{(5)}^+(M_L)p_L+\mathcal{P}_{(5)}^-(M_R)p_R+P_u\\
P_u= -K_u \mathcal{P}_{(5)}^+ \mathcal{P}_{(5)}^- (\rho_L+\rho_R)(f_a a_{1/2})(u_R-u_L)\\
\alpha_{coeff} = \frac{3}{16} (-4 + 5 f_a^2)
\end{eqnarray}$$

| Coefficient| Value  |
| ---------- | ------ | 
| $$K_p$$    | 0.25   | 
| $$K_u$$    | 0.75   | 
| $$\sigma$$ | 1.0    | 
| $$\beta$$  | 1/8    | 

### AUSM+up for two phases

The below calculations are for gas-gas or liquid-liquid interfaces.  Gas-liquid or liquid-gas interfaces must use a Riemann solver.  Any definitions not listed in this section are the same as the single phase definitions.

| Coefficient| Value  |
| ---------- | ------ | 
| $$K_p$$    | 1.0    | 
| $$K_u$$    | 1.0    | 
| $$\sigma$$ | 2.0    | 

$$\begin{eqnarray}\frac{1}{a^2_{mix}} \left(\frac{\alpha_l}{\rho_l} + \frac{\alpha_g}{\rho_g}  \right) = \frac{\alpha_l}{\rho_l a_l^2} + \frac{\alpha_g}{\rho_g a_g^2}\\
M_{L/R} =  \frac{u_{L/R}}{a_{mix}}\\
\mathcal{M}_{(4)}^\pm (M)= \begin{cases}
\mathcal{M}_{(1)}^\pm & \text{if } |M| \geq 1 \\
\mathcal{M}_{(2)}^\pm (1 \mp 2 \mathcal{M}_{(2)}^\mp ) & \text{otherwise}
\end{cases}\\
\mathcal{P}_{(5)}^\pm = \begin{cases}
\frac{1}{M} \mathcal{M}^\pm_{(1)} & \text{if } |M| \geq 1 \\
\mathcal{M}^\pm_{(2)}[(\pm 2 -M) \mp 3 M \mathcal{M}^\mp_{(2)}] & \text{otherwise}
\end{cases}\\
M_{1/2}=\mathcal{M}_{(4)}^+(M_L)+\mathcal{M}_{(4)}^-(M_R)\\
\dot{m} = a_{mix} M_{1/2} \rho_{L/R} + D_p\\
D_p = K_p \frac{\Delta M \max(1-\bar{M}^2,0)(p_L-p_R)}{a_{1/2}} \\
\Delta M = \mathcal{M}_{(4)}^+(M_L) - \mathcal{M}_{(1)}^+(M_L) -\mathcal{M}_{(4)}^-(M_R)+\mathcal{M}_{(1)}^+(M_R)\\
  P_u= K_u \mathcal{P}_{(5)}^+(\bar{M}) \mathcal{P}_{(5)}^-(\bar{M}) \rho_{1/2} a_{1/2}(u_L-u_R)\\
  \end{eqnarray}$$

where $$\bar{M}$$ is the average between $$M_L$$ and $$M_R$$.

### Riemann Flux

Riemann problem is a classic 1D initial value problem. The exact solution can be used as a means to calculate the fluxes in each cell. Based on the adjacent cell's state, $$ W_{L/R} = \{\rho_{L/R},\ u_{L/R},\ p_{L/R}\}$$, an iterative scheme can be used to achieve exact solution.
The initial condition is set as

$$ \begin{equation}
W(x,0) = \begin{cases}
W_L, & \text{ if  } 1 < x \\
W_R, & \text{ if  } x \geq 1
\end{cases}
\end{equation}$$

The pressure $$p^*$$ can be found by solving the root for the following equation

$$
f(p, W_L, W_R) \equiv f_L(p, W_L) + f_R(p, W_R) + (u_R-u_L) =0
$$

where $$f_L$$ is given as
$$
f_L(p, W_L) = \begin{cases}
(p-p_L)\left[ \frac{A_L}{p+B_L}\right]^{1/2}, & \text{ if } p>p_L (Shock) \\
\frac{2a_L}{\gamma-1}\left[\left(\frac{p}{p_L}^\frac{\gamma-1}{2\gamma}\right) -1\right], &\text{ if }p\leq p_L (Rarefaction)
\end{cases}
$$

$$f_R$$ is given as

$$
f_R(p, W_R) = \begin{cases}
\frac{2a_R}{\gamma-1}\left[\left(\frac{p}{p_R}^\frac{\gamma-1}{2\gamma}\right) -1\right], &\text{ if }p\leq p_R (Rarefaction) \\
(p-p_R)\left[ \frac{A_R}{p+B_R}\right]^{1/2}, &\text{ if }p> p_R (Shock)
\end{cases}
$$

$$A_L=\frac{2}{(\gamma+1)\rho_L},\ B_L=\frac{\gamma-1}{\gamma+1}p_L\\
A_R=\frac{2}{(\gamma+1)\rho_R},\ B_R=\frac{\gamma-1}{\gamma+1}p_R  \\
u_*=\frac{1}{2}(u_L+u_R)+\frac{1}{2}[f_R(p_*)-f_L(p_*)]
$$

After solving for the stared region properties, we can then back out the flow properties via the following logics.

|     if     |  $$u_{*} \geq 0$$  |               |                              |                              |                |
|:----------:|:---------------:|:-------------:|:----------------------------:|:----------------------------:|:--------------:|
|            |    $$p_* > p_L$$ (Left Shock)   |               | $$p_*\leq p_L$$ (Left Expansion)        |                              |                |
|            |      $$S_L \geq 0 $$   |  $$S_L < 0$$  | $$S_{TL} \geq 0 \& S_{HL} \geq 0$$ | $$S_{TL} \geq 0 \& S_{HL} < 0$$ | $$S_{TL} < 0$$ |
|   $$u(x=0) $$  |      $$ u_L $$      |   $$u_{*L}$$  |            $$u_L$$           | $$\frac{2}{\gamma+1}\left[ a_L+\frac{\gamma-1}{2}u_L\right] $$ |   $$u_{*L}$$   |
|  $$\rho(x=0)$$ |    $$\rho_L$$   | $$\rho_{*L}$$ |          $$\rho_L$$          | $$\rho_L\left[\frac{2}{\gamma+1}+\frac{\gamma-1}{(\gamma+1)a_L}u_L \right]^{\frac{2}{\gamma-1}}$$  |  $$\rho_{*L}$$ |
| $$p(x=0)$$ |     $$p_L$$     |   $$p_{*L}$$  |            $$p_L$$           |$$p_L\left[ \frac{2}{\gamma+1}+\frac{\gamma-1}{(\gamma+1)a_L}u_L\right]^{\frac{2\gamma}{\gamma-1}}$$|   $$p_{*L}$$   |
|     if     |   $$u_{*} < 0 $$  |             |                              |                              |                |
|            |  $$p_* < p_R$$ (Right Shock)  |             |  $$p_*\geq p_R$$ (Right Expansion)                       |                              |                |
|            |  $$S_R \geq 0$$  | $$S_R < 0$$ | $$S_{HR} \geq 0 \& S_{TR} \geq 0$$ | $$S_{HR} \geq 0 \& S_{TR} < 0$$ | $$S_{HR} < 0$$ |
| $$  u(x=0)  $$ |   $$u_{*R}$$  |  $$ u_R $$  |          $$u_{*R}$$          | $$\frac{2}{\gamma+1}\left[-a_R+\frac{\gamma-1}{2}u_R\right] $$ |     $$u_R$$    |
|  $$\rho(x=0)$$ | $$\rho_{*R}$$ |  $$\rho_R$$ |         $$\rho_{*R}$$        |$$\rho_R\left[\frac{2}{\gamma+1}-\frac{\gamma-1}{(\gamma+1)a_R}u_R \right]^{\frac{2}{\gamma-1}}$$|   $$\rho_R$$   |
| $$p(x=0)$$ |   $$p_{*R}$$  |   $$p_R$$   |          $$p_{*R}$$          |$$p_R\left[ \frac{2}{\gamma+1}-\frac{\gamma-1}{(\gamma+1)a_R}u_R\right]^{\frac{2\gamma}{\gamma-1}}$$|     $$p_R$$    |

### Pressure Gradient Scaling (PGS)
Pressure Gradient Scaling (PGS) is a technique to address the very small-time steps sometimes associated with compressible low-speed flows.  In this approach the acoustic wave speeds are artificially reduce thereby increasing the allowable step size.  Implementation details and the affects on the LODI and AUSM+ flux vector splits methods are available in DesJardin et al. [4].

## References
- [1] Liou, M. S. (2006). "A sequel to AUSM, Part II: AUSM+-up for all speeds." Journal of Computational Physics, 214, 137-170.
- [2] Chang, C. H. and Liou, M. S. (2007), "A robust and accurate approach to computing compressible multiphase flow: Stratified flow model and AUSM+up scheme." Journal of Computational Physics, 225, 840-873.
- [3] Toro, E. F. (2009). "Riemann Solvers and Numerical Methods for Fluid Dynamics - A PracticalIntroduction. International series of monographs on physics." Springer. ISBN: 978-3-540-49834-6
- [4] DesJardin, Paul E., Timothy J. O’Hern, and Sheldon R. Tieszen. "Large eddy simulation and experimental measurements of the near-field of a large turbulent helium plume." Physics of fluids 16.6 (2004): 1866-1883.
