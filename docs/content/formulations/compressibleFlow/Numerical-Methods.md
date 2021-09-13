---
layout: default
title: Numerical Methods
parent: Compressible Flow Formulation
grand_parent: Flow Formulations
nav_order: 2
---

## Numerical Methods

### AUSM+up for All Speeds, Single Phase

$$M_{L/R} =  \frac{u_{L/R}}{a_{1/2}}$$

where $$a_{1/2}$$ is an average of $$a_L$$ and $$a_R$$

$$\begin{eqnarray}\bar{M}^2 = \frac{(u^2_L+u^2_R)}{2 a^2_{1/2}} \\
M_o^2=\min(1,\max(\bar{M}^2,M_\infty^2))\\
f_a = M_o(2-M_o)\\
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

The below calculations are for gas-gas or liquid-liquid interfaces.  Gas-liquid or liquid-gas interfaces must use a Riemann solver.

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

## References
- Liou, M. S. (2006). "A sequel to AUSM, Part II: AUSM+-up for all speeds." Journal of Computational Physics, 214, 137-170.
- Chang, C. H. and Liou, M. S. (2007), "A robust and accurate approach to computing compressible multiphase flow: Stratified flow model and AUSM+up scheme." Journal of Computational Physics, 225, 840-873.
