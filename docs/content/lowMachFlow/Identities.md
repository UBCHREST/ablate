---
layout: default
title: Reference Identities
parent: Low Mach Flow Formulation
nav_order: 4
---
Given that our field variable $$u$$ or $$\boldsymbol{u}$$ is represented as $$u = \sum_k c_k \psi_k $$ or $$ \boldsymbol{u} = \sum_k \boldsymbol{c_k} \psi_k $$ we can express the change in the field variable in terms of nodal value $${c_j}$$ as

$$
\frac{\partial u}{\partial c_j} = \frac{\partial}{\partial c_j} \sum_k c_k \psi_k = \sum_k \frac{\partial c_k}{\partial c_j} \psi_k
$$

Knowing that $$\frac{\partial c_k}{\partial c_j} = 1$$ when $$k = j$$ and $$\frac{\partial c_k}{\partial c_j} = 0 $$ when $$k \neq j$$ simplifies to

$$
\frac{\partial u}{\partial c_j} = \psi_j
$$

Similar the following can be shown:

$$
\frac{\partial}{\partial c_j}\nabla u = \nabla \psi_j \\
$$

## Vector Field
For vector fields $$ \boldsymbol{u} = u_c \hat{e}_c $$

$$
\frac{\partial \boldsymbol{u}}{\partial c_{c,j}} = \psi_j \hat{e}_c \\
$$

For illustrative purposes, this can be show in cartesian coordinates for a single component of $$c_j$$. 

$$
\boldsymbol{u} = u_1 \hat{e}_1 + u_2 \hat{e}_2 + u_3 \hat{e}_3 \\
u_1 = \sum_k c_{1,k} \psi_k , u_2 = \sum_k c_{2,k} \psi_k , u_3 = \sum_k c_{3,k} \psi_k \\ 
\therefore \boldsymbol{u} =  \sum_k c_{1,k} \psi_k \hat{e}_1 + \sum_k c_{2,k} \psi_k  \hat{e}_2 + \sum_k c_{3,k} \psi_k \hat{e}_3 \\
\frac{\partial \boldsymbol{u}}{\partial c_{1,j}} = \sum_k \frac{c_{1,k}}{c_{1,j}} \psi_k \hat{e}_1 + \sum_k \frac{c_{2,k}}{c_{1,j}} \psi_k  \hat{e}_2 + \sum_k \frac{c_{3,k}}{c_{1,j}} \psi_k \hat{e}_3
$$

when $$j = k$$, $$\frac{c_{1,j}}{c_{1,j}} = 1$$, $$\frac{c_{2,j}}{c_{1,j}} = 0$$, $$\frac{c_{3,j}}{c_{1,j}} = 0$$, resulting in 

$$
\frac{\partial \boldsymbol{u}}{\partial c_{1,j}} = \psi_j \hat{e}_1 
$$

Similarly for a gradient of a vector field

$$
\frac{\partial \nabla\boldsymbol{u}}{\partial c_{c,j}} = \hat{e}_l \frac{\partial \psi_j}{\partial x_l} \hat{e}_c
$$

So for $$c_{1,j}$$ this would reduce to

$$
\frac{\partial \nabla\boldsymbol{u}}{\partial c_{1,j}} = \hat{e}_1 \frac{\partial \psi_j}{\partial x_1} \hat{e}_1 + \hat{e}_2 \frac{\partial \psi_j}{\partial x_2} \hat{e}_1 + \hat{e}_3 \frac{\partial \psi_j}{\partial x_3} \hat{e}_1
$$

For the divergence of a vector field

$$ 
\nabla \cdot \boldsymbol{u} = \frac{\partial u_{cc}}{\partial x_{cc}}  = \sum_k c_{cc,k} \frac{\partial \psi_k}{\partial x_{cc}} \\
\frac{\partial \nabla \cdot \boldsymbol{u}}{\nabla c_{c,j}} = \frac{\partial \psi_j}{\partial x_c}
$$

