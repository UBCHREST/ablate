---
layout: default
title: Radiation Formulation
nav_order: 8
has_children: false
---

## Mathematical Formulation

Radiation heat transfer makes up a source term in the energy equation, which has the radiative gains and losses as its
components. While the losses are easy to calculate, (using only the properties of the cell in question) the gains
involve the entire domain. This means that in order to solve for the radiative gains of a single cell, rays are traced
from the boundaries of the domain into the cell center. The summation of these rays about a solid sphere results in the
radiative gain that the cell experiences from its environment. The energy equation is shown below.

$$ -\nabla \cdot (\vec{q_{rad}}) = -\kappa (4 \sigma T^4 - G_{irr}) $$

The calculated radiation term becomes a source term in the energy equation. The energy equation is as follows.

$$ \rho C_p \frac{D T}{D t} + \nabla \cdot (\rho e \vec{v}) = - \nabla \cdot (\vec{q_{rad}} + \vec{q_{cond}}) + Q_
{source} $$

\begin{itemize}
\item $\rho$ is density
\item $C_p$ is constant pressure specific heat
\item $T$ is temperature
\item $e$ is internal energy
\item $\vec{v}$ is velocity
\item $\vec{q_{rad}}$ is radiative heat
\item $\vec{q_{cond}}$ is conductive heat
\item $Q_{source}$ represents an energy source term
\end{itemize}

$G_{irr}$ represents the irradiation on the cell by its environment. Ray tracing solves the problem of
calculating irradiation by discretizing the solid sphere, or breaking it into pieces, and casting discrete rays into the
domain. A finite number of rays can be used to approximate the total irradiation from every direction away from the cell
by summing their effects. For our purposes, $\theta$ is extended to $2 \pi$ in order to include the entire sphere and
the irradiation of the cell as a volume. The figure describing the solid sphere formulation is shown below.

Solid angle figure here

$$ G_{irr} = \int_{0}^{2\pi} \int_{0}^{\pi} I_{pt}(\theta,\phi)\ sin\theta\ d\theta\ d\phi $$

The summing of ray intensities around the whole solid sphere is as follows. The presence of the sine term is
an artifact of the polar coordinates. For example, at $\theta = 0$, all rays at each $\phi$ occupy the same point.
Therefore, they are all weighted at 0. The effects of this integral must be broken into a discrete formulation in order
to be of use computationally.

$$ G_{irr}\ = \sum_{\theta=0}^{n_{\theta}}\sum_{\phi=0}^{n_{\phi}} I_{pt}(\theta,\phi)\ sin\theta\ \Delta \theta\ \Delta
\phi $$

The third equation shows how the irradiation is calculated for a cell. Each ray is weighted by its partial
area. The “sine of theta” in this equation represents a conversion from polar coordinates. At each theta there is an
equal number of rays around the circumference of the band. However, the lower bands have a lower density of rays
compared to the high bands (small theta). Therefore, when theta is 0 and many rays occupy the same point, they are
weighted as 0 and so on.

With this solid angle formulation, many rays can be generated for each cell and their intensities individually
calculated.

The radiation implementation in ABLATE must handle participating media, an absorbing media that attenuates passing
radiation. Radiation solvers are typically formulated to calculate the radiation into a point based on the amount of
radiation incoming from a solid sphere of directions.

The discrete transfer method involves decomposing the solid sphere into many discrete rays. These rays describe the
amount of radiation that reaches the point for which the radiation is being calculated.

The calculation of radiative intensity implemented in this radiation solver is based on the radiative transfer equation.
This equation describes how the intensity of radiation changes through participating media.

$$\frac{d I}{d x} = \kappa (\frac{\sigma T^4}{\pi} - I)$$

Note that the change in intensity is proportional to the product of the intensity difference and the absorption at a
given point. This radiative transfer equation is implemented for each ray. All rays are summed along the solid sphere.

## Computational Methods

The procedure for the implementation of this method is as follows. The solver is broken into an initialization and a
solve step.

The initialization of the solver forms the infrastructure for the gathering of ray information and
communication across domains.

The solve is completed in three stages. 1. Locally compute the source and absorption for each stored ray ID while
updating the segment values to the fields of the particles. (based on this ray ID) 2. Send the particles to their origin
ranks in order that the final computation can be completed. 3. Compute energy for every cell by searching through all
present particles.

## Verification

Multiple test cases will be used to verify that the solver is properly functioning. The first test case used is a set of
parallel plates with a media of defined temperature distribution. The one dimensional analytical solution of this
problem is compared against the results from the solver in order to define the error.

    /** To transport a particle from one location to another, this simply happens within a coordinate field. The particle is transported to a different rank based on its coordinates every time Migrate
     * is called. The initialization particle field can have a field of coordinates that the DMLocatePoints function reads from in order to build the local storage of ray segments. This field could be
     * essentially deleted during the solve portion. It must be replaced with a set of particles associated with every ray segment. The field initialized for the solve portion will have more particles
     * than the initial field. Having two fields is easier than dynamically adjusting the size of the particle field as the ray length increases for each ray.
     *
     * Steps of the search:
     *      Initialize a particle field with particles at the coordinates of their origin cell, one for each ray. (The search field should probably be a PIC field because it interacts with the mesh)
     *      Store the direction of the ray motion in the particle as a field.
     *      Loop through the particles that are present within a given domain.
     *      March the particle coordinates in the direction of the direction vector.
     *          Do existing ray filling routine.
     *          Run swarm migrate and check if the particle has left the domain for every space step that is taken. This is currently the best known way to check for domain crosses.
     *          If yes: Finish that ray segment and store it with its ray ID / domain number.
     *          If no: Repeat march and filling routine.
     *      The ray segments should be stored as vectors, with the indices matching the ray identity. These indices can be the same as the existing rays vector most likely.
     *      The difference is that this is an entirely local variable. Only the local ray segment identities which have ray segments passing through this domain will be non-empty for the local rays
     *      vector. This provides a global indexing scheme that the particles and domains can interface between without occupying a lot of local memory.
     *          Sub-task: During the cell search, form a vector (the same rays vector) from the information provided by the particle. In other words, the particle will "seed" the ray segments
     *          within each domain. Just pack the ray segment into whatever rays index matches the global scheme. This way when the particles are looped through in their local configuration, the
     *          memory location of the local ray segment can be accessed immediately.
     *          Sub-task: As the particle search routine is taking place, they should be simultaneously forming a particle field containing the solve field characterstics. This includes the
     *
     * Steps of the solve:
     *      Locally compute the source and absorption for each stored ray ID. (Loop through the local ray segments by index and run through them if they are not empty).
     *      Update the values to the fields of the particles (based on this ray ID). (Loop through the particles in the domain and update the values by the assocated index)
     *      Send the particles to their origin ranks.
     *      Compute energy for every cell by searching through all present particles.
     *      Delete the particles that are not from this rank.
     *
     * The local calculation of the ray absorption and intensity needs to be enabled by the local storage of ray segment cell indices.
     * This could be achieved by storing them within a vector that contains identifying information.
     *      Sub-task: The local calculation must loop through all ray identities, doing the calculation for only those rays that are present within the process. (If this segment index !empty)
     *      Sub-task: The ray segments must update the particle field by looping though the particles present in the domain and grabbing the calculated values from their associated ray segments.
     *      Since the associated ray segments are globally indexed, this might be faster.
     *
     * During the communication solve portion, the only information that needs to be transported is: ray ID, K, Ij, and domain #.
     *      Sub-task: Loop through every particle and call a non-deleting migrate on every particle in the domain (which does not belong to this domain).
     *
     *
     * During the global solve, each process needs to loop through the ray identities that are stored within that subdomain.
     *      Sub-task: Figure out how to iterate through cells within a single process and not the global subdomain.
     *      Sub-task: Delete the cells that are not from this rank.
     * */