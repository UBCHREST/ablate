---
layout: default
title: Radiation Formulation
nav_order: 8
has_children: false
---

## Mathematical Formulation

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