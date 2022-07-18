#include "radiation.hpp"
#include <set>
#include <utility>
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "utilities/mathUtilities.hpp"

ablate::radiation::Radiation::Radiation(std::string solverId, std::shared_ptr<domain::Region> region, const PetscInt raynumber, std::shared_ptr<parameters::Parameters> options,
                                        std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, std::shared_ptr<ablate::monitors::logs::Log> log)
    : CellSolver(std::move(solverId), std::move(region), std::move(options)), radiationModel(std::move(radiationModelIn)), log(std::move(log)) {
    nTheta = raynumber;    //!< The number of angles to solve with, given by user input
    nPhi = 2 * raynumber;  //!< The number of angles to solve with, given by user input
}

void ablate::radiation::Radiation::Setup() { /** allows initialization after the subdomain and dm is established */
    ablate::solver::CellSolver::Setup();
    dim = subDomain->GetDimensions();  //!< Number of dimensions already defined in the setup
}

void ablate::radiation::Radiation::Initialize() {
    /** Transporting a particle from one location to another simply happens within a coordinate field. The particle is transported to a different rank based on its coordinates every time Migrate
     * is called. The initialization particle field has a field of coordinates that the DMLocatePoints function reads from in order to build the local storage of ray segments. This field ceases to
     * exist at the end of initialization. It must be replaced with a set of particles associated with every ray segment. The solve particle field will be based only on the local ray segment and the
     * rank that it is travelling to. In other words, these particles have no coordinates or spatial location. The field initialized for the solve portion will have more particles than the initial
     * field. This is because the solve field will be represented by a particle for every ray segment as opposed to a particle for every ray. Having two fields is easier than dynamically adjusting the
     * size of the particle field as the ray length increases for each ray.
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
     *          Sub-task: As the particle search routine is taking place, they should be simultaneously forming a particle field containing the solve field characteristics. This includes the
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

    /** Begins radiation properties model
     * Runs the ray initialization, finding cell indices
     * Initialize the log if provided
     */
    // Store the required data for the low level c functions
    absorptivityFunction = radiationModel->GetRadiationPropertiesTemperatureFunction(eos::radiationProperties::RadiationProperty::Absorptivity, subDomain->GetFields());

    if (log) {
        log->Initialize(subDomain->GetComm());
    }
    this->RayInit();
}

void ablate::radiation::Radiation::RayInit() {
    /** Initialization to call, draws each ray vector and gets all of the cells associated with it
     * (sorted by distance and starting at the boundary working in)
     * This is done by creating particles at the center of each cell and iterating through them
     * */

    /** Get setup things for the position vector of the current cell index
     * Declare the variables that will contain the geometry of the cells
     * Obtain the geometric information about the cells in the DM
     * */

    if (log) StartEvent("Radiation Initialization");
    if (log) PetscPrintf(subDomain->GetComm(), "Starting Initialize\n");

    const PetscScalar* cellGeomArray;
    PetscReal minCellRadius;
    DM cellDM;
    VecGetDM(cellGeomVec, &cellDM);
    DMPlexGetGeometryFVM(cellDM, nullptr, nullptr, &minCellRadius) >> checkError;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    PetscFVCellGeom* cellGeom;

    PetscMPIInt rank;
    MPI_Comm_rank(subDomain->GetComm(), &rank);      //!< Get the origin rank of the current process. The particle belongs to this rank. The rank only needs to be read once.
    MPI_Comm_size(subDomain->GetComm(), &numRanks);  //!< Get the number of ranks in the simulation.

    /** Declare some local variables */
    double theta;  //!< represents the actual current angle (inclination)
    double phi;    //!< represents the actual current angle (rotation)

    /**Locally get a range of cells that are included in this subdomain at this time step for the ray initialization
     * */
    solver::Range cellRange;
    GetCellRange(cellRange);

    //    if (log) printf("Checking Ghost Labels\n");

    // check to see if there is a ghost label
    DMLabel ghostLabel;
    DMGetLabel(subDomain->GetDM(), "ghost", &ghostLabel) >> checkError;
    PetscInt cellCount = 0;

    /** Make sure that the cells being iterated over are within the region of the subdomain
     *
     * */
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        const PetscInt cell = cellRange.points ? cellRange.points[c] : c;

        // make sure we are not working on a ghost cell
        PetscInt ghost = -1;
        if (ghostLabel) {
            DMLabelGetValue(ghostLabel, cell, &ghost) >> checkError;
        }
        if (ghost < 0) {
            cellCount++;
        }
    }

    //    if (log) printf("Particle DMs\n");

    /** Setup the particles and their associated fields including: origin domain/ ray identifier / # domains crossed, and coordinates. Instantiate ray particles for each local cell only. */

    PetscInt npoints = (cellCount) * (nTheta - 1) * nPhi;  //!< Number of points to insert into the particle field. One particle for each ray.
    PetscInt nsolvepoints = 0;                             //!< Counts the solve points in the current domain. This will be adjusted over the course of the loop.

    /** Create the DMSwarm */
    DMCreate(subDomain->GetComm(), &radsearch) >> checkError;
    DMSetType(radsearch, DMSWARM) >> checkError;
    DMSetDimension(radsearch, dim) >> checkError;

    DMCreate(subDomain->GetComm(), &radsolve) >> checkError;
    DMSetType(radsolve, DMSWARM) >> checkError;
    DMSetDimension(radsolve, dim) >> checkError;

    /** Configure radsearch to be of type PIC/Basic */
    DMSwarmSetType(radsearch, DMSWARM_PIC) >> checkError;
    DMSwarmSetCellDM(radsearch, subDomain->GetDM()) >> checkError;

    DMSwarmSetType(radsolve, DMSWARM_BASIC) >> checkError;
    DMSwarmSetCellDM(radsolve, subDomain->GetDM()) >> checkError;

    //    if (log) printf("Particle Fields\n");

    /** Register fields within the DMSwarm */
    DMSwarmRegisterUserStructField(radsearch, "identifier", sizeof(Identifier)) >> checkError;  //!< A field to store the ray identifier [origin][iCell][ntheta][nphi][ndomain]
    DMSwarmRegisterUserStructField(radsearch, "virtual coord", sizeof(Virtualcoord)) >>
        checkError;                                         //!< A field representing the three dimensional coordinates of the particle. Three "virtual" dims are required.
    DMSwarmFinalizeFieldRegister(radsearch) >> checkError;  //!< Initialize the fields that have been defined

    DMSwarmRegisterUserStructField(radsolve, "identifier", sizeof(Identifier)) >> checkError;  //!< A field to store the ray identifier [origin][iCell][ntheta][nphi][ndomain]
    DMSwarmRegisterUserStructField(radsolve, "carrier", sizeof(Carrier)) >> checkError;        //!< A struct to carry information about the ray segment that the particle is communicating from
    DMSwarmFinalizeFieldRegister(radsolve) >> checkError;                                      //!< Initialize the fields that have been defined

    /** Set initial local sizes of the DMSwarm with a buffer length of zero */
    DMSwarmSetLocalSizes(radsearch, npoints, 10) >> checkError;  //!< Set the number of initial particles to the number of rays in the subdomain. Set the buffer size to zero.
    DMSwarmSetLocalSizes(radsolve, 0, 10) >> checkError;         //!< Set the number of initial particles to the number of rays in the subdomain. Set the buffer size to zero.

    /** Set the spatial step size to the minimum cell radius */
    PetscReal hstep = minCellRadius;

    /** Declare some information associated with the field declarations */
    PetscReal* coord;                    //!< Pointer to the coordinate field information
    struct Virtualcoord* virtualcoord;   //!< Pointer to the primary (virtual) coordinate field information
    struct Identifier* identifier;       //!< Pointer to the ray identifier information
    struct Carrier* carrier;             //!< Pointer to the ray carrier information
    struct Identifier* solveidentifier;  //!< Pointer to the ray identifier information

    /** Get the fields associated with the particle swarm so that they can be modified */
    DMSwarmGetField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void**)&coord) >> checkError;
    DMSwarmGetField(radsearch, "identifier", NULL, NULL, (void**)&identifier) >> checkError;
    DMSwarmGetField(radsearch, "virtual coord", NULL, NULL, (void**)&virtualcoord) >> checkError;

    //    if (log) printf("Writing Locations\n");

    PetscInt ipart = 0;  //!< Initialize a counter to represent the particle index. This will be iterated every time that the inner loop is passed through.

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {            //!< This will iterate only though local cells
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;  //!< Isolates the valid cells

        /** make sure we are not working on a ghost cell */
        PetscInt ghost = -1;
        if (ghostLabel) {
            DMLabelGetValue(ghostLabel, iCell, &ghost) >> checkError;
        }
        if (ghost >= 0) {
            continue;
        }

        DMPlexPointLocalRead(cellDM, iCell, cellGeomArray, &cellGeom) >> checkError;  //!< Reads the cell location from the current cell

        /** for every angle theta
         * for every angle phi
         */
        for (int ntheta = 1; ntheta < nTheta; ntheta++) {
            for (int nphi = 0; nphi < nPhi; nphi++) {
                //                if (!(cellGeom->centroid[0] < 0.3 && cellGeom->centroid[0] > 0.2 && cellGeom->centroid[1] < (0.0105 / 3.0) && cellGeom->centroid[1] > (-0.0105 / 3.0))) {
                //                    continue;
                //                }

                /** Get the particle coordinate field and write the cellGeom->centroid[xyz] into it */
                virtualcoord[ipart].x = cellGeom->centroid[0];
                virtualcoord[ipart].y = cellGeom->centroid[1];
                virtualcoord[ipart].z = cellGeom->centroid[2];
                virtualcoord[ipart].current = -1;  //!< Set this to a null value so that it can't get confused about where it starts.

                /** Get the initial direction of the search particle from the angle number that it was initialized with */
                theta = ((double)ntheta / (double)nTheta) * pi;  //!< Theta angle of the ray
                phi = ((double)nphi / (double)nPhi) * 2.0 * pi;  //!<  Phi angle of the ray

                /** Update the direction vector of the search particle */
                virtualcoord[ipart].xdir = hstep * (sin(theta) * cos(phi));  //!< x component conversion from spherical coordinates, adding the position of the current cell
                virtualcoord[ipart].ydir = hstep * (sin(theta) * sin(phi));  //!< y component conversion from spherical coordinates, adding the position of the current cell
                virtualcoord[ipart].zdir = hstep * (cos(theta));             //!< z component conversion from spherical coordinates, adding the position of the current cell

                /** Update the physical coordinate field so that the real particle location can be updated. */
                UpdateCoordinates(ipart, virtualcoord, coord);

                /** Label the particle with the ray identifier. (Use an array of 4 ints, [ncell][theta][phi][domains crossed])
                 * Label the particle with nsegment = 0; so that this can be iterated after each domain cross.
                 * */
                identifier[ipart].origin = rank;    //!< Input the ray identifier. This location scheme represents stepping four entries for every particle index increase
                identifier[ipart].iCell = iCell;    //!< Input the ray identifier. This location scheme represents stepping four entries for every particle index increase
                identifier[ipart].ntheta = ntheta;  //!< Input the ray identifier.
                identifier[ipart].nphi = nphi;      //!< Input the ray identifier.
                identifier[ipart].nsegment = 0;     //!< Initialize the number of domains crossed as zero

                /** Set the index of the field value so that it can be written to for every particle */
                ipart++;  //!< Must be iterated at the end since the value is initialized at zero.
            }
        }
    }

    /** Restore the fields associated with the particles */
    DMSwarmRestoreField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void**)&coord) >> checkError;
    DMSwarmRestoreField(radsearch, "identifier", NULL, NULL, (void**)&identifier) >> checkError;
    DMSwarmRestoreField(radsearch, "virtual coord", NULL, NULL, (void**)&virtualcoord) >> checkError;

    if (log) {
        PetscPrintf(subDomain->GetComm(), "Particles Initialized\n");
    }

    /** ***********************************************************************************************************************************************
     * Now that the particles have been created, they can be iterated over and each marched one step in space. The global indices of the local
     * ray segment storage can be easily accessed and appended. This forms a local collection of globally index ray segments.
     * */

    PetscInt nglobalpoints = 0;
    DMSwarmGetLocalSize(radsearch, &npoints) >> checkError;  //!< Recalculate the number of particles that are in the domain
    DMSwarmGetSize(radsearch, &nglobalpoints) >> checkError;
    PetscInt stepcount = 0;       //!< Count the number of steps that the particles have taken
    while (nglobalpoints != 0) {  //!< WHILE THERE ARE PARTICLES IN ANY DOMAIN
        /** Get all of the ray information from the particle
         * Get the ntheta and nphi from the particle that is currently being looked at. This will be used to identify its ray and calculate its direction. */
        DMSwarmGetField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void**)&coord) >> checkError;
        DMSwarmGetField(radsearch, "identifier", NULL, NULL, (void**)&identifier) >> checkError;
        DMSwarmGetField(radsearch, "virtual coord", NULL, NULL, (void**)&virtualcoord) >> checkError;

        /** Iterate over the particles that are present in the domain
         * Add the cell index to the ray
         * Step every particle in the domain one step and then perform a migration
         * */
        PetscInt index;
        Vec intersect;
        VecCreate(PETSC_COMM_SELF, &intersect) >> checkError;  //!< Instantiates the vector
        VecSetBlockSize(intersect, dim) >> checkError;
        VecSetSizes(intersect, PETSC_DECIDE, npoints * dim) >> checkError;  //!< Set size
        VecSetFromOptions(intersect) >> checkError;
        PetscInt i[3] = {0, 1, 2};              //!< Establish the vector here so that it can be iterated.
        for (int ip = 0; ip < npoints; ip++) {  //!< Iterate over the particles present in the domain. How to isolate the particles in this domain and iterate over them? If there are no
                                                //!< particles then pass out of initialization.
            ipart = ip;                         //!< Set the particle index as a different variable in the loop so it doesn't make the compiler unhappy.

            /** Update the physical coordinate field so that the real particle location can be updated. */
            UpdateCoordinates(ipart, virtualcoord, coord);  //!< Update the particle coordinates into the physical coordinate system

            /** FIRST TAKE THIS LOCATION INTO THE RAYS VECTOR
             * "I found a particle in my domain. Maybe it was just moved here and I've never seen it before.
             * Therefore, my first step should be to add this location to the local rays vector. Then I can adjust the coordinates and migrate the particle." */

            /** Get the particle coordinates here and put them into the intersect */
            PetscReal position[3] = {(virtualcoord[ipart].x),   // x component conversion from spherical coordinates, adding the position of the current cell
                                     (virtualcoord[ipart].y),   // y component conversion from spherical coordinates, adding the position of the current cell
                                     (virtualcoord[ipart].z)};  // z component conversion from spherical coordinates, adding the position of the current cell

            /** This block creates the vector pointing to the cell whose index will be stored during the current loop */
            VecSetValues(intersect, dim, i, position, INSERT_VALUES);  //!< Actually input the values of the vector (There are 'dim' values to input)
            i[0] += dim;                                               //!< Iterate the index by the number of dimensions so that the DMLocatePoints function can be called collectively.
            i[1] += dim;
            i[2] += dim;
        }

        /** Loop through points to try to get the cell that is sitting on that point*/
        PetscSF cellSF = nullptr;  //!< PETSc object for setting up and managing the communication of certain entries of arrays and Vecs between MPI processes.
        DMLocatePoints(subDomain->GetDM(), intersect, DM_POINTLOCATION_NONE, &cellSF) >> checkError;  //!< Call DMLocatePoints here, all of the processes have to call it at once.

        /** An array that maps each point to its containing cell can be obtained with the below
         * We want to get a PetscInt index out of the DMLocatePoints function (cell[n].index)
         * */
        PetscInt nFound;
        const PetscInt* point = nullptr;
        const PetscSFNode* cell = nullptr;
        PetscSFGetGraph(cellSF, nullptr, &nFound, &point, &cell) >> checkError;  //!< Using this to get the petsc int cell number from the struct (SF)

        for (int ip = 0; ip < npoints; ip++) {  //!< Iterate over the particles present in the domain. How to isolate the particles in this domain and iterate over them? If there are no

            ipart = ip;  //!< Iterate the loop variable

            /** IF THE CELL NUMBER IS RETURNED NEGATIVE, THEN WE HAVE REACHED THE BOUNDARY OF THE DOMAIN >> This exits the loop
             * This function returns multiple values if multiple points are input to it
             * Make sure that whatever cell is returned is in the stencil set (and not outside of the radiation domain)
             * Assemble a vector of vectors etc associated with each cell index, angular coordinate, and space step?
             * The boundary has been reached if any of these conditions don't hold
             * */
            /** make sure we are not working on a ghost cell */
            PetscInt ghost = -1;
            if (ghostLabel) DMLabelGetValue(ghostLabel, cell[ip].index, &ghost) >> checkError;
            if (nFound > -1 && cell[ip].index >= 0 && subDomain->InRegion(cell[ip].index) && (ghost == -1)) {
                index = cell[ip].index;
            } else {
                DMSwarmRestoreField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void**)&coord) >> checkError;
                DMSwarmRestoreField(radsearch, "identifier", NULL, NULL, (void**)&identifier) >> checkError;
                DMSwarmRestoreField(radsearch, "virtual coord", NULL, NULL, (void**)&virtualcoord) >> checkError;

                DMSwarmRemovePointAtIndex(radsearch, ipart);  //!< Delete the particle!

                DMSwarmGetLocalSize(radsearch, &npoints) >> checkError;  //!< Need to recalculate the number of particles that are in the domain again
                DMSwarmGetField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void**)&coord) >> checkError;
                DMSwarmGetField(radsearch, "identifier", NULL, NULL, (void**)&identifier) >> checkError;
                DMSwarmGetField(radsearch, "virtual coord", NULL, NULL, (void**)&virtualcoord) >> checkError;
                ipart--;  //!< Check the point replacing the one that was deleted
                ip--;
                index = -1;
                continue;
            }

            if (index > -1) {
                if (virtualcoord[ipart].current != index) {
                    /** If this local rank has never seen this search particle before, then it needs to add a new ray segment to local memory
                     * Hash the identifier into a key value that can be used in the map
                     * We should only iterate the identifier of the search particle (/ add a solver particle) if the point is valid in the domain and is being used
                     * */
                    if (rays.count(Key(identifier[ipart])) ==
                        0) {                                      //!< IF THIS RAYS VECTOR IS EMPTY FOR THIS DOMAIN, THEN THE PARTICLE HAS NEVER BEEN HERE BEFORE. THEREFORE, ITERATE THE NDOMAINS BY 1.
                        identifier[ipart].nsegment++;             //!< The particle has passed through another domain!
                        DMSwarmAddPoint(radsolve) >> checkError;  //!< Another solve particle is added here because the search particle has entered a new domain

                        DMSwarmGetLocalSize(radsolve,
                                            &nsolvepoints) >>
                            checkError;  //!< Recalculate the number of solve particles so that the last one in the list can be accessed. (I assume that the last one is newest)

                        DMSwarmGetField(radsolve, "identifier", NULL, NULL, (void**)&solveidentifier) >>
                            checkError;  //!< Get the fields from the radsolve swarm so the new point can be written to them
                        DMSwarmGetField(radsolve, "carrier", NULL, NULL, (void**)&carrier) >> checkError;

                        PetscInt newpoint = nsolvepoints - 1;           //!< This must be replaced with the index of whatever particle there is. Maybe the last index?
                        solveidentifier[newpoint] = identifier[ipart];  //!< Give the particle an identifier which matches the particle it was created with
                        carrier[newpoint].Krad = 1;  //!< The new particle gets an empty carrier because it is holding no information yet (Krad must be initialized to 1 here: everything is init 0)

                        DMSwarmRestoreField(radsolve, "identifier", NULL, NULL, (void**)&solveidentifier) >> checkError;  //!< The fields must be returned so that the swarm can be updated correctly?
                        DMSwarmRestoreField(radsolve, "carrier", NULL, NULL, (void**)&carrier) >> checkError;
                    }

                    /** ********************************************
                     * Adaptive stepping stuff lives here: to be added after each time the position is updated
                     * The current cell should be added before the loop begins*/
                    rays[Key(identifier[ipart])].cells.push_back(index);
                    rays[Key(identifier[ipart])].h.push_back(virtualcoord[ipart].hhere);  //!< Add this space step if the current index is being added.
                    virtualcoord[ipart].hhere = 0;
                    virtualcoord[ipart].current = index;  //!< Sets the current cell for the adaptive space stepping to compare against

                } else {
                    virtualcoord[ipart].hhere += hstep;  //!< If the cell is not different then we simply increase the stored path length by one step.
                }
            }

            /** Step the vector forward in space until it is no longer in the cell it was ins
             * After the coordinates have left the cell it was it, the coordinates of the particle should be updated
             * Update the coordinates of the particle (virtual and physical)
             * */
            virtualcoord[ipart].x += virtualcoord[ipart].xdir;  //!< x component: add one step to the coordinate position
            virtualcoord[ipart].y += virtualcoord[ipart].ydir;  //!< y component: add one step to the coordinate position
            virtualcoord[ipart].z += virtualcoord[ipart].zdir;  //!< z component: add one step to the coordinate position

            UpdateCoordinates(ipart, virtualcoord, coord);  //!< Update the particle coordinates into the physical coordinate system
        }
        /** Restore the fields associated with the particles after all of the particles have been stepped */
        DMSwarmRestoreField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void**)&coord) >> checkError;
        DMSwarmRestoreField(radsearch, "identifier", NULL, NULL, (void**)&identifier) >> checkError;
        DMSwarmRestoreField(radsearch, "virtual coord", NULL, NULL, (void**)&virtualcoord) >> checkError;

        /** Cleanup */
        VecDestroy(&intersect) >> checkError;   //!< Return the vector to PETSc
        PetscSFDestroy(&cellSF) >> checkError;  //!< Return the stuff to PETSc

        if (log) PetscPrintf(subDomain->GetComm(), "Migrate ...");

        /** DMSwarm Migrate to move the ray search particle into the next domain if it has crossed. If it no longer appears in this domain then end the ray segment. */
        DMSwarmMigrate(radsearch, PETSC_TRUE) >> checkError;  //!< Migrate the search particles and remove the particles that have left the domain space.

        DMSwarmGetSize(radsearch, &nglobalpoints) >> checkError;  //!< Update the loop condition. Recalculate the number of particles that are in the domain.
        DMSwarmGetLocalSize(radsearch, &npoints) >> checkError;   //!< Update the loop condition. Recalculate the number of particles that are in the domain.

        if (log) {
            PetscPrintf(subDomain->GetComm(), " Global Steps: %" PetscInt_FMT "    Global Points: %" PetscInt_FMT "\n", stepcount, nglobalpoints);
            stepcount++;
        }
    }
    /** Cleanup*/
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    RestoreRange(cellRange);

    if (log) EndEvent();
}

PetscErrorCode ablate::radiation::Radiation::ComputeRHSFunction(PetscReal time, Vec solVec, Vec rhs) {
    PetscFunctionBeginUser;

    StartEvent("Radiation Solve");

    /** Get the array of the local f vector, put the intensity into part of that array instead of using the radiative gain variable. */
    const PetscScalar* rhsArray;
    VecGetArrayRead(rhs, &rhsArray);

    /** Get the array of the solution vector. */
    const PetscScalar* solArray;
    VecGetArrayRead(solVec, &solArray);

    /** Get the array of the aux vector. */
    const auto auxVec = subDomain->GetAuxVector();
    const PetscScalar* auxArray;
    VecGetArrayRead(auxVec, &auxArray);

    const auto& eulerFieldInfo = subDomain->GetField("euler");

    /** Get the temperature field.
     * For ABLATE implementation, get temperature based on this function.
     */
    const auto& temperatureField = subDomain->GetField("temperature");

    /** Declare the basic information*/
    PetscReal* sol;          //!< The solution value at any given location
    PetscReal* temperature;  //!< The temperature at any given location
    PetscReal dTheta = pi / nTheta;
    PetscReal dPhi = (2 * pi) / nPhi;
    double kappa = 1;  //!< Absorptivity coefficient, property of each cell
    double theta;

    std::vector<std::vector<PetscReal>> locations;  //!< 2 Dimensional vector which stores the locations of the cell centers

    auto absorptivityFunctionContext = absorptivityFunction.context.get();  //!< Get access to the absorption function

    /** Declare some information associated with the field declarations */
    struct Carrier* carrier;        //!< Pointer to the ray carrier information
    struct Identifier* identifier;  //!< Pointer to the ray identifier information

    /** Get the current rank associated with this process */
    PetscMPIInt rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  //!< Get the origin rank of the current process. The particle belongs to this rank. The rank only needs to be read once.

    /** Get all of the ray information from the particle
     * Get the ntheta and nphi from the particle that is currently being looked at. This will be used to identify its ray and calculate its direction. */
    //    if (log) StartEvent("GetFields");
    PetscInt npoints;
    DMSwarmGetLocalSize(radsolve, &npoints);  //!< Recalculate the number of particles that are in the domain
    DMSwarmGetField(radsolve, "identifier", NULL, NULL, (void**)&identifier);
    DMSwarmGetField(radsolve, "carrier", NULL, NULL, (void**)&carrier);
    //    if (log) EndEvent();

    /** ********************************************************************************************************************************
     * Iterate over the particles that are present in the domain
     * The cells that are in the domain at this point should represent the solve cells attached to the ray segments. They will be transported after local calculation and the non-native ones will
     * be destroyed.
     * First the particles should be zeroed in case they are carrying information from the last time step.
     * Then the entire solve sequence can be run through. This will require that the particles are iterated through twice.
     * */
    //    if (log) StartEvent("Zero Particles");
    for (int ipart = 0; ipart < npoints; ipart++) {  //!< Iterate through the particles in the space to zero their information.
        carrier[ipart].Ij = 0;                       //!< Zero the intensity of the segment
        carrier[ipart].Krad = 1;                     //!< Zero the total absorption for this domain
        carrier[ipart].I0 = 0;                       //!< Zero the initial intensity of the ray segment
    }
    //    if (log) EndEvent();
    /** Now that the particle information has been zeroed, the solve can begin. */
    for (int ipart = 0; ipart < npoints; ipart++) {  //!< Iterate over the particles present in the domain. How to isolate the particles in this domain and iterate over them? If there are no
                                                     //!< particles then pass out of initialization.
        /** Each ray is born here. They begin at the far field temperature.
            Initial ray intensity should be set based on which boundary it is coming from.
            If the ray originates from the walls, then set the initial ray intensity to the wall temperature, etc.
         */
        /** For each domain in the ray (The rays vector will have an added index, splitting every x points) */
        int numPoints = static_cast<int>(rays[Key(identifier[ipart])].cells.size());

        if (numPoints > 0) {
            for (int n = 0; n < (numPoints); n++) {
                /** Go through every cell point that is stored within the ray >> FROM THE BOUNDARY TO THE SOURCE
                    Define the absorptivity and temperature in this section
                    For ABLATE implementation, get temperature based on this function
                    Get the array that lives inside the vector
                    Gets the temperature from the cell index specified
                */
                //                if (log) StartEvent("Field Read");
                DMPlexPointLocalFieldRead(subDomain->GetDM(), rays[Key(identifier[ipart])].cells[n], temperatureField.id, auxArray, &temperature);
                DMPlexPointLocalRead(subDomain->GetDM(), rays[Key(identifier[ipart])].cells[n], solArray, &sol);
                /** Input absorptivity (kappa) values from model here. */
                absorptivityFunction.function(sol, *temperature, &kappa, absorptivityFunctionContext);
                //                if (log) EndEvent();

                //                if (log) StartEvent("Local Compute");
                carrier[ipart].Ij += FlameIntensity(1 - exp(-kappa * rays[Key(identifier[ipart])].h[n]), *temperature) * carrier[ipart].Krad;
                carrier[ipart].Krad *= exp(-kappa * rays[Key(identifier[ipart])].h[n]);  //!< Compute the total absorption for this domain

                if (n == (numPoints - 1)) { /** If this is the beginning of the ray, set this as the initial intensity. (The segment intensities will be filtered through during the origin run) */
                    carrier[ipart].I0 = FlameIntensity(1, *temperature);  //!< Set the initial intensity of the ray segment
                }
                //                if (log) EndEvent();
            }
        }
    }

    /** Restore the fields associated with the particles after all of the particles have been stepped */
    DMSwarmRestoreField(radsolve, "identifier", NULL, NULL, (void**)&identifier);
    DMSwarmRestoreField(radsolve, "carrier", NULL, NULL, (void**)&carrier);

    /** ********************************************************************************************************************************
     * Now the carrier has all of the information from the rays that are needed to compute the final ray intensity. Therefore, we will perform the migration.
     * Then, all of the carrier particles will be looped through and the local Origins associated with each cell will be updated
     * */
    //    if (log) StartEvent("Rank Assign");
    PetscInt* rankid;
    DMSwarmGetField(radsolve, "DMSwarm_rank", NULL, NULL, (void**)&rankid);
    DMSwarmGetField(radsolve, "identifier", NULL, NULL, (void**)&identifier);
    for (int ipart = 0; ipart < npoints; ipart++) {
        rankid[ipart] = identifier[ipart].origin;
    }
    DMSwarmRestoreField(radsolve, "DMSwarm_rank", NULL, NULL, (void**)&rankid);
    DMSwarmRestoreField(radsolve, "identifier", NULL, NULL, (void**)&identifier);
    //    if (log) EndEvent();

    //    if (log) StartEvent("Solve Migrate");
    DMSwarmMigrate(radsolve, PETSC_FALSE);  //!< After iterating through all of the particles, perform a migration to the origin ranks. This will move the particles.
                                            //    if (log) EndEvent();

    /** ********************************************************************************************************************************
     * Now iterate through all of the particles in order to perform the information transfer */
    DMSwarmGetLocalSize(radsolve, &npoints);                                   //!< Recalculate the number of particles that are in the domain
    DMSwarmGetField(radsolve, "identifier", NULL, NULL, (void**)&identifier);  //!< Field information is needed in order to read data from the incoming particles.
    DMSwarmGetField(radsolve, "carrier", NULL, NULL, (void**)&carrier);

    /** Iterate through the particles and offload the information to their associated origin cell struct. */
    for (int ipart = 0; ipart < npoints; ipart++) {
        origin[identifier[ipart].iCell].handler[Key(identifier[ipart])].Krad = carrier[ipart].Krad;
        origin[identifier[ipart].iCell].handler[Key(identifier[ipart])].Ij = carrier[ipart].Ij;
        origin[identifier[ipart].iCell].handler[Key(identifier[ipart])].I0 = carrier[ipart].I0;
    }

    /** ********************************************************************************************************************************
     * Now iterate through all of the ray identifiers in order to compute the final ray intensities */

    solver::Range cellRange;  //!< Access to the cell index information is important here to get all of the ray identifier information.
    GetCellRange(cellRange);

    /** check to see if there is a ghost label */
    DMLabel ghostLabel;
    DMGetLabel(subDomain->GetDM(), "ghost", &ghostLabel) >> checkError;

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {            //!< This will iterate only though local cells
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;  //!< Isolates the valid cells

        /** Make sure we are not working on a ghost cell */
        PetscInt ghost = -1;
        if (ghostLabel) {
            DMLabelGetValue(ghostLabel, iCell, &ghost);
        }
        if (ghost >= 0) {
            continue;
        }

        origin[iCell].intensity = 0;  //!< Make sure to zero the intensity of every cell before beginning to calculate the intensity for this time step.

        /** for every angle theta
         * for every angle phi
         */
        for (int ntheta = 1; ntheta < nTheta; ntheta++) {
            for (int nphi = 0; nphi < nPhi; nphi++) {
                /** Now that we are iterating over every ray identifier in this local domain, we can get all of the particles that are associated with this ray.
                 * We will need to sort the rays in order of domain segment. We need to start at the end of the ray and go towards the beginning of the ray. */
                Identifier loopid = {.origin = rank, .iCell = iCell, .ntheta = ntheta, .nphi = nphi, .nsegment = 1};  //!< Instantiate an identifier associated with this loop location.

                /** Get the maximum nsegment by looping through all of the particles and searching for it. (This is dumb and slow but easy to think of)*/
                //                if (log) StartEvent("Order Segment");
                bool pointfound = true;
                PetscInt oldsegment = loopid.nsegment;
                while (pointfound) {
                    /** Starting at the first possible segment for this ID
                     * //                             If it exists, increase the segment number that is being checked for.
                     * Also, set the maximum segment that is available for this ray to the segment that is currently being checked.
                     * */
                    if (origin[iCell].handler.count(Key(loopid)) > 0) {
                        loopid.nsegment++;
                    }
                    pointfound = !(oldsegment == loopid.nsegment);  //!< If no point was found during the whole for loop, then we must have stumbled on the last segment in this ray.
                    oldsegment = loopid.nsegment;                   //!< Set the old segment
                }
                //                if (log) EndEvent();

                /** Now that we have found the maximum segment in the domain, we can iterate from the last segment to the beginning segment of this ray identifier */
                /** Iterate over the particles that are present in the domain
                 * The particles present at this point should represent the migrated particles carrying ray information in order to perform the final solve.
                 * The I0 (beginning ray intensity) will also need to be found before the ray is added.
                 * The source and absorption must be set to zero at the beginning of each new ray.
                 * */
                origin[iCell].Kradd = 1;       //!< This must be reset at the beginning of each new ray.
                origin[iCell].Isource = 0;     //!< This must be reset at the beginning of each new ray.
                loopid.nsegment--;             //!< Decrement the segment identifier to the last known segment that was found.
                oldsegment = loopid.nsegment;  //!< Set the old segment to be the head of the ray
                origin[iCell].I0 = 0;          //!< For the last segment in the domain, take that as the black body intensity of the far field.

                //                if (log) StartEvent("Global Compute");
                while (loopid.nsegment > 0) {  //!< Need to go through all of the ray segments until the origin of the ray is reached

                    origin[iCell].I0 = (oldsegment == loopid.nsegment) ? origin[iCell].handler[Key(loopid)].I0 : origin[iCell].I0;  //!< Set I0 if it is the last segment in the ray

                    /** Global ray computation happens here, grabbing values from the transported particles.
                     * The rays end here, their intensity is added to the total intensity of the cell.
                     * Gives the partial impact of the ray on the total sphere.
                     * The sin(theta) is a result of the polar coordinate discretization.
                     * In the parallel form at the end of each ray, the absorption of the initial ray and the absorption of the black body source are computed individually at the end.
                     * */
                    /** Parallel things are here
                     * Meaning that the variables required for the parallelizable analytical solution will be declared here */
                    origin[iCell].Isource += origin[iCell].handler[Key(loopid)].Ij * origin[iCell].Kradd;  //!< Add the black body radiation transmitted through the domain to the source term
                    origin[iCell].Kradd *= origin[iCell].handler[Key(loopid)].Krad;                        //!< Add the absorption for this domain to the total absorption of the ray
                    loopid.nsegment--;                                                                     //!< Decrement the segment number to move to the next closer segment in the ray.
                }
                //                if (log) EndEvent();

                //                if (log) StartEvent("Ray Sum");
                theta = ((double)ntheta / (double)nTheta) * pi;  //!< This is a fine method of determining theta because it is in the original domain
                origin[iCell].intensity += ((origin[iCell].I0 * origin[iCell].Kradd) + origin[iCell].Isource) * sin(theta) * dTheta * dPhi;  //!< Final ray calculation
                                                                                                                                             //                if (log)
                printf("I0: %f Kradd: %f Isource: %f\n", origin[iCell].I0, origin[iCell].Kradd, origin[iCell].Isource);                      //!< Debugging for Quartz
                //                if (log) EndEvent();
            }
        }
        //        if (log) PetscPrintf(PETSC_COMM_WORLD, "Cell: %" PetscInt_FMT " Intensity: %f\n", iCell, origin[iCell].intensity);
    }

    /** ********************************************************************************************************************************
     * Need to delete all of the particles that were transported to different domains so that the process can be repeated in the next step. */

    /** Delete all of the particles that were transported to their origin domains -> Delete if (identifier.origin == MPI_Rank() && identifier.nsegment != 0) */
    //    if (log) StartEvent("Delete Carriers");
    for (int ipart = 0; ipart < npoints; ipart++) {
        if (identifier[ipart].origin == rank && identifier[ipart].nsegment != 1) {
            DMSwarmRestoreField(radsolve, "identifier", NULL, NULL, (void**)&identifier);  //!< Need to restore the field access before deleting a point
            DMSwarmRestoreField(radsolve, "carrier", NULL, NULL, (void**)&carrier);

            DMSwarmRemovePointAtIndex(radsolve, ipart);  //!< Delete the particle!

            DMSwarmGetLocalSize(radsolve, &npoints);                                   //!< Need to recalculate the number of particles that are in the domain again
            DMSwarmGetField(radsolve, "identifier", NULL, NULL, (void**)&identifier);  //!< Get the field back
            DMSwarmGetField(radsolve, "carrier", NULL, NULL, (void**)&carrier);
            ipart--;  //!< Check the point replacing the one that was deleted
        }
    }

    /** Restore the fields associated with the particles after all of the particles have been stepped. */
    DMSwarmRestoreField(radsolve, "identifier", NULL, NULL, (void**)&identifier);
    DMSwarmRestoreField(radsolve, "carrier", NULL, NULL, (void**)&carrier);
    //    if (log) EndEvent();

    /** ********************************************************************************************************************************
     * Loop through the cell range and compute the origin contributions. */

    //    if (log) StartEvent("Cell Energy");
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {            //!< This will iterate only though local cells
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;  //!< Isolates the valid cells

        // make sure we are not working on a ghost cell
        PetscInt ghost = -1;
        if (ghostLabel) {
            DMLabelGetValue(ghostLabel, iCell, &ghost);
        }
        if (ghost >= 0) {
            continue;
        }

        /** Gets the temperature from the cell index specified */
        DMPlexPointLocalFieldRead(subDomain->GetDM(), iCell, temperatureField.id, auxArray, &temperature);

        /** Put the irradiation into the right hand side function */
        PetscScalar* rhsValues;
        DMPlexPointLocalFieldRead(subDomain->GetDM(), iCell, eulerFieldInfo.id, rhsArray, &rhsValues);
        PetscReal losses = 4 * sbc * *temperature * *temperature * *temperature * *temperature;
        rhsValues[ablate::finiteVolume::CompressibleFlowFields::RHOE] += -kappa * (losses - origin[iCell].intensity);
        if (log) printf("Cell: %" PetscInt_FMT " Intensity: %f\n", iCell, origin[iCell].intensity);
    }

    /** Cleanup*/
    VecRestoreArrayRead(rhs, &rhsArray);
    RestoreRange(cellRange);

    if (log) EndEvent();

    PetscFunctionReturn(0);
}

PetscReal ablate::radiation::Radiation::FlameIntensity(double epsilon, double temperature) { /** Gets the flame intensity based on temperature and emissivity*/
    const PetscReal sbc = 5.6696e-8;                                                         //!< Stefan-Boltzman Constant (J/K)
    const PetscReal pi = 3.1415926535897932384626433832795028841971693993;
    return epsilon * sbc * temperature * temperature * temperature * temperature / pi;
}

std::string ablate::radiation::Radiation::Key(Identifier id) {  //!< Nested Cantor pairing function in order to identify ray segment

    std::string key = std::to_string(id.origin) + "." + std::to_string(id.iCell) + "." + std::to_string(id.ntheta) + "." + std::to_string(id.nphi) + "." + std::to_string(id.nsegment);
    return key;
}

void ablate::radiation::Radiation::UpdateCoordinates(PetscInt ipart, Virtualcoord* virtualcoord, PetscReal* coord) {
    switch (dim) {
        case 2:                                        //!< If there are only two dimensions in this simulation
            coord[2 * ipart] = virtualcoord[ipart].x;  //!< Update the two physical coordinates
            coord[(2 * ipart) + 1] = virtualcoord[ipart].y;
            break;
        case 3:                                        //!< If there are three dimensions in this simulation
            coord[3 * ipart] = virtualcoord[ipart].x;  //!< Update the three physical coordinates
            coord[(3 * ipart) + 1] = virtualcoord[ipart].y;
            coord[(3 * ipart) + 2] = virtualcoord[ipart].z;
            break;
    }
}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::radiation::Radiation, "A solver for radiative heat transfer in participating media", ARG(std::string, "id", "the name of the flow field"),
         ARG(ablate::domain::Region, "region", "the region to apply this solver."), ARG(int, "rays", "number of rays used by the solver"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"),
         ARG(ablate::eos::radiationProperties::RadiationModel, "properties", "the radiation properties model"), OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));