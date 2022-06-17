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
    // TODO: For the solving step
    // TODO: The swarm particles can be teleported directly from one rank to another without removing the points from the current rank
    //     TODO: This means that the particles and their information can be sent during the solve without needing to send them back

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
    const PetscScalar* cellGeomArray;  //, *faceGeomArray;
    PetscReal minCellRadius;
    DM cellDM;  //, faceDM;
    VecGetDM(cellGeomVec, &cellDM);
    //    VecGetDM(faceGeomVec, &faceDM);
    DMPlexGetGeometryFVM(cellDM, nullptr, nullptr, &minCellRadius);
    VecGetArrayRead(cellGeomVec, &cellGeomArray);
    //    VecGetArrayRead(faceGeomVec, &faceGeomArray);
    PetscFVCellGeom* cellGeom;
    //    PetscFVFaceGeom* faceGeom;

    /** Declare some local variables */
    double theta;  //!< represents the actual current angle (inclination)
    double phi;    //!< represents the actual current angle (rotation)

    /**Locally get a range of cells that are included in this subdomain at this time step for the ray initialization
     * */
    solver::Range cellRange;
    GetCellRange(cellRange);
    PetscInt iCellMax = 0;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {      //!< This will iterate only though local cells
        PetscInt iCell = cellRange.points ? cellRange.points[c] : c;  //!< Isolates the valid cells
        iCellMax = (iCell > iCellMax) ? iCell : iCellMax;
    }

    /** Setup the particles and their associated fields including: origin domain/ ray identifier / # domains crossed, and coordinates. Instantiate ray particles for each local cell only. */

    PetscInt npoints = (cellRange.end - cellRange.start) * nTheta * nPhi;  //!< Number of points to insert into the particle field. One particle for each ray.

    /* Create the DMSwarm */
    DMCreate(PETSC_COMM_WORLD, &radsearch);
    DMSetType(radsearch, DMSWARM);
    DMSetDimension(radsearch, dim);

    /* Configure radsearch to be of type PIC */
    DMSwarmSetType(radsearch, DMSWARM_PIC);
    DMSwarmSetCellDM(radsearch, cellDM);

    /* Register fields within the DMSwarm */
    DMSwarmRegisterPetscDatatypeField(radsearch, "identifier", 5, PETSC_INT);      //!< A field to store the ray identifier [iCell][ntheta][nphi][ndomain]
    DMSwarmRegisterPetscDatatypeField(radsearch, "virtual coord", 3, PETSC_REAL);  //!< A field representing the three dimensional coordinates of the particle. Three "virtual" dims are required.
    DMSwarmFinalizeFieldRegister(radsearch);                                       //!< Initialize the fields that have been defined

    /* Set initial local sizes of the DMSwarm with a buffer length of zero */
    DMSwarmSetLocalSizes(radsearch, npoints, 0);  //!< Set the number of initial particles to the number of rays in the subdomain. Set the buffer size to zero.

    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);  //!< Get the number of ranks in the simulation.

    /** This is still relevant, as there will be global indexing of these values but with local information only.
     * Create a nested vector which can store cell locations based on origin cell, theta, phi, and space step
     * Preallocate the sub-vectors in order to avoid dynamic sizing as much as possible
     * Make vector to store this dimensional row
     * Indices: Cell, angle (theta), angle(phi), domain, space steps
     * */
    std::vector<PetscInt> rayDomains;
    std::vector<std::vector<PetscInt>> rayPhis;
    std::vector<std::vector<std::vector<PetscInt>>> rayThetas(nPhi, rayPhis);
    std::vector<std::vector<std::vector<std::vector<PetscInt>>>> rayCells(nTheta, rayThetas);
    std::vector<std::vector<std::vector<std::vector<std::vector<PetscInt>>>>> rayRanks((iCellMax + 1), rayCells);
    rays.resize(numRanks, rayRanks);
    h.resize(numRanks, rayRanks);  //!< Store a vector of space steps that the solver will use to compute absorption effects

    std::vector<PetscReal> Ij1Phis;
    std::vector<std::vector<PetscReal>> Ij1Thetas(nPhi, Ij1Phis);
    std::vector<std::vector<std::vector<PetscReal>>> Ij1Cells(nTheta, Ij1Thetas);
    std::vector<std::vector<std::vector<std::vector<PetscReal>>>> Ij1Ranks((iCellMax + 1), Ij1Cells);
    Ij1.resize(numRanks, Ij1Ranks);  //!< This sets the previous iteration intensity so that each ray can store multiple intensities.
    Krad.resize(numRanks, Ij1Ranks);

    /** Set the spatial step size to the minimum cell radius */
    PetscReal hstep = minCellRadius;

    PetscInt rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  //!< Get the origin rank of the current process. The particle belongs to this rank. The rank only needs to be read once.

    /** Declare some information associated with the field declarations */
    PetscReal* coord;         //!< Pointer to the coordinate field information
    PetscReal* virtualcoord;  //!< Pointer to the primary (virtual) coordinate field information
    PetscInt* identifier;     //!< Pointer to the ray identifier information
    // TODO: Need to define the additional field information pointers also

    /** Get the fields associated with the particle swarm so that they can be modified */
    //                DMSwarmGetLocalSize(radsearch, &npoints); //We already know the size of the particle field since it was just created. This is probably unnecessary.
    DMSwarmGetField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void**)&coord);
    DMSwarmGetField(radsearch, "identifier", NULL, NULL, (void**)&identifier);
    DMSwarmGetField(radsearch, "virtual coord", NULL, NULL, (void**)&virtualcoord);

    PetscInt ncells = 0;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {            //!< This will iterate only though local cells
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;  //!< Isolates the valid cells
        DMPlexPointLocalRead(cellDM, iCell, cellGeomArray, &cellGeom);      //!< Reads the cell location from the current cell

        /** for every angle theta
         * for every angle phi
         */
        for (int ntheta = 1; ntheta < nTheta; ntheta++) {
            for (int nphi = 0; nphi < nPhi; nphi++) {
                /** Set the index of the field value so that it can be written to for every particle */
                int i = ncells * ntheta * nphi;  //!< This represents the index of the particle field entry

                //                origin[i] = originrank;  //!< The origin rank of the particle, which is this domain (during the initialization)

                /** Get the particle coordinate field and write the cellGeom->centroid[xyz] into it */
                virtualcoord[3 * i] = cellGeom->centroid[0];
                virtualcoord[3 * i + 1] = cellGeom->centroid[1];
                virtualcoord[3 * i + 2] = cellGeom->centroid[2];

                /** Update the physical coordinate field so that the real particle location can be updated. */
                for (int d = 0; d < dim; d++) {                  //!< For the number of dimensions that actually exist physically.
                    coord[3 * i + d] = virtualcoord[3 * i + d];  //!< Insert the virtual coordinates into the physical coordinate field.
                }

                /** Label the particle with the ray identifier. (Use an array of 4 ints, [ncell][theta][phi][domains crossed])
                 * Label the particle with domainscrossed = 0; so that this can be iterated after each domain cross.
                 * */
                identifier[5 * i] = rank;        //!< Input the ray identifier. This location scheme represents stepping four entries for every particle index increase
                identifier[5 * i + 1] = iCell;   //!< Input the ray identifier. This location scheme represents stepping four entries for every particle index increase
                identifier[5 * i + 2] = ntheta;  //!< Input the ray identifier
                identifier[5 * i + 3] = nphi;    //!< Input the ray identifier
                identifier[5 * i + 4] = 0;       //!< Initialize the number of domains crossed as zero
            }
        }
        ncells++;  //!< Increase the number of cells that have been finished.
    }

    /** Restore the fields associated with the particles */
    DMSwarmRestoreField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void**)&coord);
    DMSwarmRestoreField(radsearch, "identifier", NULL, NULL, (void**)&identifier);
    DMSwarmRestoreField(radsearch, "virtual coord", NULL, NULL, (void**)&virtualcoord);

    if (log) {
        PetscPrintf(MPI_COMM_WORLD, "Particles Initialized\n");
    }

    /** ***********************************************************************************************************************************************
     * Now that the particles have been created, they can be iterated over and each marched one step in space. The global indices of the local
     * ray segment storage can be easily accessed and appended. This forms a local collection of globally index ray segments.
     * */
    DMSwarmGetLocalSize(radsearch, &npoints);  //!< Recalculate the number of particles that are in the domain
    while (npoints != 0) {                     //!< WHILE THERE ARE PARTICLES IN THE DOMAIN
        PetscInt stepcount = 0;
        /** Get all of the ray information from the particle
         * Get the ntheta and nphi from the particle that is currently being looked at. This will be used to identify its ray and calculate its direction. */
        DMSwarmGetField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void**)&coord);
        DMSwarmGetField(radsearch, "identifier", NULL, NULL, (void**)&identifier);
        DMSwarmGetField(radsearch, "virtual coord", NULL, NULL, (void**)&virtualcoord);
        /** Iterate over the particles that are present in the domain
         * Add the cell index to the ray
         * Step every particle in the domain one step and then perform a migration
         * */
        for (int ipart = 0; ipart < npoints; ipart++) {  //!< Iterate over the particles present in the domain. How to isolate the particles in this domain and iterate over them? If there are no
                                                         //!< particles then pass out of initialization.
            /** Should represent the distance from the origin cell to the boundary. How to get this? By marching out of the domain! (and checking whether we are still inside)
             * Number of spatial steps that  the ray has taken towards the origin
             * Keeps track of whether the ray has intersected the boundary at this point or not
             */
            //    PetscReal magnitude = hstep; This will no loger apply if we simply += the particle coordinate on every pass
            int nsteps = 0;  //        bool boundary = false;  //!< I don't even think we need to check for the boundary anymore.

            /** Insert zeros into the Ij1 initialization so that the solver has an initial assumption of 0 to work with.
             * Put as many zeros as there are domains so that there are matching indices
             * Domain split every x points
             * */
            PetscReal initialValue = 0.0;  //!< This is the intensity being given to the initial values of the rays
            PetscReal anotherInitialValue = 1;
            //                    std::vector<PetscInt> rayDomain;

            /** Update the physical coordinate field so that the real particle location can be updated. */
            for (int d = 0; d < dim; d++) {                            //!< For the number of dimensions that actually exist physically.
                coord[dim * ipart + d] = virtualcoord[3 * ipart + d];  //!< Insert the virtual coordinates into the physical coordinate field.
            }

            /** Label the particle with the ray identifier. (Use an array of 5 ints, [origin][ncell][theta][phi][domains crossed])
             * Label the particle with domainscrossed = 0; so that this can be iterated after each domain cross.
             * */
            PetscInt origin = identifier[5 * ipart];       //!< Get the origin rank of the ray from the particle identifier.
            PetscInt ncell = identifier[5 * ipart + 1];    ///!< Get the cell index of the ray from the particle identifier.
            PetscInt ntheta = identifier[5 * ipart + 2];   //!< Get the angle number from the particle field label for the rays vector.
            PetscInt nphi = identifier[5 * ipart + 3];     //!< Input the ray identifier
            PetscInt ndomain = identifier[5 * ipart + 4];  //!< How many domains has the particle passed through?

            /** Resize the rays vectors */
            while (static_cast<int>(rays[origin][ncell][ntheta][nphi].size()) <= ndomain) {
                Ij1[origin][ncell][ntheta][nphi].push_back(initialValue);          //!< The initial value to input for Ij1 should be 0 for the number of domains that the ray crosses
                Krad[origin][ncell][ntheta][nphi].push_back(anotherInitialValue);  //!< This needs to be stored locally and not transported
                rays[origin][ncell][ntheta][nphi].push_back(rayDomains);           //!< This needs to be stored locally and not transported
                h[origin][ncell][ntheta][nphi].push_back(rayDomains);              //!< This needs to be stored locally and not transported
            }

            // TODO: IF THIS RAYS VECTOR IS EMPTY FOR THIS DOMAIN, THEN THE PARTICLE HAS NEVER BEEN HERE BEFORE. THEREFORE, ITERATE THE NDOMAINS BY 1.
            //  This does not account for if the particle has passed out of and then back into the domain. If there is a way to check which particles are here that were not here last time step, that
            //  would be more convenient.
            if (static_cast<int>(rays[origin][ncell][ntheta][nphi][ndomain].size()) == 0) {
                identifier[4 * ipart + 4]++;  //!< The particle has passed through another domain.
            }

            PetscReal hhere = hstep;  //!< Represents the space step of the current cell
            //            PetscInt currentCell = -1;  //!< Represents the cell that the ray is currently in. If the cell that the ray is in changes, then this cell should be added to the ray. Also,
            //            the space
            //                                        //!< step that was taken between cells should be saved.

            nsteps = 0;  //!< Reset the number of steps that the domain contains, moving on to a new domain

            /** FIRST TAKE THIS LOCATION INTO THE RAYS VECTOR */

            /** "I found a particle in my domain. Maybe it was just moved here and I've never seen it before.
             * Therefore, my first step should be to add this location to the local rays vector. Then I can adjust the coordinates and migrate the particle." */
            Vec intersect;
            PetscInt i[3] = {0, 1, 2};
            /** Get the particle coordinates here and put them into the intersect */
            PetscReal direction[3] = {(virtualcoord[3 * ipart]),       // x component conversion from spherical coordinates, adding the position of the current cell
                                      (virtualcoord[3 * ipart + 1]),   // y component conversion from spherical coordinates, adding the position of the current cell
                                      (virtualcoord[3 * ipart + 2])};  // z component conversion from spherical coordinates, adding the position of the current cell
            //(Reference for coordinate transformation: Rad. Heat Transf. Modest pg. 11) Create a direction vector in the current angle direction

            /** This block creates the vector pointing to the cell whose index will be stored during the current loop*/
            VecCreate(PETSC_COMM_WORLD, &intersect);  //!< Instantiates the vector
            VecSetBlockSize(intersect, dim);
            VecSetSizes(intersect, PETSC_DECIDE, dim);  //!< Set size
            VecSetFromOptions(intersect);
            VecSetValues(intersect, dim, i, direction, INSERT_VALUES);  //!< Actually input the values of the vector (There are 'dim' values to input)
            /** Loop through points to try to get the cell that is sitting on that point*/
            PetscSF cellSF = nullptr;  //!< PETSc object for setting up and managing the communication of certain entries of arrays and Vecs between MPI processes.
            DMLocatePoints(cellDM, intersect, DM_POINTLOCATION_NONE, &cellSF);  //!< Locate the points in v in the mesh and return a PetscSF of the containing cells

            /** An array that maps each point to its containing cell can be obtained with the below
             * We want to get a PetscInt index out of the DMLocatePoints function (cell[n].index)
             * */
            PetscInt nFound;
            const PetscInt* point = nullptr;
            const PetscSFNode* cell = nullptr;
            PetscSFGetGraph(cellSF, nullptr, &nFound, &point, &cell);  //!< Using this to get the petsc int cell number from the struct (SF)

            /** IF THE CELL NUMBER IS RETURNED NEGATIVE, THEN WE HAVE REACHED THE BOUNDARY OF THE DOMAIN >> This exits the loop
             * This function returns multiple values if multiple points are input to it
             * Make sure that whatever cell is returned is in the stencil set (and not outside of the radiation domain)
             * Assemble a vector of vectors etc associated with each cell index, angular coordinate, and space step?
             * The boundary has been reached if any of these conditions don't hold
             * */
            if (nFound > -1 && cell[0].index >= 0 && subDomain->InRegion(cell[0].index)) {
                rays[origin][ncell][ntheta][nphi][ndomain].push_back(cell[0].index);
                h[origin][ncell][ntheta][nphi][ndomain].push_back(hhere);
            }

            /** Step the vector forward in space until it is no longer in the cell it was ins
             * After the coordinates have left the cell it was it, the coordinates of the particle should be updated
             * */
            //            PetscInt currentCell = cell[0].index;  //!< Represents the cell that the ray is currently in.
            //            while (currentCell == cell[0].index) {
            theta = ((double)ntheta / (double)nTheta) * pi;
            phi = ((double)nphi / (double)nPhi) * 2.0 * pi;
            direction[0] += (hstep * sin(theta) * cos(phi));  //!< x component conversion from spherical coordinates, adding the position of the current cell
            direction[1] += (hstep * sin(theta) * sin(phi));  //!< y component conversion from spherical coordinates, adding the position of the current cell
            direction[2] += (hstep * cos(theta));             //!< z component conversion from spherical coordinates, adding the position of the current cell
            //(Reference for coordinate transformation: Rad. Heat Transf. Modest pg. 11) Create a direction vector in the current angle direction
            //                VecSetValues(intersect, dim, i, direction, INSERT_VALUES);          //!< Actually input the values of the vector (There are 'dim' values to input)
            //                DMLocatePoints(cellDM, intersect, DM_POINTLOCATION_NONE, &cellSF);  //!< Locate the points in v in the mesh and return a PetscSF of the containing cells
            //                PetscSFGetGraph(cellSF, nullptr, &nFound, &point, &cell);           //!< Using this to get the petsc int cell number from the struct (SF)
            //                if (nFound > -1 && cell[0].index >= 0 && subDomain->InRegion(cell[0].index)) {
            //                } else {
            //                    currentCell = -1;
            //                }
            //            }

            /** Update the coordinates of the particle (virtual and physical) */
            virtualcoord[3 * ipart] = direction[0];
            virtualcoord[3 * ipart + 1] = direction[1];
            virtualcoord[3 * ipart + 2] = direction[2];
            for (int d = 0; d < dim; d++) {                            //!< For the number of dimensions that actually exist physically.
                coord[dim * ipart + d] = virtualcoord[3 * ipart + d];  //!< Insert the virtual coordinates into the physical coordinate field.
            }

            nsteps++;
            /** Cleanup*/
            PetscSFDestroy(&cellSF);
            VecDestroy(&intersect);

            /** This protects the last domain from becoming empty, which would prevent the boundary initialization of the black body intensity */
            if (static_cast<int>(rays[origin][ncell][ntheta][nphi][ndomain].size()) == 0) {  //!< If there are no points stored in this domain when the ray has hit the boundary
                rays[origin][rank][ncell][ntheta][nphi].pop_back();                          //!< Remove the last domain entry in the rays vector
                Ij1[origin][ncell][ntheta][nphi].pop_back();                                 //!< There will no longer be a ray segment here to keep track of
            }
        }
        /** Restore the fields associated with the particles after all of the particles have been stepped s*/
        DMSwarmRestoreField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void**)&coord);
        DMSwarmRestoreField(radsearch, "identifier", NULL, NULL, (void**)&identifier);
        DMSwarmRestoreField(radsearch, "virtual coord", NULL, NULL, (void**)&virtualcoord);

        /** DMSwarm Migrate to move the ray search particle into the next domain if it has crossed. If it no longer appears in this domain then end the ray segment. */
        DMSwarmMigrate(radsearch, PETSC_TRUE);     //!< Migrate the search particles and remove the particles that have left the domain space
        DMSwarmGetLocalSize(radsearch, &npoints);  //!< Recalculate the number of particles that are in the domain

        if (log) {
            PetscPrintf(MPI_COMM_WORLD, "Global Step %3i\n", stepcount);
            stepcount++;
        }
    }
    /** Cleanup*/
    Ij = Ij1;
    Izeros = Ij1;
    Kones = Krad;
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray);
    RestoreRange(cellRange);

    // TODO: When all of the particles have disappeared, each domain will need to create a particle field for each ray segment located at the head of the ray segment and tagged with the origin rank.
    //  Then the particle field will already exist for what the solve uses.
}

PetscErrorCode ablate::radiation::Radiation::ComputeRHSFunction(PetscReal time, Vec solVec, Vec rhs) {
    PetscFunctionBeginUser;
    /** Abstract PETSc object that manages an abstract grid object and its interactions with the algebraic solvers
     * These DM objects belong to the temperature field which is used to calculate radiation transport
     */
    DM vdm;
    Vec loctemp;
    IS vis;

    /** Get the array of the local f vector, put the intensity into part of that array instead of using the radiative gain variable*/
    const PetscScalar* rhsArray;
    VecGetArrayRead(rhs, &rhsArray);

    /** Get the array of the solution vector*/
    const PetscScalar* solArray;
    VecGetArrayRead(solVec, &solArray);

    const auto& eulerFieldInfo = subDomain->GetField("euler");
    auto dm = subDomain->GetDM();  //!< Get the main DM for the solution vector

    /** Get the temperature field
     * For ABLATE implementation, get temperature based on this function
     */
    const auto& temperatureField = subDomain->GetField("temperature");
    PetscScalar* temperatureArray = nullptr;
    subDomain->GetFieldLocalVector(temperatureField, time, &vis, &loctemp, &vdm);
    VecGetArray(loctemp, &temperatureArray);

    /** Declare the basic information*/
    PetscReal* sol;          //!< The solution value at any given location
    PetscReal* temperature;  //!< The temperature at any given location
    PetscReal dTheta = pi / nTheta;
    PetscReal dPhi = (2 * pi) / nPhi;
    double kappa = 1;  //!< Absorptivity coefficient, property of each cell
    double theta;

    std::vector<std::vector<PetscReal>> locations;  //!< 2 Dimensional vector which stores the locations of the cell centers
    PetscInt ncells = 0;

    auto absorptivityFunctionContext = absorptivityFunction.context.get();

    /** loop through subdomain cell indices
     * for every angle theta
     * converts the present angle number into a real angle
     * */
    solver::Range cellRange;
    GetCellRange(cellRange);

    // TODO: Loop over all particles instead of all ray identifiers. Grab ray information from the identifier.
    //  The solving particle field will need to include

    /** Loop over all of the cells in the domain */
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;
        PetscReal intensity = 0;
        for (int ntheta = 1; ntheta < nTheta; ntheta++) {    // for every angle theta
            theta = ((double)ntheta / (double)nTheta) * pi;  // converts the present angle number into a real angle
            for (int nphi = 0; nphi < nPhi; nphi++) {
                /** Each ray is born here. They begin at the far field temperature.
                    Initial ray intensity should be set based on which boundary it is coming from.
                    If the ray originates from the walls, then set the initial ray intensity to the wall temperature, etc.
                 */
                int nDomain = static_cast<int>(rays[0][ncells][ntheta][nphi].size());
                PetscReal rayIntensity = 0.0;

                PetscReal I0 = 0;

                /** Local ray computation happens here */
                for (int ndomain = (nDomain - 1); ndomain >= 0; ndomain--) {
                    /** For each domain in the ray (The rays vector will have an added index, splitting every x points) */
                    int numPoints = static_cast<int>(rays[0][ncells][ntheta][nphi][ndomain].size());
                    rayIntensity = 0;

                    if (numPoints > 0) {
                        for (int n = 0; n < (numPoints); n++) {
                            /** Go through every cell point that is stored within the ray >> FROM THE BOUNDARY TO THE SOURCE
                                Define the absorptivity and temperature in this section
                                For ABLATE implementation, get temperature based on this function
                                Get the array that lives inside the vector
                                Gets the temperature from the cell index specified
                            */
                            DMPlexPointLocalRead(vdm, rays[0][ncells][ntheta][nphi][ndomain][n], temperatureArray, &temperature);
                            DMPlexPointLocalRead(dm, rays[0][ncells][ntheta][nphi][ndomain][n], solArray, &sol);
                            /** Input absorptivity (kappa) values from model here. */
                            absorptivityFunction.function(sol, *temperature, &kappa, absorptivityFunctionContext);

                            Ij[0][ncells][ntheta][nphi][ndomain] += FlameIntensity(1 - exp(-kappa * h[0][ncells][ntheta][nphi][ndomain][n]), *temperature) * Krad[0][ncells][ntheta][nphi][ndomain];
                            Krad[0][ncells][ntheta][nphi][ndomain] *= exp(-kappa * h[0][ncells][ntheta][nphi][ndomain][n]);  //!< Compute the total absorption for this domain

                            if (n == (numPoints - 1) && ndomain == (nDomain - 1)) { /** If this is the beginning of the ray, set this as the initial intensity. */
                                I0 = FlameIntensity(1, *temperature);               //!< Set the initial intensity of the ray for the linearized assumption
                            }
                        }
                    }
                }

                // TODO: After iterating through all of the particles, perform a migration to the origin ranks. This will move the particles.

                // TODO: Now iterate through all of the particles again

                /** Global ray computation happens here, grabbing values from the transported particles
                 * The rays end here, their intensity is added to the total intensity of the cell
                 * Gives the partial impact of the ray on the total sphere.
                 * The sin(theta) is a result of the polar coordinate discretization
                 *
                 * If the linearization is being used, the domain intensity calculation will be used
                 * If not, the ray intensity will be taken directly from the end of the ray
                 *
                 * In the parallel form at the end of each ray, the absorption of the initial ray and the absorption of the black body source are computed individually at the end.
                 * */
                /** Parallel things are here
                 * Meaning that the variables required for the parallelizable analytical solution will be declared here */
                PetscReal Kradd = 1;
                PetscReal Isource = 0;
                for (int ndomain = 0; ndomain < nDomain; ndomain++) {
                    // TODO: Add ray identifier index to these variables so that they can be stored while the particles get looped through
                    Isource += Ij[0][ncells][ntheta][nphi][ndomain] * Kradd;  //!< Add the black body radiation transmitted through the domain to the source term
                    Kradd *= Krad[0][ncells][ntheta][nphi][ndomain];          //!< Add the absorption for this domain to the total absorption of the ray
                }
                rayIntensity = (I0 * Kradd) + Isource;                   //!< This variable doesn't have any reason for existing.
                intensity += rayIntensity * sin(theta) * dTheta * dPhi;  // TODO: Add ncells index to intensity so that each cell intensity can be stored.
            }
        }

        /** Provide progress updates if the log is enabled.
         * Radiative gain can be useful for debugging purposes.
         * Error calculation is done here.
         */
        if (log) {
            //        DMPlexPointLocalRead(cellDM, iCell, cellGeomArray, &cellGeom);  //!< Reads the cell location from the current cell
            //                        log->Printf("%g ", cellGeom->centroid[dim - 1]);                //!< Print the height location of the cell for which the intensity is being printed
            //                        log->Printf("%g\n", intensity);  //!< Print the intensity of this cell. This will be compared to the analytical solution.
        }

        /** Gets the temperature from the cell index specified */
        DMPlexPointLocalRef(vdm, iCell, temperatureArray, &temperature);

        /** Put the irradiation into the right hand side function */
        PetscScalar* rhsValues;
        DMPlexPointLocalFieldRead(subDomain->GetDM(), iCell, eulerFieldInfo.id, rhsArray, &rhsValues);
        PetscReal losses = 4 * sbc * *temperature * *temperature * *temperature * *temperature;
        rhsValues[ablate::finiteVolume::CompressibleFlowFields::RHOE] += -kappa * (losses - intensity);

        /** Add to the number of cells that has been counted */
        ncells++;
    }
    /** Cleanup*/
    Ij = Izeros;
    Krad = Kones;
    VecRestoreArrayRead(rhs, &rhsArray);
    VecRestoreArray(loctemp, &temperatureArray);
    subDomain->RestoreFieldLocalVector(temperatureField, &vis, &loctemp, &vdm);
    RestoreRange(cellRange);

    PetscFunctionReturn(0);
}

PetscReal ablate::radiation::Radiation::FlameIntensity(double epsilon, double temperature) { /** Gets the flame intensity based on temperature and emissivity*/
    const PetscReal sbc = 5.6696e-8;                                                         //!< Stefan-Boltzman Constant (J/K)
    const PetscReal pi = 3.1415926535897932384626433832795028841971693993;
    return epsilon * sbc * temperature * temperature * temperature * temperature / pi;
}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::radiation::Radiation, "A solver for radiative heat transfer in participating media", ARG(std::string, "id", "the name of the flow field"),
         ARG(ablate::domain::Region, "region", "the region to apply this solver."), ARG(int, "rays", "number of rays used by the solver"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"),
         ARG(ablate::eos::radiationProperties::RadiationModel, "properties", "the radiation properties model"), OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));