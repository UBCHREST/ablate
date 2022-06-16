#include "radiation.hpp"
#include <set>
#include <utility>
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "particles/particles.hpp"
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
     * */

    // TODO: Setup the particles and their associated fields including: ray identifier, # domains crossed, coordinates, and origin domain. Instantiate ray particles for each local cell only.

    /** Instantiate a vector of fields which can store the information required in the search particles? */
    std::vector<ParticleField> fields;

    /** Create a DM that is associated with the radiation particle field
     * This is the feild that will be associated with the particle search
     * */
    DMCreate(PETSC_COMM_WORLD, &radDM) >> checkError;
    DMSetType(radDM, DMSWARM) >> checkError;
    DMSetDimension(radDM, dim) >> checkError;
    DMSwarmSetType(radDM, DMSWARM_PIC) >> checkError;
    std::vector<std::string> coordComponents;  //!< Record the defaul fields.
    coordComponents = {"X", "Y", "Z"};         //!< The number of coordinate components will always be three because of the nature of the radiation solver.

    auto coordField = ParticleField{.name = DMSwarmPICField_coor, .components = coordComponents, .type = domain::FieldLocation::SOL, .dataType = PETSC_REAL};
    particleFieldDescriptors.push_back(coordField);
    particleFieldDescriptors.emplace_back(ParticleField{.name = DMSwarmField_pid, .type = domain::FieldLocation::AUX, .dataType = PETSC_INT64});

    for (auto& field : fields) {
        RegisterParticleField(field);  //!< register each field
    }

    /** Declare some local variables */
    double theta;  //!< represents the actual current angle (inclination)
    double phi;    //!< represents the actual current angle (rotation)

    /**Locally get a range of cells that are included in this subdomain at this time step for the ray initialization
     * */
    solver::Range cellRange;
    GetCellRange(cellRange);

    // TODO: This is still relevant, as there will be global indexing of these values but with local information only.

    /** Create a nested vector which can store cell locations based on origin cell, theta, phi, and space step
     * Preallocate the sub-vectors in order to avoid dynamic sizing as much as possible
     * Make vector to store this dimensional row
     * Indices: Cell, angle (theta), angle(phi), domain, space steps
     * */
    std::vector<PetscInt> rayDomains;
    std::vector<std::vector<PetscInt>> rayPhis;
    std::vector<std::vector<std::vector<PetscInt>>> rayThetas(nPhi, rayPhis);
    std::vector<std::vector<std::vector<std::vector<PetscInt>>>> rayCells(nTheta, rayThetas);
    rays.resize((cellRange.end - cellRange.start), rayCells);
    h.resize((cellRange.end - cellRange.start), rayCells);  //!< Store a vector of space steps that the solver will use to compute absorption effects

    std::vector<PetscReal> Ij1Phis;
    std::vector<std::vector<PetscReal>> Ij1Thetas(nPhi, Ij1Phis);
    std::vector<std::vector<std::vector<PetscReal>>> Ij1Cells(nTheta, Ij1Thetas);
    Ij1.resize((cellRange.end - cellRange.start), Ij1Cells);  //!< This sets the previous iteration intensity so that each ray can store multiple intensities.

    Krad.resize((cellRange.end - cellRange.start), Ij1Cells);

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

    /** Set the spatial step size to the minimum cell radius */
    PetscReal hstep = minCellRadius;

    PetscInt originrank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &originrank);  //!< Get the origin rank of the current process. The particle belongs to this rank.

    PetscInt ncells = 0;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {  // TODO: Only for cells within my rank domain. How to only iterate over local cells? Do we need to only iterate over local cells?
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;  //!< Isolates the valid cells
        DMPlexPointLocalRead(cellDM, iCell, cellGeomArray, &cellGeom);      //!< Reads the cell location from the current cell

        /** for every angle theta
         * for every angle phi
         */
        for (int ntheta = 1; ntheta < nTheta; ntheta++) {
            for (int nphi = 0; nphi < nPhi; nphi++) {
                // TODO: Create a particle for every new ray at this coordinate with this direction, putting in the origin rank, ray id, domains crossed, etc.
                // Get the particle coordinate field and write the cellGeom->centroid[xyz] into it

                // TODO: Label the particle with the originrank so it knows where to get sent to

                // TODO: Label the particle with domainscrossed = 0; so that this can be iterated. How to detect when a particle has crossed a domain?

                // TODO: Label the particle with the ray identifier. (Use an array of 3 ints, [ncell][theta][phi])
            }
        }
        ncells++;  //!< Increase the number of cells that have been finished.
    }

    /** ***********************************************************************************************************************************************
     * Now that the particles have been created, they can be iterated over and each marched one step in space. The global indices of the local
     * ray segment storage can be easily accessed and appended.
     * */

    /** Iterate over the particles that are present in the domain
     * Add the cell index to the ray
     * Step every particle in the domain one step and then perform a migration
     * */
    // TODO: Iterate over the particles present in the domain. How to isolate the particles in this domain and iterate over them? If there are no particles then pass out of initialization.

    /** Should represent the distance from the origin cell to the boundary. How to get this? By marching out of the domain! (and checking whether we are still inside)
     * Number of spatial steps that  the ray has taken towards the origin
     * Keeps track of whether the ray has intersected the boundary at this point or not
     */
//    PetscReal magnitude = hstep; This will no loger apply if we simply += the particle coordinate on every pass
    int nsteps = 0;
    int ndomain = 0;        //!< Number of domains that have been crossed by the particle.
    bool boundary = false;  //!< I don't even think we need to check for the boundary anymore.

    /** Insert zeros into the Ij1 initialization so that the solver has an initial assumption of 0 to work with.
     * Put as many zeros as there are domains so that there are matching indices
     * Domain split every x points
     * */
    PetscReal initialValue = 0.0;  //!< This is the intensity being given to the initial values of the rays
    PetscReal anotherInitialValue = 1;
    //                    std::vector<PetscInt> rayDomain;

    // TODO: Get the ntheta and nphi from the particle that is currently being looked at. This will be used to identify its ray and calculate its direction.
    PetscInt ntheta = 1;  //!< Get the angle number from the particle field label for the rays vector. Resize the rays vector if necessary for the global indexing scheme.
    PetscInt nphi = 1;    //!< Get the angle number from the particle field label for the rays vector

    if (rays[ncells][ntheta][nphi].size() < ndomain) {
        Ij1[ncells][ntheta][nphi].push_back(initialValue);          //!< The initial value to input for Ij1 should be 0 for the number of domains that the ray crosses
        Krad[ncells][ntheta][nphi].push_back(anotherInitialValue);  //!< This needs to be stored locally and not transported
        rays[ncells][ntheta][nphi].push_back(rayDomains);           //!< This needs to be stored locally and not transported
        h[ncells][ntheta][nphi].push_back(rayDomains);              //!< This needs to be stored locally and not transported
    }

    // TODO: IF THIS RAYS VECTOR IS EMPTY FOR THIS DOMAIN, THEN THE PARTICLE HAS NEVER BEEN HERE BEFORE. THEREFORE, ITERATE THE NDOMAINS BY 1.
    //  This does not account for if the particle has passed out of and then back into the domain. If there is a way to check which particles are here that were not here last time step, that would be
    //  more convenient.

    PetscReal hhere = 0;       //!< Represents the space step of the current cell
    PetscInt currentCell = 0;  //!< Represents the cell that the ray is currently in. If the cell that the ray is in changes, then this cell should be added to the ray. Also, the space
                               //!< step that was taken between cells should be saved.

    nsteps = 0;  //!< Reset the number of steps that the domain contains, moving on to a new domain

    // TODO: FIRST TAKE THIS LOCATION INTO THE RAYS VECTOR

    /** "I found a particle in my domain. Maybe it was just moved here and I've never seen it before.
     * Therefore, my first step should be to add this location to the local rays vector. Then I can adjust the coordinates and migrate the particle." */
    // TODO: The vector previously calculated as intersect should be replaced by the particle coordinates.
    // TODO: Get the particle coordinates here and put them into the intersect

    /** Loop through points to try to get the cell that is sitting on that point*/
    PetscSF cellSF = nullptr;  //!< PETSc object for setting up and managing the communication of certain entries of arrays and Vecs between MPI processes.
    DMLocatePoints(cellDM, intersect, DM_POINTLOCATION_NONE, &cellSF) >> checkError;  //!< Locate the points in v in the mesh and return a PetscSF of the containing cells

    /** An array that maps each point to its containing cell can be obtained with the below
     * We want to get a PetscInt index out of the DMLocatePoints function (cell[n].index)
     * */
    PetscInt nFound;
    const PetscInt* point = nullptr;
    const PetscSFNode* cell = nullptr;
    PetscSFGetGraph(cellSF, nullptr, &nFound, &point, &cell) >> checkError;  //!< Using this to get the petsc int cell number from the struct (SF)

    /** IF THE CELL NUMBER IS RETURNED NEGATIVE, THEN WE HAVE REACHED THE BOUNDARY OF THE DOMAIN >> This exits the loop
     * This function returns multiple values if multiple points are input to it
     * Make sure that whatever cell is returned is in the stencil set (and not outside of the radiation domain)
     * Assemble a vector of vectors etc associated with each cell index, angular coordinate, and space step?
     * The boundary has been reached if any of these conditions don't hold
     * */
    if (nFound > -1 && cell[0].index >= 0 && subDomain->InRegion(cell[0].index)) {
        if (currentCell != cell[0].index) {
            rays[ncells][ntheta][nphi][ndomain].push_back(cell[0].index);
            h[ncells][ntheta][nphi][ndomain].push_back(hhere);
            hhere = 0; //!< Reset the adaptive step checker
            currentCell = cell[0].index; //!< Reset the adaptive step checker
        } else {
            hhere += hstep;
        }
    } else {
        boundary = true;  //!< The boundary has been reached if any of these conditions don't hold
    }

    // TODO: Step the particle in space: Set the coordinates of the particle += to the step direction

    /** Draw a point on which to grab the cells of the DM
     * Intersect should point to the boundary, and then be pushed back to the origin, getting each cell
     * converts the present angle number into a real angle
     * */
    Vec intersect;
    theta = ((double)ntheta / (double)nTheta) * pi;
    phi = ((double)nphi / (double)nPhi) * 2.0 * pi;
    PetscInt i[3] = {0, 1, 2};

    /** x component conversion from spherical coordinates, adding the position of the current cell
     * y component conversion from spherical coordinates, adding the position of the current cell
     * z component conversion from spherical coordinates, adding the position of the current cell
     * (Reference for coordinate transformation: Rad. Heat Transf. Modest pg. 11) Create a direction vector in the current angle direction
     * */
    PetscReal direction[3] = {(hstep * sin(theta) * cos(phi)) + cellGeom->centroid[0],  // x component conversion from spherical coordinates, adding the position of the current cell
                              (hstep * sin(theta) * sin(phi)) + cellGeom->centroid[1],  // y component conversion from spherical coordinates, adding the position of the current cell
                              (hstep * cos(theta)) + cellGeom->centroid[2]};            // z component conversion from spherical coordinates, adding the position of the current cell
    //(Reference for coordinate transformation: Rad. Heat Transf. Modest pg. 11) Create a direction vector in the current angle direction

    /** This block creates the vector pointing to the cell whose index will be stored during the current loop*/
    VecCreate(PETSC_COMM_WORLD, &intersect);  //!< Instantiates the vector
    VecSetBlockSize(intersect, dim) >> checkError;
    VecSetSizes(intersect, PETSC_DECIDE, dim);  //!< Set size
    VecSetFromOptions(intersect);
    VecSetValues(intersect, dim, i, direction, INSERT_VALUES);  //!< Actually input the values of the vector (There are 'dim' values to input)

    /** Increase the step size of the ray toward the boundary by one more minimum cell radius
     * Increase the magnitude of the ray tracing vector by one space step
     * Increase the number of steps taken, informs how many indices the vector has
     * */
//    magnitude += hstep; This won't matter if we += the coordinate on every pass
    nsteps++;

    // TODO: DMSwarm Migrate to move the ray search particle into the next domain if it has crossed. If it no longer appears in this domain then end the ray segment

    /** Cleanup*/
    PetscSFDestroy(&cellSF);
    VecDestroy(&intersect);

    /** This protects the last domain from becoming empty, which would prevent the boundary initialization of the black body intensity */
    if (static_cast<int>(rays[ncells][ntheta][nphi][ndomain].size()) == 0) {  //!< If there are no points stored in this domain when the ray has hit the boundary
        rays[ncells][ntheta][nphi].pop_back();                                //!< Remove the last domain entry in the rays vector
        Ij1[ncells][ntheta][nphi].pop_back();                                 //!< There will no longer be a ray segment here to keep track of
    }
    ndomain++;
    /** Cleanup*/
    Ij = Ij1;
    Izeros = Ij1;
    Kones = Krad;
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray);
    RestoreRange(cellRange);
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

    //TODO: Loop over all particles instead of all ray identifiers. Grab ray information from the identifier.
    // The solving particle field will need to include

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
                int nDomain = static_cast<int>(rays[ncells][ntheta][nphi].size());
                PetscReal rayIntensity = 0.0;

                PetscReal I0 = 0;

                /** Local ray computation happens here */
                for (int ndomain = (nDomain - 1); ndomain >= 0; ndomain--) {
                    /** For each domain in the ray (The rays vector will have an added index, splitting every x points) */
                    int numPoints = static_cast<int>(rays[ncells][ntheta][nphi][ndomain].size());
                    rayIntensity = 0;

                    if (numPoints > 0) {
                        for (int n = 0; n < (numPoints); n++) {
                            /** Go through every cell point that is stored within the ray >> FROM THE BOUNDARY TO THE SOURCE
                                Define the absorptivity and temperature in this section
                                For ABLATE implementation, get temperature based on this function
                                Get the array that lives inside the vector
                                Gets the temperature from the cell index specified
                            */
                            DMPlexPointLocalRead(vdm, rays[ncells][ntheta][nphi][ndomain][n], temperatureArray, &temperature);
                            DMPlexPointLocalRead(dm, rays[ncells][ntheta][nphi][ndomain][n], solArray, &sol);
                            /** Input absorptivity (kappa) values from model here. */
                            absorptivityFunction.function(sol, *temperature, &kappa, absorptivityFunctionContext);

                            Ij[ncells][ntheta][nphi][ndomain] += FlameIntensity(1 - exp(-kappa * h[ncells][ntheta][nphi][ndomain][n]), *temperature) * Krad[ncells][ntheta][nphi][ndomain];
                            Krad[ncells][ntheta][nphi][ndomain] *= exp(-kappa * h[ncells][ntheta][nphi][ndomain][n]);  //!< Compute the total absorption for this domain

                            if (n == (numPoints - 1) && ndomain == (nDomain - 1)) { /** If this is the beginning of the ray, set this as the initial intensity. */
                                I0 = FlameIntensity(1, *temperature);               //!< Set the initial intensity of the ray for the linearized assumption
                            }
                        }
                    }
                }

                // TODO: After iterating through all of the particles, perform a migration to the origin ranks. This will move the particles.

                //TODO: Now iterate through all of the particles again

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
                    Isource += Ij[ncells][ntheta][nphi][ndomain] * Kradd;  //!< Add the black body radiation transmitted through the domain to the source term
                    Kradd *= Krad[ncells][ntheta][nphi][ndomain];          //!< Add the absorption for this domain to the total absorption of the ray
                }
                rayIntensity = (I0 * Kradd) + Isource; //!< This variable doesn't have any reason for existing.
                intensity += rayIntensity * sin(theta) * dTheta * dPhi; //TODO: Add ncells index to intensity so that each cell intensity can be stored.
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