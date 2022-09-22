#include "radiation.hpp"
#include <petsc/private/dmimpl.h>
#include <petscdm.h>
#include <petscdmswarm.h>
#include <petscsf.h>
#include <utility>
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"

ablate::radiation::Radiation::Radiation(const std::string& solverId, const std::shared_ptr<domain::Region>& region, std::shared_ptr<domain::Region> fieldBoundary, const PetscInt raynumber,
                                        std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, std::shared_ptr<ablate::monitors::logs::Log> log)
    : solverId((std::basic_string<char> &&) solverId), region(region), radiationModel(std::move(radiationModelIn)), fieldBoundary(std::move(fieldBoundary)), log(std::move(log)) {
    nTheta = raynumber;    //!< The number of angles to solve with, given by user input
    nPhi = 2 * raynumber;  //!< The number of angles to solve with, given by user input
}

ablate::radiation::Radiation::~Radiation() {
    if (radsolve) DMDestroy(&radsolve) >> checkError;  //!< Destroy the radiation particle swarm
    VecDestroy(&faceGeomVec) >> checkError;
    VecDestroy(&cellGeomVec) >> checkError;
}

/** allows initialization after the subdomain and dm is established */
void ablate::radiation::Radiation::Setup(const solver::Range& cellRange, ablate::domain::SubDomain& subDomain, bool surfaceIn) {
    surface = surfaceIn;
    dim = subDomain.GetDimensions();   //!< Number of dimensions already defined in the setup
    nTheta = (dim == 1) ? 1 : nTheta;  //!< Reduce the number of rays if one dimensional symmetry can be taken advantage of

    /** Begins radiation properties model
     * Runs the ray initialization, finding cell indices
     * Initialize the log if provided
     */
    absorptivityFunction = radiationModel->GetRadiationPropertiesTemperatureFunction(eos::radiationProperties::RadiationProperty::Absorptivity, subDomain.GetFields());

    if (log) {
        log->Initialize(subDomain.GetComm());
    }

    /** Initialization to call, draws each ray vector and gets all of the cells associated with it
     * (sorted by distance and starting at the boundary working in)
     * This is done by creating particles at the center of each cell and iterating through them
     * Get setup things for the position vector of the current cell index
     * Declare the variables that will contain the geometry of the cells
     * Obtain the geometric information about the cells in the DM
     * */

    if (log) StartEvent("Radiation Initialization");
    if (log) PetscPrintf(subDomain.GetComm(), "Starting Initialize\n");

    DMPlexGetMinRadius(subDomain.GetDM(), &minCellRadius) >> checkError;

    /** do a simple sanity check for labels */
    PetscMPIInt rank = 0;
    MPI_Comm_rank(subDomain.GetComm(), &rank);      //!< Get the origin rank of the current process. The particle belongs to this rank. The rank only needs to be read once.
    MPI_Comm_size(subDomain.GetComm(), &numRanks);  //!< Get the number of ranks in the simulation.

    /** Declare some local variables */
    double theta;  //!< represents the actual current angle (inclination)
    double phi;    //!< represents the actual current angle (rotation)

    /** Setup the particles and their associated fields including: origin domain/ ray identifier / # domains crossed, and coordinates. Instantiate ray particles for each local cell only. */
    PetscInt npoints = (cellRange.end - cellRange.start) * nTheta * nPhi;  //!< Number of points to insert into the particle field. One particle for each ray.

    /** Create the DMSwarm */
    DMCreate(subDomain.GetComm(), &radsearch) >> checkError;
    DMSetType(radsearch, DMSWARM) >> checkError;
    DMSetDimension(radsearch, dim) >> checkError;

    DMCreate(subDomain.GetComm(), &radsolve) >> checkError;
    DMSetType(radsolve, DMSWARM) >> checkError;
    DMSetDimension(radsolve, dim) >> checkError;

    /** Configure radsearch to be of type PIC/Basic */
    DMSwarmSetType(radsearch, DMSWARM_PIC) >> checkError;
    DMSwarmSetCellDM(radsearch, subDomain.GetDM()) >> checkError;

    DMSwarmSetType(radsolve, DMSWARM_BASIC) >> checkError;
    DMSwarmSetCellDM(radsolve, subDomain.GetDM()) >> checkError;

    /** Register fields within the DMSwarm */
    DMSwarmRegisterUserStructField(radsearch, "identifier", sizeof(Identifier)) >> checkError;  //!< A field to store the ray identifier [origin][iCell][ntheta][nphi][ndomain]
    DMSwarmRegisterUserStructField(radsearch, "virtual coord", sizeof(Virtualcoord)) >>
        checkError;                                         //!< A field representing the three dimensional coordinates of the particle. Three "virtual" dims are required.
    DMSwarmFinalizeFieldRegister(radsearch) >> checkError;  //!< Initialize the fields that have been defined

    DMSwarmRegisterUserStructField(radsolve, "identifier", sizeof(Identifier)) >> checkError;  //!< A field to store the ray identifier [origin][iCell][ntheta][nphi][ndomain]
    DMSwarmRegisterUserStructField(radsolve, "carrier", sizeof(Carrier)) >> checkError;        //!< A struct to carry information about the ray segment that the particle is communicating from
    DMSwarmFinalizeFieldRegister(radsolve) >> checkError;                                      //!< Initialize the fields that have been defined

    /** Set initial local sizes of the DMSwarm with a buffer length of zero */
    DMSwarmSetLocalSizes(radsearch, npoints, 0) >> checkError;  //!< Set the number of initial particles to the number of rays in the subdomain. Set the buffer size to zero.
    DMSwarmSetLocalSizes(radsolve, 0, 0) >> checkError;         //!< Set the number of initial particles to the number of rays in the subdomain. Set the buffer size to zero.

    /** Declare some information associated with the field declarations */
    PetscReal* coord;                   //!< Pointer to the coordinate field information
    struct Virtualcoord* virtualcoord;  //!< Pointer to the primary (virtual) coordinate field information
    struct Identifier* identifier;      //!< Pointer to the ray identifier information

    /** Get the fields associated with the particle swarm so that they can be modified */
    DMSwarmGetField(radsearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> checkError;
    DMSwarmGetField(radsearch, "identifier", nullptr, nullptr, (void**)&identifier) >> checkError;
    DMSwarmGetField(radsearch, "virtual coord", nullptr, nullptr, (void**)&virtualcoord) >> checkError;

    PetscInt ipart = 0;  //!< Initialize a counter to represent the particle index. This will be iterated every time that the inner loop is passed through.

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {            //!< This will iterate only though local cells
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;  //!< Isolates the valid cells
        PetscReal centroid[3];
        DMPlexComputeCellGeometryFVM(subDomain.GetDM(), iCell, nullptr, centroid, nullptr) >> checkError;

        /** for every angle theta
         * for every angle phi
         */
        for (PetscInt ntheta = 0; ntheta < nTheta; ntheta++) {
            for (PetscInt nphi = 0; nphi < nPhi; nphi++) {
                /** Get the initial direction of the search particle from the angle number that it was initialized with */
                theta = (((double)ntheta + 0.5) / (double)nTheta) * ablate::utilities::Constants::pi;  //!< Theta angle of the ray
                phi = ((double)nphi / (double)nPhi) * 2.0 * ablate::utilities::Constants::pi;          //!<  Phi angle of the ray

                /** Update the direction vector of the search particle */
                virtualcoord[ipart].xdir = (sin(theta) * cos(phi));  //!< x component conversion from spherical coordinates, adding the position of the current cell
                virtualcoord[ipart].ydir = (sin(theta) * sin(phi));  //!< y component conversion from spherical coordinates, adding the position of the current cell
                virtualcoord[ipart].zdir = (cos(theta));             //!< z component conversion from spherical coordinates, adding the position of the current cell

                /** Get the particle coordinate field and write the cellGeom->centroid[xyz] into it */
                virtualcoord[ipart].x = centroid[0] + (virtualcoord[ipart].xdir * 0.001 * minCellRadius);  //!< Offset from the centroid slightly so they sit in a cell if they are on its face.
                virtualcoord[ipart].y = centroid[1] + (virtualcoord[ipart].ydir * 0.001 * minCellRadius);
                virtualcoord[ipart].z = centroid[2] + (virtualcoord[ipart].zdir * 0.001 * minCellRadius);

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
    DMSwarmRestoreField(radsearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> checkError;
    DMSwarmRestoreField(radsearch, "identifier", nullptr, nullptr, (void**)&identifier) >> checkError;
    DMSwarmRestoreField(radsearch, "virtual coord", nullptr, nullptr, (void**)&virtualcoord) >> checkError;

    if (log) {
        PetscPrintf(subDomain.GetComm(), "Particles Setup\n");
    }

    if (surface) InitializationConvertSurface(subDomain);  //!< Convert to surface if this is a surface
}

void ablate::radiation::Radiation::InitializationConvertSurface(ablate::domain::SubDomain& subDomain) {
    /** Declare some information associated with the field declarations */
    PetscReal* coord;                   //!< Pointer to the coordinate field information
    struct Virtualcoord* virtualcoord;  //!< Pointer to the primary (virtual) coordinate field information
    struct Identifier* identifier;      //!< Pointer to the ray identifier information

    /** Get the fields associated with the particle swarm so that they can be modified */
    DMSwarmGetField(radsearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> checkError;
    DMSwarmGetField(radsearch, "identifier", nullptr, nullptr, (void**)&identifier) >> checkError;
    DMSwarmGetField(radsearch, "virtual coord", nullptr, nullptr, (void**)&virtualcoord) >> checkError;

    PetscInt npoints = 0;
    DMSwarmGetLocalSize(radsearch, &npoints) >> checkError;  //!< Recalculate the number of particles that are in the domain
    PetscMPIInt rank = 0;
    MPI_Comm_rank(subDomain.GetComm(), &rank);

    PetscInt numberNeighborCells;
    const PetscInt* neighborCells;

    /** Delete all of the particles that were transported to their origin domains -> Delete if the particle has travelled to get here and isn't native */
    for (PetscInt ipart = 0; ipart < npoints; ipart++) {
        DMPlexGetSupportSize(subDomain.GetDM(), identifier[ipart].iCell, &numberNeighborCells) >> ablate::checkError;  //!< Get the cells on each side of this face to check for boundary cells
        DMPlexGetSupport(subDomain.GetDM(), identifier[ipart].iCell, &neighborCells) >> ablate::checkError;
        PetscInt index = -1;  //!< Index value to compare the Locate Points result against.
        for (PetscInt n = 0; n < numberNeighborCells; n++) {
            PetscInt cell = neighborCells[n];  //!< Contains the cell indexes of the neighbor cells
            if (fieldBoundary->InRegion(fieldBoundary, subDomain.GetDM(), cell)) {
                index = cell;
            }
        }
        if (index != -1) {  //!< If there was no boundary cell adjacent to this particle's initial face, then this check is not worth doing in the first place
            //!< Locate the points so that the cell index of the particle can be compare with the boundary cell to see whether the particle should be deleted
            Vec intersect;
            VecCreate(PETSC_COMM_SELF, &intersect) >> checkError;  //!< Instantiates the vector
            VecSetBlockSize(intersect, dim) >> checkError;
            VecSetSizes(intersect, PETSC_DECIDE, npoints * dim) >> checkError;  //!< Set size
            VecSetFromOptions(intersect) >> checkError;
            PetscInt i[3] = {0, 1, 2};  //!< Establish the vector here so that it can be iterated.
            /** Get the particle coordinates here and put them into the intersect */
            PetscReal position[3] = {(coord[dim * ipart + 0]),   //!< x component conversion from spherical coordinates, adding the position of the current cell
                                     (coord[dim * ipart + 1]),   //!< y component conversion from spherical coordinates, adding the position of the current cell
                                     (coord[dim * ipart + 2])};  //!< z component conversion from spherical coordinates, adding the position of the current cell

            /** This block creates the vector pointing to the cell whose index will be stored during the current loop */
            VecSetValues(intersect, dim, i, position, INSERT_VALUES);  //!< Actually input the values of the vector (There are 'dim' values to input)
            i[0] += dim;                                               //!< Iterate the index by the number of dimensions so that the DMLocatePoints function can be called collectively.
            i[1] += dim;
            i[2] += dim;

            /** Loop through points to try to get the cell that is sitting on that point*/
            PetscSF cellSF = nullptr;  //!< PETSc object for setting up and managing the communication of certain entries of arrays and Vecs between MPI processes.
            DMLocatePoints(subDomain.GetDM(), intersect, DM_POINTLOCATION_NONE, &cellSF) >> checkError;  //!< Call DMLocatePoints here, all of the processes have to call it at once.

            /** An array that maps each point to its containing cell can be obtained with the below
             * We want to get a PetscInt index out of the DMLocatePoints function (cell[n].index)
             * */
            PetscInt nFound;
            const PetscInt* point = nullptr;
            const PetscSFNode* cell = nullptr;
            PetscSFGetGraph(cellSF, nullptr, &nFound, &point, &cell) >> checkError;  //!< Using this to get the petsc int cell number from the struct (SF)

            //!< If the particles that were just created are sitting in the boundary cell of the face that they belong to, delete them
            if (index == cell[0].index) {  //!< If the particle location index and boundary cell index are the same, then they should be deleted
                DMSwarmRestoreField(radsearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> checkError;
                DMSwarmRestoreField(radsearch, "identifier", nullptr, nullptr, (void**)&identifier) >> checkError;
                DMSwarmRestoreField(radsearch, "virtual coord", nullptr, nullptr, (void**)&virtualcoord) >> checkError;

                DMSwarmRemovePointAtIndex(radsearch, ipart);             //!< Delete the particle!
                DMSwarmGetLocalSize(radsearch, &npoints) >> checkError;  //!< Recalculate the number of particles that are in the domain

                DMSwarmGetField(radsearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> checkError;
                DMSwarmGetField(radsearch, "identifier", nullptr, nullptr, (void**)&identifier) >> checkError;
                DMSwarmGetField(radsearch, "virtual coord", nullptr, nullptr, (void**)&virtualcoord) >> checkError;
                ipart--;  //!< Check the point replacing the one that was deleted
            }
            VecDestroy(&intersect) >> checkError;   //!< Return the vector to PETSc
            PetscSFDestroy(&cellSF) >> checkError;  //!< Return the stuff to PETSc
        }
    }

    /** Restore the fields associated with the particles */
    DMSwarmRestoreField(radsearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> checkError;
    DMSwarmRestoreField(radsearch, "identifier", nullptr, nullptr, (void**)&identifier) >> checkError;
    DMSwarmRestoreField(radsearch, "virtual coord", nullptr, nullptr, (void**)&virtualcoord) >> checkError;
}

void ablate::radiation::Radiation::Initialize(const solver::Range& cellRange, ablate::domain::SubDomain& subDomain) {
    DM faceDM;
    const PetscScalar* faceGeomArray;
    PetscFVFaceGeom* faceGeom;

    DMPlexComputeGeometryFVM(subDomain.GetDM(), &cellGeomVec, &faceGeomVec) >> checkError;  //!< Get the geometry vectors
    VecGetDM(faceGeomVec, &faceDM) >> checkError;
    VecGetArrayRead(faceGeomVec, &faceGeomArray) >> checkError;

    /** Declare some information associated with the field declarations */
    PetscReal* coord;                    //!< Pointer to the coordinate field information
    struct Virtualcoord* virtualcoord;   //!< Pointer to the primary (virtual) coordinate field information
    struct Identifier* identifier;       //!< Pointer to the ray identifier information
    struct Carrier* carrier;             //!< Pointer to the ray carrier information
    struct Identifier* solveidentifier;  //!< Pointer to the ray identifier information

    /** ***********************************************************************************************************************************************
     * Now that the particles have been created, they can be iterated over and each marched one step in space. The global indices of the local
     * ray segment storage can be easily accessed and appended. This forms a local collection of globally index ray segments.
     * */

    PetscInt nglobalpoints = 0;
    PetscInt npoints = 0;
    PetscInt nsolvepoints = 0;                               //!< Counts the solve points in the current domain. This will be adjusted over the course of the loop.
    DMSwarmGetLocalSize(radsearch, &npoints) >> checkError;  //!< Recalculate the number of particles that are in the domain
    DMSwarmGetSize(radsearch, &nglobalpoints) >> checkError;
    PetscInt stepcount = 0;       //!< Count the number of steps that the particles have taken
    while (nglobalpoints != 0) {  //!< WHILE THERE ARE PARTICLES IN ANY DOMAIN
        /** Get all of the ray information from the particle
         * Get the ntheta and nphi from the particle that is currently being looked at. This will be used to identify its ray and calculate its direction. */
        DMSwarmGetField(radsearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> checkError;
        DMSwarmGetField(radsearch, "identifier", nullptr, nullptr, (void**)&identifier) >> checkError;
        DMSwarmGetField(radsearch, "virtual coord", nullptr, nullptr, (void**)&virtualcoord) >> checkError;

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
        PetscInt i[3] = {0, 1, 2};                   //!< Establish the vector here so that it can be iterated.
        for (PetscInt ip = 0; ip < npoints; ip++) {  //!< Iterate over the particles present in the domain. How to isolate the particles in this domain and iterate over them? If there are no
                                                     //!< particles then pass out of initialization.

            /** FIRST TAKE THIS LOCATION INTO THE RAYS VECTOR
             * "I found a particle in my domain. Maybe it was just moved here and I've never seen it before.
             * Therefore, my first step should be to add this location to the local rays vector. Then I can adjust the coordinates and migrate the particle." */

            /** Get the particle coordinates here and put them into the intersect */
            PetscReal position[3] = {(coord[dim * ip + 0]),   //!< x component conversion from spherical coordinates, adding the position of the current cell
                                     (coord[dim * ip + 1]),   //!< y component conversion from spherical coordinates, adding the position of the current cell
                                     (coord[dim * ip + 2])};  //!< z component conversion from spherical coordinates, adding the position of the current cell

            /** This block creates the vector pointing to the cell whose index will be stored during the current loop */
            VecSetValues(intersect, dim, i, position, INSERT_VALUES);  //!< Actually input the values of the vector (There are 'dim' values to input)
            i[0] += dim;                                               //!< Iterate the index by the number of dimensions so that the DMLocatePoints function can be called collectively.
            i[1] += dim;
            i[2] += dim;
        }

        /** Loop through points to try to get the cell that is sitting on that point*/
        PetscSF cellSF = nullptr;  //!< PETSc object for setting up and managing the communication of certain entries of arrays and Vecs between MPI processes.
        DMLocatePoints(subDomain.GetDM(), intersect, DM_POINTLOCATION_NONE, &cellSF) >> checkError;  //!< Call DMLocatePoints here, all of the processes have to call it at once.

        /** An array that maps each point to its containing cell can be obtained with the below
         * We want to get a PetscInt index out of the DMLocatePoints function (cell[n].index)
         * */
        PetscInt nFound;
        const PetscInt* point = nullptr;
        const PetscSFNode* cell = nullptr;
        PetscSFGetGraph(cellSF, nullptr, &nFound, &point, &cell) >> checkError;  //!< Using this to get the petsc int cell number from the struct (SF)

        PetscInt ipart = -1;
        for (PetscInt ip = 0; ip < npoints; ip++) {  //!< Iterate over the particles present in the domain. How to isolate the particles in this domain and iterate over them? If there are no
            ipart++;                                 //!< USE IP TO DEAL WITH DMLOCATE POINTS, USE IPART TO DEAL WITH PARTICLES

            /** IF THE CELL NUMBER IS RETURNED NEGATIVE, THEN WE HAVE REACHED THE BOUNDARY OF THE DOMAIN >> This exits the loop
             * This function returns multiple values if multiple points are input to it
             * Make sure that whatever cell is returned is in the stencil set (and not outside of the radiation domain)
             * Assemble a vector of vectors etc associated with each cell index, angular coordinate, and space step?
             * The boundary has been reached if any of these conditions don't hold
             * */

            /** Check that the particle is in a valid region */
            if ((nFound > -1 && cell[ip].index >= 0 && subDomain.InRegion(cell[ip].index)) || stepcount == 0) {
                index = (stepcount == 0 && !surface) ? identifier[ipart].iCell
                                                     : cell[ip].index;  //!< If this is a surface implementation, then the search particle should never actually enter the boundary cell

                /** If this local rank has never seen this search particle before, then it needs to add a new ray segment to local memory
                 * Hash the identifier into a key value that can be used in the map
                 * We should only iterate the identifier of the search particle (/ add a solver particle) if the point is valid in the domain and is being used
                 * */
                if (rays.count(Key(&identifier[ipart])) == 0) {  //!< IF THIS RAYS VECTOR IS EMPTY FOR THIS DOMAIN, THEN THE PARTICLE HAS NEVER BEEN HERE BEFORE. THEREFORE, ITERATE THE NDOMAINS BY 1.
                    identifier[ipart].nsegment++;                //!< The particle has passed through another domain!
                    DMSwarmAddPoint(radsolve) >> checkError;     //!< Another solve particle is added here because the search particle has entered a new domain

                    DMSwarmGetLocalSize(radsolve,
                                        &nsolvepoints) >>
                        checkError;  //!< Recalculate the number of solve particles so that the last one in the list can be accessed. (I assume that the last one is newest)

                    DMSwarmGetField(radsolve, "identifier", nullptr, nullptr, (void**)&solveidentifier) >>
                        checkError;  //!< Get the fields from the radsolve swarm so the new point can be written to them
                    DMSwarmGetField(radsolve, "carrier", nullptr, nullptr, (void**)&carrier) >> checkError;

                    PetscInt newpoint = nsolvepoints - 1;           //!< This must be replaced with the index of whatever particle there is. Maybe the last index?
                    solveidentifier[newpoint] = identifier[ipart];  //!< Give the particle an identifier which matches the particle it was created with
                    carrier[newpoint].Krad = 1;  //!< The new particle gets an empty carrier because it is holding no information yet (Krad must be initialized to 1 here: everything is init 0)

                    DMSwarmRestoreField(radsolve, "identifier", nullptr, nullptr, (void**)&solveidentifier) >> checkError;  //!< The fields must be returned so that the swarm can be updated correctly?
                    DMSwarmRestoreField(radsolve, "carrier", nullptr, nullptr, (void**)&carrier) >> checkError;
                }

                /** ********************************************
                 * The face stepping routine will give the precise path length of the mesh without any error. It will also allow the faces of the cells to be accounted for so that the
                 * boundary conditions and the conditions at reflection can be accounted for. This will make the entire initialization much faster by only requiring a single step through each
                 * cell. Additionally, the option for reflection is opened because the faces and their normals are now more easily accessed during the initialization. In the future, the carrier
                 * particles may want to be given some information that the boundary label carries when the search particle happens upon it so that imperfect reflection can be implemented.
                 * */

                /** Step 1: Register the current cell index in the rays vector. The physical coordinates that have been set in the previous step / loop will be immediately registered.
                 * */
                rays[Key(&identifier[ipart])].cells.push_back(index);

                /** Step 2: Acquire the intersection of the particle search line with the segment or face. In the case if a two dimensional mesh, the virtual coordinate in the z direction will
                 * need to be solved for because the three dimensional line will not have a literal intersection with the segment of the cell. The third coordinate can be solved for in this case.
                 * Here we are figuring out what distance the ray spends inside the cell that it has just registered.
                 * */
                /** March over each face on this cell in order to check them for the one which intersects this ray next */
                PetscInt numberFaces;
                const PetscInt* cellFaces;
                DMPlexGetConeSize(subDomain.GetDM(), index, &numberFaces) >> checkError;
                DMPlexGetCone(subDomain.GetDM(), index, &cellFaces) >> checkError;  //!< Get the face geometry associated with the current cell
                PetscReal path;

                /** Check every face for intersection with the segment.
                 * The segment with the shortest path length for intersection will be the one that physically intercepts with the cell face and not with the nonphysical plane beyond the face.
                 * */
                for (PetscInt f = 0; f < numberFaces; f++) {
                    PetscInt face = cellFaces[f];
                    DMPlexPointLocalRead(faceDM, face, faceGeomArray, &faceGeom) >> checkError;  //!< Reads the cell location from the current cell

                    /** Get the intersection of the direction vector with the cell face
                     * Use the plane equation and ray segment equation in order to get the face intersection with the shortest path length
                     * This will be the next position of the search particle
                     * */
                    path = FaceIntersect(ipart, virtualcoord, faceGeom);  //!< Use plane intersection equation by getting the centroid and normal vector of the face

                    /** Step 3: Take this path if it is shorter than the previous one, getting the shortest path.
                     * The path should never be zero if the forwardIntersect check is functioning properly.
                     * */
                    if (path > 0) {
                        virtualcoord[ipart].hhere = (virtualcoord[ipart].hhere == 0) ? (path * 1.1) : virtualcoord[ipart].hhere;  //!< Dumb check to ensure that the path length is always updated
                        if (virtualcoord[ipart].hhere > path) {
                            virtualcoord[ipart].hhere =
                                path;  //!> Get the shortest path length of all of the faces. The point must be in the direction that the ray is travelling in order to be valid.
                        }
                    }
                }
                virtualcoord[ipart].hhere = (virtualcoord[ipart].hhere == 0) ? minCellRadius : virtualcoord[ipart].hhere;
                rays[Key(&identifier[ipart])].h.push_back(virtualcoord[ipart].hhere);  //!< Add this space step if the current index is being added.
            }
            /** Step 3.5: Condition for one dimensional domains to avoid infinite rays perpendicular to the x-axis
             * If the domain is 1D and the x-direction of the particle is zero then delete the particle here
             * */
            if ((dim == 1) && (abs(virtualcoord[ipart].xdir) < 0.0000001)) {
                DMSwarmRestoreField(radsearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> checkError;
                DMSwarmRestoreField(radsearch, "identifier", nullptr, nullptr, (void**)&identifier) >> checkError;
                DMSwarmRestoreField(radsearch, "virtual coord", nullptr, nullptr, (void**)&virtualcoord) >> checkError;

                DMSwarmRemovePointAtIndex(radsearch, ipart);  //!< Delete the particle!

                DMSwarmGetField(radsearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> checkError;
                DMSwarmGetField(radsearch, "identifier", nullptr, nullptr, (void**)&identifier) >> checkError;
                DMSwarmGetField(radsearch, "virtual coord", nullptr, nullptr, (void**)&virtualcoord) >> checkError;
                ipart--;  //!< Check the point replacing the one that was deleted
            } else {
                /** Step 4: Push the particle virtual coordinates to the intersection that was found in the previous step.
                 * This ensures that the next calculated path length will start from the boundary of the adjacent cell.
                 * */
                virtualcoord[ipart].x += virtualcoord[ipart].xdir * virtualcoord[ipart].hhere;
                virtualcoord[ipart].y += virtualcoord[ipart].ydir * virtualcoord[ipart].hhere;
                virtualcoord[ipart].z += virtualcoord[ipart].zdir * virtualcoord[ipart].hhere;  //!< Only use the literal intersection coordinate if it exists. This will be decided above.

                /** Step 5: Instead of using the cell face to step into the opposite cell, step the physical coordinates just beyond the intersection.
                 * This avoids issues with hitting corners and potential ghost cell weirdness.
                 * It will be slower than the face flipping but it will be more reliable.
                 * Update the coordinates of the particle.
                 * It doesn't matter which method is used,
                 * this will be the same procedure.
                 * */
                switch (dim) {
                    case 1:
                        coord[ipart] = virtualcoord[ipart].x + (virtualcoord[ipart].xdir * 0.1 * minCellRadius);
                        break;
                    case 2:                                                                                           //!< If there are only two dimensions in this simulation
                        coord[2 * ipart] = virtualcoord[ipart].x + (virtualcoord[ipart].xdir * 0.1 * minCellRadius);  //!< Update the two physical coordinates
                        coord[(2 * ipart) + 1] = virtualcoord[ipart].y + (virtualcoord[ipart].ydir * 0.1 * minCellRadius);
                        break;
                    case 3:                                                                                           //!< If there are three dimensions in this simulation
                        coord[3 * ipart] = virtualcoord[ipart].x + (virtualcoord[ipart].xdir * 0.1 * minCellRadius);  //!< Update the three physical coordinates
                        coord[(3 * ipart) + 1] = virtualcoord[ipart].y + (virtualcoord[ipart].ydir * 0.1 * minCellRadius);
                        coord[(3 * ipart) + 2] = virtualcoord[ipart].z + (virtualcoord[ipart].zdir * 0.1 * minCellRadius);
                        break;
                }                               //!< Update the coordinates of the particle to move it to the center of the adjacent particle.
                virtualcoord[ipart].hhere = 0;  //!< Reset the path length to zero
            }
        }
        /** Restore the fields associated with the particles after all of the particles have been stepped */
        DMSwarmRestoreField(radsearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> checkError;
        DMSwarmRestoreField(radsearch, "identifier", nullptr, nullptr, (void**)&identifier) >> checkError;
        DMSwarmRestoreField(radsearch, "virtual coord", nullptr, nullptr, (void**)&virtualcoord) >> checkError;

        /** Cleanup */
        VecDestroy(&intersect) >> checkError;   //!< Return the vector to PETSc
        PetscSFDestroy(&cellSF) >> checkError;  //!< Return the stuff to PETSc

        if (log) PetscPrintf(subDomain.GetComm(), "Migrate ...");

        /** DMSwarm Migrate to move the ray search particle into the next domain if it has crossed. If it no longer appears in this domain then end the ray segment. */
        DMSwarmMigrate(radsearch, PETSC_TRUE) >> checkError;  //!< Migrate the search particles and remove the particles that have left the domain space.

        DMSwarmGetSize(radsearch, &nglobalpoints) >> checkError;  //!< Update the loop condition. Recalculate the number of particles that are in the domain.
        DMSwarmGetLocalSize(radsearch, &npoints) >> checkError;   //!< Update the loop condition. Recalculate the number of particles that are in the domain.

        if (log) {
            PetscPrintf(subDomain.GetComm(), " Global Steps: %" PetscInt_FMT "    Global Points: %" PetscInt_FMT "\n", stepcount, nglobalpoints);
        }
        stepcount++;
    }
    /** Cleanup */
    DMDestroy(&radsearch) >> checkError;
    VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> checkError;

    if (log) EndEvent();
}

void ablate::radiation::Radiation::Solve(Vec solVec, ablate::domain::Field temperatureField, Vec auxVec) {  //!< Pass in const auto for temperature and Vec for aux
    if (log) StartEvent("Radiation Solve");

    /** Get the array of the solution vector. */
    const PetscScalar* solArray;
    DM solDm;
    VecGetDM(solVec, &solDm);
    VecGetArrayRead(solVec, &solArray);

    /** Get the array of the aux vector. */
    const PetscScalar* auxArray;
    DM auxDm;
    VecGetDM(auxVec, &auxDm);
    VecGetArrayRead(auxVec, &auxArray);

    /** Declare the basic information*/
    PetscReal* sol;          //!< The solution value at any given location
    PetscReal* temperature;  //!< The temperature at any given location
    PetscReal dTheta = ablate::utilities::Constants::pi / (nTheta);
    PetscReal dPhi = (2 * ablate::utilities::Constants::pi) / (nPhi);
    double kappa = 1;  //!< Absorptivity coefficient, property of each cell
    double theta;

    auto absorptivityFunctionContext = absorptivityFunction.context.get();  //!< Get access to the absorption function

    /** Declare some information associated with the field declarations */
    struct Carrier* carrier;        //!< Pointer to the ray carrier information
    struct Identifier* identifier;  //!< Pointer to the ray identifier information

    /** Get the current rank associated with this process */
    PetscMPIInt rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  //!< Get the origin rank of the current process. The particle belongs to this rank. The rank only needs to be read once.

    /** Get all of the ray information from the particle
     * Get the ntheta and nphi from the particle that is currently being looked at. This will be used to identify its ray and calculate its direction. */
    PetscInt npoints;
    DMSwarmGetLocalSize(radsolve, &npoints);  //!< Recalculate the number of particles that are in the domain
    DMSwarmGetField(radsolve, "identifier", nullptr, nullptr, (void**)&identifier);
    DMSwarmGetField(radsolve, "carrier", nullptr, nullptr, (void**)&carrier);

    /** ********************************************************************************************************************************
     * Iterate over the particles that are present in the domain
     * The cells that are in the domain at this point should represent the solve cells attached to the ray segments. They will be transported after local calculation and the non-native ones will
     * be destroyed.
     * First the particles should be zeroed in case they are carrying information from the last time step.
     * Then the entire solve sequence can be run through. This will require that the particles are iterated through twice.
     * */
    for (PetscInt ipart = 0; ipart < npoints; ipart++) {  //!< Iterate through the particles in the space to zero their information.
        carrier[ipart].Ij = 0;                            //!< Zero the intensity of the segment
        carrier[ipart].Krad = 1;                          //!< Zero the total absorption for this domain
        carrier[ipart].I0 = 0;                            //!< Zero the initial intensity of the ray segment
    }
    /** Now that the particle information has been zeroed, the solve can begin. */
    for (PetscInt ipart = 0; ipart < npoints; ipart++) {  //!< Iterate over the particles present in the domain. How to isolate the particles in this domain and iterate over them? If there are no
                                                          //!< particles then pass out of initialization.
        /** Each ray is born here. They begin at the far field temperature.
            Initial ray intensity should be set based on which boundary it is coming from.
            If the ray originates from the walls, then set the initial ray intensity to the wall temperature, etc.
         */
        /** For each domain in the ray (The rays vector will have an added index, splitting every x points) */
        PetscInt numPoints = static_cast<int>(rays[Key(&identifier[ipart])].cells.size());

        if (numPoints > 0) {
            for (PetscInt n = 0; n < numPoints; n++) {
                /** Go through every cell point that is stored within the ray >> FROM THE BOUNDARY TO THE SOURCE
                    Define the absorptivity and temperature in this section
                    For ABLATE implementation, get temperature based on this function
                    Get the array that lives inside the vector
                    Gets the temperature from the cell index specified
                */
                DMPlexPointLocalFieldRead(auxDm, rays[Key(&identifier[ipart])].cells[n], temperatureField.id, auxArray, &temperature);
                DMPlexPointLocalRead(solDm, rays[Key(&identifier[ipart])].cells[n], solArray, &sol);
                /** Input absorptivity (kappa) values from model here. */
                absorptivityFunction.function(sol, *temperature, &kappa, absorptivityFunctionContext);

                carrier[ipart].Ij += FlameIntensity(1 - exp(-kappa * rays[Key(&identifier[ipart])].h[n]), *temperature) * carrier[ipart].Krad;
                carrier[ipart].Krad *= exp(-kappa * rays[Key(&identifier[ipart])].h[n]);  //!< Compute the total absorption for this domain

                if (n == (numPoints - 1)) { /** If this is the beginning of the ray, set this as the initial intensity. (The segment intensities will be filtered through during the origin run) */
                    carrier[ipart].I0 = FlameIntensity(1, *temperature);  //!< Set the initial intensity of the ray segment
                }
            }
        }
    }

    /** Restore the fields associated with the particles after all of the particles have been stepped */
    DMSwarmRestoreField(radsolve, "identifier", nullptr, nullptr, (void**)&identifier);
    DMSwarmRestoreField(radsolve, "carrier", nullptr, nullptr, (void**)&carrier);

    /** ********************************************************************************************************************************
     * Now the carrier has all of the information from the rays that are needed to compute the final ray intensity. Therefore, we will perform the migration.
     * Then, all of the carrier particles will be looped through and the local Origins associated with each cell will be updated
     * */
    PetscInt* rankid;
    DMSwarmGetField(radsolve, "DMSwarm_rank", nullptr, nullptr, (void**)&rankid);
    DMSwarmGetField(radsolve, "identifier", nullptr, nullptr, (void**)&identifier);
    for (PetscInt ipart = 0; ipart < npoints; ipart++) {
        rankid[ipart] = identifier[ipart].origin;
    }
    DMSwarmRestoreField(radsolve, "DMSwarm_rank", nullptr, nullptr, (void**)&rankid);
    DMSwarmRestoreField(radsolve, "identifier", nullptr, nullptr, (void**)&identifier);

    DMSwarmMigrate(radsolve, PETSC_FALSE);  //!< After iterating through all of the particles, perform a migration to the origin ranks. This will move the particles.

    /** ********************************************************************************************************************************
     * Now iterate through all of the particles in order to perform the information transfer */
    DMSwarmGetLocalSize(radsolve, &npoints);                                         //!< Recalculate the number of particles that are in the domain
    DMSwarmGetField(radsolve, "identifier", nullptr, nullptr, (void**)&identifier);  //!< Field information is needed in order to read data from the incoming particles.
    DMSwarmGetField(radsolve, "carrier", nullptr, nullptr, (void**)&carrier);

    /** Iterate through the particles and offload the information to their associated origin cell struct. */
    for (PetscInt ipart = 0; ipart < npoints; ipart++) {
        origin[identifier[ipart].iCell].handler[Key(&identifier[ipart])].Krad = carrier[ipart].Krad;
        origin[identifier[ipart].iCell].handler[Key(&identifier[ipart])].Ij = carrier[ipart].Ij;
        origin[identifier[ipart].iCell].handler[Key(&identifier[ipart])].I0 = carrier[ipart].I0;
    }

    /** ********************************************************************************************************************************
     * Now iterate through all of the ray identifiers in order to compute the final ray intensities */

    DM faceDM;
    const PetscScalar* faceGeomArray;
    PetscFVFaceGeom* faceGeom;
    VecGetDM(faceGeomVec, &faceDM) >> checkError;
    VecGetArrayRead(faceGeomVec, &faceGeomArray) >> checkError;

    for (auto& [iCell, o] : origin) {  //!< Iterate through the cells that are stored in the origin

        origin[iCell].intensity = 0;  //!< Make sure to zero the intensity of every cell before beginning to calculate the intensity for this time step.

        /** for every angle theta
         * for every angle phi
         */
        for (PetscInt ntheta = 0; ntheta < nTheta; ntheta++) {
            for (PetscInt nphi = 0; nphi < nPhi; nphi++) {
                /** Now that we are iterating over every ray identifier in this local domain, we can get all of the particles that are associated with this ray.
                 * We will need to sort the rays in order of domain segment. We need to start at the end of the ray and go towards the beginning of the ray. */
                Identifier loopid = {.origin = rank, .iCell = iCell, .ntheta = ntheta, .nphi = nphi, .nsegment = 1};  //!< Instantiate an identifier associated with this loop location.

                /** Get the maximum nsegment by looping through all of the particles and searching for it. (This is dumb and slow but easy to think of)*/
                bool pointfound = true;
                PetscInt oldsegment = loopid.nsegment;
                while (pointfound) {
                    /** Starting at the first possible segment for this ID
                     * //                             If it exists, increase the segment number that is being checked for.
                     * Also, set the maximum segment that is available for this ray to the segment that is currently being checked.
                     * */
                    if (origin[iCell].handler.count(Key(&loopid)) > 0) {
                        loopid.nsegment++;
                    }
                    pointfound = oldsegment != loopid.nsegment;  //!< If no point was found during the whole for loop, then we must have stumbled on the last segment in this ray.
                    oldsegment = loopid.nsegment;                //!< Set the old segment
                }

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

                loopid.nsegment = 0;
                while (loopid.nsegment <= oldsegment) {  //!< Need to go through all of the ray segments until the origin of the ray is reached

                    origin[iCell].I0 = (oldsegment == loopid.nsegment) ? origin[iCell].handler[Key(&loopid)].I0 : origin[iCell].I0;  //!< Set I0 if it is the last segment in the ray

                    /** Global ray computation happens here, grabbing values from the transported particles.
                     * The rays end here, their intensity is added to the total intensity of the cell.
                     * Gives the partial impact of the ray on the total sphere.
                     * The sin(theta) is a result of the polar coordinate discretization.
                     * In the parallel form at the end of each ray, the absorption of the initial ray and the absorption of the black body source are computed individually at the end.
                     * */
                    /** Parallel things are here
                     * Meaning that the variables required for the parallelizable analytical solution will be declared here */
                    origin[iCell].Isource += origin[iCell].handler[Key(&loopid)].Ij * origin[iCell].Kradd;  //!< Add the black body radiation transmitted through the domain to the source term
                    origin[iCell].Kradd *= origin[iCell].handler[Key(&loopid)].Krad;                        //!< Add the absorption for this domain to the total absorption of the ray
                    loopid.nsegment++;                                                                      //!< Decrement the segment number to move to the next closer segment in the ray.
                }

                if (dim != 1) {
                    theta = (((double)ntheta + 0.5) / (double)nTheta) * ablate::utilities::Constants::pi;  //!< This is a fine method of determining theta because it is in the original domain
                } else {
                    theta = (((double)nphi) / (double)nPhi) * 2 * ablate::utilities::Constants::pi;
                }
                PetscReal ldotn = 1;  //!< If the perpendicular component is not being computed then including this will have no effect.

                //!< If computing surface flux, get the perpendicular component here and multiply the result by it
                if (surface) {  //!< Add the option to the initialization call and make sure that it is stored as a class variable
                    DMPlexPointLocalRead(faceDM, iCell, faceGeomArray, &faceGeom) >> checkError;

                    /** Now that we are iterating over every ray identifier in this local domain, we can get all of the particles that are associated with this ray.
                     * We will need to sort the rays in order of domain segment. We need to start at the end of the ray and go towards the beginning of the ray. */
                    PetscReal faceNormNormalized = sqrt((faceGeom->normal[0] * faceGeom->normal[0]) + (faceGeom->normal[1] * faceGeom->normal[1]) + (faceGeom->normal[2] * faceGeom->normal[2]));
                    PetscReal faceNormx = faceGeom->normal[0] / faceNormNormalized;  //!< Get the normalized face normal (not area scaled)
                    PetscReal faceNormy = faceGeom->normal[1] / faceNormNormalized;
                    PetscReal faceNormz = faceGeom->normal[2] / faceNormNormalized;
                    /** Update the direction vector of the search particle */
                    PetscReal phi = ((double)nphi / (double)nPhi) * 2.0 * ablate::utilities::Constants::pi;
                    PetscReal thetalocal = (((double)ntheta + 0.5) / (double)nTheta) * ablate::utilities::Constants::pi;
                    ldotn = abs(((sin(thetalocal) * cos(phi)) * faceNormx) + ((sin(thetalocal) * sin(phi)) * faceNormy) + (cos(thetalocal) * faceNormz));
                }
                origin[iCell].intensity += ((origin[iCell].I0 * origin[iCell].Kradd) + origin[iCell].Isource) * abs(sin(theta)) * dTheta * dPhi * ldotn;  //!< Final ray calculation
            }
        }
    }

    /** ********************************************************************************************************************************
     * Need to delete all of the particles that were transported to different domains so that the process can be repeated in the next step. */

    /** Delete all of the particles that were transported to their origin domains -> Delete if the particle has travelled to get here and isn't native */
    for (PetscInt ipart = 0; ipart < npoints; ipart++) {
        if (identifier[ipart].origin == rank && identifier[ipart].nsegment != 1) {
            DMSwarmRestoreField(radsolve, "identifier", nullptr, nullptr, (void**)&identifier);  //!< Need to restore the field access before deleting a point
            DMSwarmRestoreField(radsolve, "carrier", nullptr, nullptr, (void**)&carrier);

            DMSwarmRemovePointAtIndex(radsolve, ipart);  //!< Delete the particle!

            DMSwarmGetLocalSize(radsolve, &npoints);                                         //!< Need to recalculate the number of particles that are in the domain again
            DMSwarmGetField(radsolve, "identifier", nullptr, nullptr, (void**)&identifier);  //!< Get the field back
            DMSwarmGetField(radsolve, "carrier", nullptr, nullptr, (void**)&carrier);
            ipart--;  //!< Check the point replacing the one that was deleted
        }
    }

    /** Restore the fields associated with the particles after all of the particles have been stepped. */
    DMSwarmRestoreField(radsolve, "identifier", nullptr, nullptr, (void**)&identifier);
    DMSwarmRestoreField(radsolve, "carrier", nullptr, nullptr, (void**)&carrier);

    /** ********************************************************************************************************************************
     * Loop through the cell range and compute the origin contributions. */

    DM cellDM;
    const PetscScalar* cellGeomArray;

    if (log) {
        VecGetDM(cellGeomVec, &cellDM) >> checkError;
        VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
        printf("x           y           z           G           L           T\n");  //!< Line labelling the log outputs for readability
    }

    for (auto& [iCell, o] : origin) {  //!< Iterate through the cells that are stored in the origin
        /** Gets the temperature from the cell index specified */
        PetscInt index = -1;  //!< Index value for the losses temperature reading
        /** In the case of a surface implementation, the temperature for the losses will be the temperature of the boundary cell that the face is attached to.
         * In the case of a volume implementation, the temperature of the losses will be the temperature of the volumetric origin cell.
         * This distinction must be made because the temperature of faces is undefined.
         * */
        if (surface) {
            PetscInt numberNeighborCells;
            const PetscInt* neighborCells;

            DMPlexGetSupportSize(solDm, iCell, &numberNeighborCells) >> ablate::checkError;  //!< Get the cells on each side of this face to check for boundary cells
            DMPlexGetSupport(solDm, iCell, &neighborCells) >> ablate::checkError;
            for (PetscInt n = 0; n < numberNeighborCells; n++) {
                PetscInt cell = neighborCells[n];  //!< Contains the cell indexes of the neighbor cells
                if (fieldBoundary->InRegion(fieldBoundary, solDm, cell)) {
                    index = cell;
                }
            }
        } else {
            index = iCell;
        }
        DMPlexPointLocalFieldRead(auxDm, index, temperatureField.id, auxArray, &temperature);
        PetscReal losses = 4 * ablate::utilities::Constants::sbc * *temperature * *temperature * *temperature * *temperature;
        if (surface) losses /= 2;  //!< If this is a surface then losses will only leave the hemisphere
        if (log) {
            PetscReal centroid[3];
            DMPlexComputeCellGeometryFVM(cellDM, iCell, nullptr, centroid, nullptr) >> checkError;  //!< Reads the cell location from the current cell
            printf("%f %f %f %f %f %f\n", centroid[0], centroid[1], centroid[2], origin[iCell].intensity, losses, *temperature);
        }
        origin[iCell].intensity = -kappa * (losses - origin[iCell].intensity);
    }

    /** Cleanup */
    VecRestoreArrayRead(solVec, &solArray);
    VecRestoreArrayRead(auxVec, &auxArray);
    VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> checkError;

    if (log) {
        EndEvent();
        VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    }
}

PetscReal ablate::radiation::Radiation::FlameIntensity(double epsilon, double temperature) { /** Gets the flame intensity based on temperature and emissivity (black body intensity) */
    return epsilon * ablate::utilities::Constants::sbc * temperature * temperature * temperature * temperature / ablate::utilities::Constants::pi;
}

void ablate::radiation::Radiation::UpdateCoordinates(PetscInt ipart, Virtualcoord* virtualcoord, PetscReal* coord) const {
    switch (dim) {
        case 1:
            coord[ipart] = virtualcoord[ipart].x;
            break;
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

PetscReal ablate::radiation::Radiation::FaceIntersect(PetscInt ip, Virtualcoord* virtualcoord, PetscFVFaceGeom* faceGeom) const {
    PetscReal ldotn = (virtualcoord[ip].xdir * faceGeom->normal[0]) + (virtualcoord[ip].ydir * faceGeom->normal[1]) + (virtualcoord[ip].zdir * faceGeom->normal[2]);
    if (ldotn == 0) return 0;
    PetscReal d = (((faceGeom->normal[0] * faceGeom->centroid[0]) + (faceGeom->normal[1] * faceGeom->centroid[1]) + (faceGeom->normal[2] * faceGeom->centroid[2])) -
                   ((faceGeom->normal[0] * virtualcoord[ip].x) + (faceGeom->normal[1] * virtualcoord[ip].y) + (faceGeom->normal[2] * virtualcoord[ip].z))) /
                  ldotn;  //!<(planeNormal.dot(planePoint) - planeNormal.dot(linePoint)) / planeNormal.dot(lineDirection.normalize())
    if (d > minCellRadius * 1E-5) {
        return d;
    } else {
        return 0;
    }
}

#include "registrar.hpp"
REGISTER_DEFAULT(ablate::radiation::Radiation, ablate::radiation::Radiation, "A solver for radiative heat transfer in participating media", ARG(std::string, "id", "the name of the flow field"),
                 ARG(ablate::domain::Region, "region", "the region to apply this solver."), ARG(ablate::domain::Region, "fieldBoundary", "boundary of the radiation region"),
                 ARG(int, "rays", "number of rays used by the solver"), ARG(ablate::eos::radiationProperties::RadiationModel, "properties", "the radiation properties model"),
                 OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));