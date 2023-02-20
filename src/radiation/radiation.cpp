#include "radiation.hpp"
#include <petsc/private/dmimpl.h>
#include <petscdm.h>
#include <petscdmswarm.h>
#include <petscsf.h>
#include <utility>
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscUtilities.hpp"

ablate::radiation::Radiation::Radiation(const std::string& solverId, const std::shared_ptr<domain::Region>& region, const PetscInt raynumber,
                                        std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, std::shared_ptr<ablate::monitors::logs::Log> log)
    : nTheta(raynumber), nPhi(2 * raynumber), solverId(solverId), region(region), radiationModel(std::move(radiationModelIn)), log(std::move(log)) {}

ablate::radiation::Radiation::~Radiation() {
    if (faceGeomVec) VecDestroy(&faceGeomVec) >> utilities::PetscUtilities::checkError;
    if (cellGeomVec) VecDestroy(&cellGeomVec) >> utilities::PetscUtilities::checkError;
    if (remoteAccess) PetscSFDestroy(&remoteAccess) >> utilities::PetscUtilities::checkError;
    MPI_Type_free(&carrierMpiType) >> utilities::MpiUtilities::checkError;
}

/** allows initialization after the subdomain and dm is established */
void ablate::radiation::Radiation::Setup(const ablate::domain::Range& cellRange, ablate::domain::SubDomain& subDomain) {
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

    StartEvent("Radiation::Setup");
    if (log) log->Printf("Starting Initialize\n");

    DMPlexGetMinRadius(subDomain.GetDM(), &minCellRadius) >> utilities::PetscUtilities::checkError;

    /** do a simple sanity check for labels */
    PetscMPIInt rank = 0;
    MPI_Comm_rank(subDomain.GetComm(), &rank);  //!< Get the origin rank of the current process. The particle belongs to this rank. The rank only needs to be read once.

    /** Setup the particles and their associated fields including: origin domain/ ray identifier / # domains crossed, and coordinates. Instantiate ray particles for each local cell only. */
    numberOriginCells = (cellRange.end - cellRange.start);
    raysPerCell = nTheta * nPhi;
    numberOriginRays = numberOriginCells * raysPerCell;  //!< Number of points to insert into the particle field. One particle for each ray.

    /** Create the DMSwarm */
    DMCreate(subDomain.GetComm(), &radSearch) >> utilities::PetscUtilities::checkError;
    DMSetType(radSearch, DMSWARM) >> utilities::PetscUtilities::checkError;
    DMSetDimension(radSearch, dim) >> utilities::PetscUtilities::checkError;

    /** Configure radsearch to be of type PIC/Basic */
    DMSwarmSetType(radSearch, DMSWARM_PIC) >> utilities::PetscUtilities::checkError;
    DMSwarmSetCellDM(radSearch, subDomain.GetDM()) >> utilities::PetscUtilities::checkError;

    /** Register fields within the DMSwarm */
    DMSwarmRegisterUserStructField(radSearch, IdentifierField, sizeof(Identifier)) >>
        utilities::PetscUtilities::checkError;  //!< A field to store the ray identifier [origin][iCell][ntheta][nphi][ndomain]
    DMSwarmRegisterUserStructField(radSearch, VirtualCoordField, sizeof(Virtualcoord)) >>
        utilities::PetscUtilities::checkError;                                         //!< A field representing the three dimensional coordinates of the particle. Three "virtual" dims are required.
    DMSwarmFinalizeFieldRegister(radSearch) >> utilities::PetscUtilities::checkError;  //!< Initialize the fields that have been defined

    /** Set initial local sizes of the DMSwarm with a buffer length of zero */
    DMSwarmSetLocalSizes(radSearch, numberOriginRays, 0) >>
        utilities::PetscUtilities::checkError;  //!< Set the number of initial particles to the number of rays in the subdomain. Set the buffer size to zero.

    /** Declare some information associated with the field declarations */
    PetscReal* coord;                   //!< Pointer to the coordinate field information
    PetscInt* index;                    //!< Pointer to the cell index information
    struct Virtualcoord* virtualcoord;  //!< Pointer to the primary (virtual) coordinate field information
    struct Identifier* identifier;      //!< Pointer to the ray identifier information

    /** Get the fields associated with the particle swarm so that they can be modified */
    DMSwarmGetField(radSearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> utilities::PetscUtilities::checkError;
    DMSwarmGetField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> utilities::PetscUtilities::checkError;
    DMSwarmGetField(radSearch, IdentifierField, nullptr, nullptr, (void**)&identifier) >> utilities::PetscUtilities::checkError;
    DMSwarmGetField(radSearch, VirtualCoordField, nullptr, nullptr, (void**)&virtualcoord) >> utilities::PetscUtilities::checkError;

    // precompute the intensityFactor
    PetscReal dTheta = ablate::utilities::Constants::pi / (nTheta);
    PetscReal dPhi = (2 * ablate::utilities::Constants::pi) / (nPhi);
    gainsFactor.resize(numberOriginRays);

    //!< Initialize a counter to represent the particle index. This will be iterated every time that the inner loop is passed through.
    PetscInt ipart = 0;

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {            //!< This will iterate only though local cells
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;  //!< Isolates the valid cells
        PetscReal centroid[3];
        PetscReal normal[3] = {0.0, 0.0, 0.0};
        DMPlexComputeCellGeometryFVM(subDomain.GetDM(), iCell, nullptr, centroid, normal) >> utilities::PetscUtilities::checkError;

        /** for every angle theta
         * for every angle phi
         */
        for (PetscInt ntheta = 0; ntheta < nTheta; ntheta++) {
            for (PetscInt nphi = 0; nphi < nPhi; nphi++) {
                /** Get the initial direction of the search particle from the angle number that it was initialized with */
                double theta = (((double)ntheta + 0.5) / (double)nTheta) * ablate::utilities::Constants::pi;  //!< Theta angle of the ray
                double phi = ((double)nphi / (double)nPhi) * 2.0 * ablate::utilities::Constants::pi;          //!<  Phi angle of the ray

                /** Update the direction vector of the search particle */
                virtualcoord[ipart].xdir = (sin(theta) * cos(phi));  //!< x component conversion from spherical coordinates, adding the position of the current cell
                virtualcoord[ipart].ydir = (sin(theta) * sin(phi));  //!< y component conversion from spherical coordinates, adding the position of the current cell
                virtualcoord[ipart].zdir = (cos(theta));             //!< z component conversion from spherical coordinates, adding the position of the current cell

                /** Get the particle coordinate field and write the cellGeom->centroid[xyz] into it */
                virtualcoord[ipart].x = centroid[0] + (virtualcoord[ipart].xdir * 0.1 * minCellRadius);  //!< Offset from the centroid slightly so they sit in a cell if they are on its face.
                virtualcoord[ipart].y = centroid[1] + (virtualcoord[ipart].ydir * 0.1 * minCellRadius);
                virtualcoord[ipart].z = centroid[2] + (virtualcoord[ipart].zdir * 0.1 * minCellRadius);

                // Init hhere to default value
                virtualcoord[ipart].hhere = 0.0;

                /** Update the physical coordinate field so that the real particle location can be updated. */
                /** Update the physical coordinate field so that the real particle location can be updated. */
                UpdateCoordinates(ipart, virtualcoord, coord, 0.0);  //! adv value of 0.0 places the particle exactly where the virtual coordinates are.

                /** Label the particle with the ray identifier. With what is known at this point**/
                identifier[ipart].originRank = rank;    //!< Input the ray identifier. This location scheme represents stepping four entries for every particle index increase
                identifier[ipart].originRayId = ipart;  //! This serve to identify the ray id on the origin
                identifier[ipart].remoteRank = PETSC_DECIDE;
                identifier[ipart].remoteRayId = PETSC_DECIDE;
                identifier[ipart].nSegment = -1;

                // Compute the intensityFactor
                // If surface, get the perpendicular component here and multiply the result by it
                gainsFactor[ipart] = abs(sin(theta)) * dTheta * dPhi * SurfaceComponent(normal, iCell, nphi, ntheta);

                /** Set the index of the field value so that it can be written to for every particle */
                ipart++;  //!< Must be iterated at the end since the value is initialized at zero.
            }
        }
    }

    /** Restore the fields associated with the particles */
    DMSwarmRestoreField(radSearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> utilities::PetscUtilities::checkError;
    DMSwarmRestoreField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> utilities::PetscUtilities::checkError;
    DMSwarmRestoreField(radSearch, IdentifierField, nullptr, nullptr, (void**)&identifier) >> utilities::PetscUtilities::checkError;
    DMSwarmRestoreField(radSearch, VirtualCoordField, nullptr, nullptr, (void**)&virtualcoord) >> utilities::PetscUtilities::checkError;

    DMSwarmMigrate(radSearch, PETSC_TRUE) >> utilities::PetscUtilities::checkError;  //!< Sets the search particles in the cell indexes to which they have been assigned

    if (log) {
        log->Printf("Particles Setup\n");
    }
    EndEvent();
}

void ablate::radiation::Radiation::Initialize(const ablate::domain::Range& cellRange, ablate::domain::SubDomain& subDomain) {
    StartEvent("Radiation::Initialize");
    DM faceDM;
    const PetscScalar* faceGeomArray;

    // create a basic swarm without pic, this will be used to hold the return identification for each particle
    DM radReturn;
    DMCreate(subDomain.GetComm(), &radReturn) >> utilities::PetscUtilities::checkError;
    DMSetType(radReturn, DMSWARM) >> utilities::PetscUtilities::checkError;
    DMSetDimension(radReturn, dim) >> utilities::PetscUtilities::checkError;
    DMSwarmSetType(radReturn, DMSWARM_BASIC) >> utilities::PetscUtilities::checkError;

    DMSwarmRegisterUserStructField(radReturn, IdentifierField, sizeof(Identifier)) >> utilities::PetscUtilities::checkError;
    DMSwarmFinalizeFieldRegister(radReturn) >> utilities::PetscUtilities::checkError;  //!< Initialize the fields that have been defined

    /** This will be added to as rays are created on each rank */
    DMSwarmSetLocalSizes(radReturn, 0, 100) >> utilities::PetscUtilities::checkError;

    DMPlexComputeGeometryFVM(subDomain.GetDM(), &cellGeomVec, &faceGeomVec) >> utilities::PetscUtilities::checkError;  //!< Get the geometry vectors
    VecGetDM(faceGeomVec, &faceDM) >> utilities::PetscUtilities::checkError;
    VecGetArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;

    /** Exact some information associated with the field declarations from the swarm*/
    PetscReal* coord;  //!< Pointer to the coordinate field information
    PetscInt* index;
    struct Virtualcoord* virtualcoord;  //!< Pointer to the primary (virtual) coordinate field information
    struct Identifier* identifier;      //!< Pointer to the ray identifier information

    /** ***********************************************************************************************************************************************
     * Now that the particles have been created, they can be iterated over and each marched one step in space. The global indices of the local
     * ray segment storage can be easily accessed and appended. This forms a local collection of globally index ray segments.
     * */

    PetscInt nglobalpoints = 0;
    PetscInt npoints = 0;
    DMSwarmGetLocalSize(radSearch, &npoints) >> utilities::PetscUtilities::checkError;  //!< Recalculate the number of particles that are in the domain
    DMSwarmGetSize(radSearch, &nglobalpoints) >> utilities::PetscUtilities::checkError;
    PetscInt stepcount = 0;       //!< Count the number of steps that the particles have taken
    while (nglobalpoints != 0) {  //!< WHILE THERE ARE PARTICLES IN ANY DOMAIN
        // If this local rank has never seen this search particle before, then it needs to add a new ray segment to local memory and record its index
        IdentifyNewRaysOnRank(subDomain, radReturn);

        /** Use the ParticleStep function to calculate the path lengths of the rays through each cell so that they can be stored.
         * This function also sets up the solve particle infrastructure.
         * */
        ParticleStep(subDomain, faceDM, faceGeomArray, radReturn);

        /** Get all of the ray information from the particle
         * Get the ntheta and nphi from the particle that is currently being looked at. This will be used to identify its ray and calculate its direction. */
        DMSwarmGetField(radSearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> utilities::PetscUtilities::checkError;
        DMSwarmGetField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> utilities::PetscUtilities::checkError;
        DMSwarmGetField(radSearch, IdentifierField, nullptr, nullptr, (void**)&identifier) >> utilities::PetscUtilities::checkError;
        DMSwarmGetField(radSearch, VirtualCoordField, nullptr, nullptr, (void**)&virtualcoord) >> utilities::PetscUtilities::checkError;

        for (PetscInt ipart = 0; ipart < npoints; ipart++) {  //!< Iterate over the particles present in the domain. How to isolate the particles in this domain and iterate over them? If there are no

            /** IF THE CELL NUMBER IS RETURNED NEGATIVE, THEN WE HAVE REACHED THE BOUNDARY OF THE DOMAIN >> This exits the loop
             * This function returns multiple values if multiple points are input to it
             * Make sure that whatever cell is returned is in the radiation domain
             * Assemble a vector of vectors etc associated with each cell index, angular coordinate, and space step?
             * The boundary has been reached if any of these conditions don't hold
             * */

            /** Step 3.5: The cells need to be removed if they are inside a boundary cell.
             * Therefore, after each step (where the particle location is fed in) check for whether the cell is still within the interior region.
             * If it is not (if it's in a boundary cell) then it should be deleted here.
             * Condition for one dimensional domains to avoid infinite rays perpendicular to the x-axis
             * If the domain is 1D and the x-direction of the particle is zero then delete the particle here
             * */
            if ((!(domain::Region::InRegion(region, subDomain.GetDM(), index[ipart]))) || ((dim == 1) && (abs(virtualcoord[ipart].xdir) < 0.0000001))) {
                //! If the boundary has been reached by this ray, then add a boundary condition segment to the ray.
                auto& ray = raySegments[identifier[ipart].remoteRayId];
                auto& raySegment = ray.emplace_back();
                raySegment.cell = index[ipart];
                raySegment.pathLength = -1;

                //! Delete the search particle associated with the ray
                DMSwarmRestoreField(radSearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> utilities::PetscUtilities::checkError;
                DMSwarmRestoreField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> utilities::PetscUtilities::checkError;
                DMSwarmRestoreField(radSearch, IdentifierField, nullptr, nullptr, (void**)&identifier) >> utilities::PetscUtilities::checkError;
                DMSwarmRestoreField(radSearch, VirtualCoordField, nullptr, nullptr, (void**)&virtualcoord) >> utilities::PetscUtilities::checkError;

                DMSwarmRemovePointAtIndex(radSearch, ipart);  //!< Delete the particle!
                DMSwarmGetLocalSize(radSearch, &npoints);

                DMSwarmGetField(radSearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> utilities::PetscUtilities::checkError;
                DMSwarmGetField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> utilities::PetscUtilities::checkError;
                DMSwarmGetField(radSearch, IdentifierField, nullptr, nullptr, (void**)&identifier) >> utilities::PetscUtilities::checkError;
                DMSwarmGetField(radSearch, VirtualCoordField, nullptr, nullptr, (void**)&virtualcoord) >> utilities::PetscUtilities::checkError;
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
                UpdateCoordinates(ipart, virtualcoord, coord, 0.1);  //!< Update the coordinates of the particle to move it to the center of the adjacent particle.
                virtualcoord[ipart].hhere = 0;                       //!< Reset the path length to zero
            }
        }
        /** Restore the fields associated with the particles after all of the particles have been stepped */
        DMSwarmRestoreField(radSearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> utilities::PetscUtilities::checkError;
        DMSwarmRestoreField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> utilities::PetscUtilities::checkError;
        DMSwarmRestoreField(radSearch, IdentifierField, nullptr, nullptr, (void**)&identifier) >> utilities::PetscUtilities::checkError;
        DMSwarmRestoreField(radSearch, VirtualCoordField, nullptr, nullptr, (void**)&virtualcoord) >> utilities::PetscUtilities::checkError;

        if (log) log->Printf("Migrate ...");

        /** DMSwarm Migrate to move the ray search particle into the next domain if it has crossed. If it no longer appears in this domain then end the ray segment. */
        DMSwarmMigrate(radSearch, PETSC_TRUE) >> utilities::PetscUtilities::checkError;  //!< Migrate the search particles and remove the particles that have left the domain space.

        DMSwarmGetSize(radSearch, &nglobalpoints) >> utilities::PetscUtilities::checkError;  //!< Update the loop condition. Recalculate the number of particles that are in the domain.
        DMSwarmGetLocalSize(radSearch, &npoints) >> utilities::PetscUtilities::checkError;   //!< Update the loop condition. Recalculate the number of particles that are in the domain.

        if (log) {
            log->Printf(" Global Steps: %" PetscInt_FMT "    Global Points: %" PetscInt_FMT "\n", stepcount, nglobalpoints);
        }
        stepcount++;
    }
    // Cleanup
    DMDestroy(&radSearch) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;
    EndEvent();

    // Move the identifiers in radReturn back to origin
    StartEvent("Radiation::RadReturn");
    DMSwarmMigrate(radReturn, PETSC_TRUE) >> utilities::PetscUtilities::checkError;
    EndEvent();

    /* radReturn contains a list of all ranks (including this one) that contain segments for each ray.
     * Count the number of ray segments per ray
     */
    raySegmentsPerOriginRay.resize(numberOriginRays, 0);

    // March over each returned segment and add to the numberOriginRays
    PetscInt numberOfReturnedSegments;
    DMSwarmGetLocalSize(radReturn, &numberOfReturnedSegments);
    struct Identifier* returnIdentifiers;  //!< Pointer to the ray identifier information
    DMSwarmGetField(radReturn, IdentifierField, nullptr, nullptr, (void**)&returnIdentifiers) >>
        utilities::PetscUtilities::checkError;  //!< Get the fields from the radsolve swarm so the new point can be written to them
    for (PetscInt p = 0; p < numberOfReturnedSegments; ++p) {
        // There may be duplicates so take the maximum based upon nSegment for the size
        raySegmentsPerOriginRay[returnIdentifiers[p].originRayId] = PetscMax(raySegmentsPerOriginRay[returnIdentifiers[p].originRayId], (returnIdentifiers[p].nSegment + 1));
    }

    // Keep track of the offset for each originRay assuming the memory is in order
    std::vector<PetscInt> rayOffset(numberOriginRays);
    PetscInt uniqueRaySegments = 0;
    for (std::size_t r = 0; r < raySegmentsPerOriginRay.size(); r++) {
        rayOffset[r] = uniqueRaySegments;
        uniqueRaySegments += raySegmentsPerOriginRay[r];
    }

    /* Build the leafs for the petscSf.
     * - Each root corresponds to a single ray/segment id in the raySegmentSummary
     * - Each corresponding leaf points to a local/remote remoteRayCalculation indexed based upon the remote ray index
     * - because there are duplicates we are taking only the returnIdentifiers for each localMemoryIndex
     */
    PetscSFNode* remoteRayInformation;
    PetscMalloc1(uniqueRaySegments, &remoteRayInformation) >> utilities::PetscUtilities::checkError;
    for (PetscInt p = 0; p < numberOfReturnedSegments; ++p) {
        // determine where in local memory this remoteRayInformation corresponds to
        // first offset it by the originRayId
        PetscInt localMemoryIndex = rayOffset[returnIdentifiers[p].originRayId];

        // order them in terms of origin to the farthest away
        localMemoryIndex += returnIdentifiers[p].nSegment;

        // Store the remote ray information at this localMemoryIndex
        remoteRayInformation[localMemoryIndex].rank = returnIdentifiers[p].remoteRank;
        remoteRayInformation[localMemoryIndex].index = returnIdentifiers[p].remoteRayId;
    }

    // remove the radReturn, the information has now been moved to the remoteRayInformation
    DMSwarmRestoreField(radReturn, IdentifierField, nullptr, nullptr, (void**)&returnIdentifiers) >>
        utilities::PetscUtilities::checkError;  //!< Get the fields from the radsolve swarm so the new point can be written to them
    DMDestroy(&radReturn) >> utilities::PetscUtilities::checkError;

    // Create the remote access structure
    PetscSFCreate(PETSC_COMM_WORLD, &remoteAccess) >> utilities::PetscUtilities::checkError;
    PetscSFSetFromOptions(remoteAccess) >> utilities::PetscUtilities::checkError;
    PetscSFSetGraph(remoteAccess, (PetscInt)raySegments.size(), uniqueRaySegments, nullptr, PETSC_OWN_POINTER, remoteRayInformation, PETSC_OWN_POINTER) >> utilities::PetscUtilities::checkError;
    PetscSFSetUp(remoteAccess) >> utilities::PetscUtilities::checkError;

    // Size up the memory to hold the local calculations and the retrieved information
    raySegmentsCalculations.resize(raySegments.size() * absorptivityFunction.propertySize);
    raySegmentSummary.resize(numberOfReturnedSegments * absorptivityFunction.propertySize);
    evaluatedGains.resize(numberOriginCells * absorptivityFunction.propertySize);  //! Size each of the entries to hold all of the wavelengths being transported.

    // Create a mpi data type to allow reducing the remoteRayCalculation to raySegmentSummary
    PetscInt count = 2 * absorptivityFunction.propertySize;  //! = 2 * (the number of independant wavelengths that are being considered). Should be read from absorption model.
    MPI_Type_contiguous(count, MPIU_REAL, &carrierMpiType) >> utilities::MpiUtilities::checkError;
    MPI_Type_commit(&carrierMpiType) >> utilities::MpiUtilities::checkError;
}

PetscReal ablate::radiation::Radiation::FlameIntensity(double epsilon, double temperature) { /** Gets the flame intensity based on temperature and emissivity (black body intensity) */
    return epsilon * ablate::utilities::Constants::sbc * temperature * temperature * temperature * temperature / ablate::utilities::Constants::pi;
}

void ablate::radiation::Radiation::UpdateCoordinates(PetscInt ipart, Virtualcoord* virtualcoord, PetscReal* coord, PetscReal adv) const {
    switch (dim) {
        case 1:
            coord[ipart] = virtualcoord[ipart].x + (virtualcoord[ipart].xdir * adv * minCellRadius);
            break;
        case 2:                                                                                           //!< If there are only two dimensions in this simulation
            coord[2 * ipart] = virtualcoord[ipart].x + (virtualcoord[ipart].xdir * adv * minCellRadius);  //!< Update the two physical coordinates
            coord[(2 * ipart) + 1] = virtualcoord[ipart].y + (virtualcoord[ipart].ydir * adv * minCellRadius);
            break;
        case 3:                                                                                           //!< If there are three dimensions in this simulation
            coord[3 * ipart] = virtualcoord[ipart].x + (virtualcoord[ipart].xdir * adv * minCellRadius);  //!< Update the three physical coordinates
            coord[(3 * ipart) + 1] = virtualcoord[ipart].y + (virtualcoord[ipart].ydir * adv * minCellRadius);
            coord[(3 * ipart) + 2] = virtualcoord[ipart].z + (virtualcoord[ipart].zdir * adv * minCellRadius);
            break;
    }
}

PetscReal ablate::radiation::Radiation::FaceIntersect(PetscInt ip, Virtualcoord* virtualcoord, PetscFVFaceGeom* faceGeom) const {
    //!<(planeNormal.dot(planePoint) - planeNormal.dot(linePoint)) / planeNormal.dot(lineDirection.normalize())
    PetscReal ldotn = 0.0;
    PetscReal d = 0.0;
    switch (dim) {
        case 3:
            ldotn += virtualcoord[ip].zdir * faceGeom->normal[2];
            d += (faceGeom->normal[2] * faceGeom->centroid[2]) - (faceGeom->normal[2] * virtualcoord[ip].z);
            [[fallthrough]];
        case 2:
            ldotn += virtualcoord[ip].ydir * faceGeom->normal[1];
            d += (faceGeom->normal[1] * faceGeom->centroid[1]) - (faceGeom->normal[1] * virtualcoord[ip].y);
            [[fallthrough]];
        default:
            ldotn += virtualcoord[ip].xdir * faceGeom->normal[0];
            d += (faceGeom->normal[0] * faceGeom->centroid[0]) - (faceGeom->normal[0] * virtualcoord[ip].x);
    }

    if (ldotn == 0) return 0;
    d /= ldotn;
    if (d > minCellRadius * 1E-5) {
        return d;
    } else {
        return 0;
    }
}

PetscReal ablate::radiation::Radiation::SurfaceComponent(const PetscReal normal[], PetscInt iCell, PetscInt nphi, PetscInt ntheta) { return 1.0; }

void ablate::radiation::Radiation::IdentifyNewRaysOnRank(ablate::domain::SubDomain& subDomain, DM radReturn) { /** Check that the particle is in a valid region */
    PetscInt npoints = 0;
    DMSwarmGetLocalSize(radSearch, &npoints) >> utilities::PetscUtilities::checkError;

    PetscMPIInt rank = 0;
    MPI_Comm_rank(subDomain.GetComm(), &rank);

    /** Declare some information associated with the field declarations */
    PetscInt* index;
    struct Identifier* identifiers;  //!< Pointer to the ray identifier information

    /** Get all of the ray information from the particle
     * Get the ntheta and nphi from the particle that is currently being looked at. This will be used to identify its ray and calculate its direction. */
    DMSwarmGetField(radSearch, IdentifierField, nullptr, nullptr, (void**)&identifiers) >> utilities::PetscUtilities::checkError;
    DMSwarmGetField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> utilities::PetscUtilities::checkError;

    for (PetscInt ipart = 0; ipart < npoints; ipart++) {
        /** Check that the particle is in a valid region */
        if (index[ipart] >= 0) {
            auto& identifier = identifiers[ipart];
            // If this local rank has never seen this search particle before, then it needs to add a new ray segment to local memory and record its index
            if (identifier.remoteRank != rank) {
                // Update the identifier for this rank.  When it gets sent to another rank a copy will be made
                identifier.remoteRank = rank;
                // set the remoteRayId to be the next one in the way
                identifier.remoteRayId = (PetscInt)raySegments.size();
                // bump the nSegment
                identifier.nSegment++;

                // Create an empty struct in the ray
                raySegments.emplace_back();

                // store this ray and information in the return data
                DMSwarmAddPoint(radReturn) >> utilities::PetscUtilities::checkError;  //!< Another solve particle is added here because the search particle has entered a new domain
                struct Identifier* returnIdentifiers;                                 //!< Pointer to the ray identifier information
                PetscInt* returnRank;                                                 //! while we are here, set the return rank.  This won't change anything until migrate is called
                DMSwarmGetField(radReturn, IdentifierField, nullptr, nullptr, (void**)&returnIdentifiers) >>
                    utilities::PetscUtilities::checkError;  //!< Get the fields from the radsolve swarm so the new point can be written to them
                DMSwarmGetField(radReturn, DMSwarmField_rank, nullptr, nullptr, (void**)&returnRank) >> utilities::PetscUtilities::checkError;

                // these are only created as remote rays are identified, so we can remoteRayId for the rank
                returnIdentifiers[identifier.remoteRayId] = identifier;
                returnRank[identifier.remoteRayId] = identifier.originRank;

                DMSwarmRestoreField(radReturn, IdentifierField, nullptr, nullptr, (void**)&returnIdentifiers) >>
                    utilities::PetscUtilities::checkError;  //!< Get the fields from the radsolve swarm so the new point can be written to them
                DMSwarmRestoreField(radReturn, DMSwarmField_rank, nullptr, nullptr, (void**)&returnRank) >> utilities::PetscUtilities::checkError;
            }
        }
    }
    DMSwarmRestoreField(radSearch, IdentifierField, nullptr, nullptr, (void**)&identifiers) >> utilities::PetscUtilities::checkError;
    DMSwarmRestoreField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> utilities::PetscUtilities::checkError;
}

void ablate::radiation::Radiation::ParticleStep(ablate::domain::SubDomain& subDomain, DM faceDM, const PetscScalar* faceGeomArray, DM radReturn) { /** Check that the particle is in a valid region */
    PetscInt npoints = 0;
    PetscInt nglobalpoints = 0;
    DMSwarmGetLocalSize(radSearch, &npoints) >> utilities::PetscUtilities::checkError;
    DMSwarmGetSize(radSearch, &nglobalpoints) >> utilities::PetscUtilities::checkError;

    PetscFVFaceGeom* faceGeom;

    PetscMPIInt rank = 0;
    MPI_Comm_rank(subDomain.GetComm(), &rank);

    /** Declare some information associated with the field declarations */
    PetscInt* index;
    struct Virtualcoord* virtualcoords;  //!< Pointer to the primary (virtual) coordinate field information
    struct Identifier* identifiers;      //!< Pointer to the ray identifier information

    /** Get all of the ray information from the particle
     * Get the ntheta and nphi from the particle that is currently being looked at. This will be used to identify its ray and calculate its direction. */
    DMSwarmGetField(radSearch, IdentifierField, nullptr, nullptr, (void**)&identifiers) >> utilities::PetscUtilities::checkError;
    DMSwarmGetField(radSearch, VirtualCoordField, nullptr, nullptr, (void**)&virtualcoords) >> utilities::PetscUtilities::checkError;
    DMSwarmGetField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> utilities::PetscUtilities::checkError;

    for (PetscInt ipart = 0; ipart < npoints; ipart++) {
        /** Check that the particle is in a valid region */
        if (index[ipart] >= 0 && subDomain.InRegion(index[ipart])) {
            auto& identifier = identifiers[ipart];
            // Exact the ray to reduce lookup
            auto& ray = raySegments[identifier.remoteRayId];

            /** ********************************************
             * The face stepping routine will give the precise path length of the mesh without any error. It will also allow the faces of the cells to be accounted for so that the
             * boundary conditions and the conditions at reflection can be accounted for. This will make the entire initialization much faster by only requiring a single step through each
             * cell. Additionally, the option for reflection is opened because the faces and their normals are now more easily accessed during the initialization. In the future, the carrier
             * particles may want to be given some information that the boundary label carries when the search particle happens upon it so that imperfect reflection can be implemented.
             * */

            /** Step 1: Register the current cell index in the rays vector. The physical coordinates that have been set in the previous step / loop will be immediately registered.
             * Because the ray comes from the origin, all of the cell indexes are naturally ordered from the center out
             * */
            auto& raySegment = ray.emplace_back();
            raySegment.cell = index[ipart];

            /** Step 2: Acquire the intersection of the particle search line with the segment or face. In the case if a two dimensional mesh, the virtual coordinate in the z direction will
             * need to be solved for because the three dimensional line will not have a literal intersection with the segment of the cell. The third coordinate can be solved for in this case.
             * Here we are figuring out what distance the ray spends inside the cell that it has just registered.
             * */
            /** March over each face on this cell in order to check them for the one which intersects this ray next */
            PetscInt numberFaces;
            const PetscInt* cellFaces;
            DMPlexGetConeSize(subDomain.GetDM(), index[ipart], &numberFaces) >> utilities::PetscUtilities::checkError;
            DMPlexGetCone(subDomain.GetDM(), index[ipart], &cellFaces) >> utilities::PetscUtilities::checkError;  //!< Get the face geometry associated with the current cell

            /** Check every face for intersection with the segment.
             * The segment with the shortest path length for intersection will be the one that physically intercepts with the cell face and not with the nonphysical plane beyond the face.
             * */
            for (PetscInt f = 0; f < numberFaces; f++) {
                PetscInt face = cellFaces[f];
                DMPlexPointLocalRead(faceDM, face, faceGeomArray, &faceGeom) >> utilities::PetscUtilities::checkError;  //!< Reads the cell location from the current cell

                /** Get the intersection of the direction vector with the cell face
                 * Use the plane equation and ray segment equation in order to get the face intersection with the shortest path length
                 * This will be the next position of the search particle
                 * */
                PetscReal path = FaceIntersect(ipart, virtualcoords, faceGeom);  //!< Use plane intersection equation by getting the centroid and normal vector of the face

                /** Step 3: Take this path if it is shorter than the previous one, getting the shortest path.
                 * The path should never be zero if the forwardIntersect check is functioning properly.
                 * */
                if (path > 0) {
                    virtualcoords[ipart].hhere = (virtualcoords[ipart].hhere == 0) ? (path * 1.1) : virtualcoords[ipart].hhere;  //!< Dumb check to ensure that the path length is always updated
                    if (virtualcoords[ipart].hhere > path) {
                        virtualcoords[ipart].hhere = path;  //!> Get the shortest path length of all of the faces. The point must be in the direction that the ray is travelling in order to be valid.
                    }
                }
            }
            virtualcoords[ipart].hhere = (virtualcoords[ipart].hhere == 0) ? minCellRadius : virtualcoords[ipart].hhere;
            raySegment.pathLength = virtualcoords[ipart].hhere;
        } else {
            virtualcoords[ipart].hhere = (virtualcoords[ipart].hhere == 0) ? minCellRadius : virtualcoords[ipart].hhere;
        }
    }
    DMSwarmRestoreField(radSearch, IdentifierField, nullptr, nullptr, (void**)&identifiers) >> utilities::PetscUtilities::checkError;
    DMSwarmRestoreField(radSearch, VirtualCoordField, nullptr, nullptr, (void**)&virtualcoords) >> utilities::PetscUtilities::checkError;
    DMSwarmRestoreField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> utilities::PetscUtilities::checkError;
}

void ablate::radiation::Radiation::EvaluateGains(Vec solVec, ablate::domain::Field temperatureField, Vec auxVec) {
    StartEvent("Radiation::EvaluateGains");

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

    // Get access to the absorption function
    auto absorptivityFunctionContext = absorptivityFunction.context.get();

    // Start by marching over all rays in this rank
    for (std::size_t raySegmentIndex = 0; raySegmentIndex < raySegments.size(); ++raySegmentIndex) {
        //! Zero this ray segment for all wavelengths
        for (unsigned short int wavelengthIndex = 0; wavelengthIndex < absorptivityFunction.propertySize; wavelengthIndex++) {  //! Iterate through every wavelength entry in this ray segment
            raySegmentsCalculations[absorptivityFunction.propertySize * raySegmentIndex + wavelengthIndex].Ij = 0.0;
            raySegmentsCalculations[absorptivityFunction.propertySize * raySegmentIndex + wavelengthIndex].Krad = 1.0;
        }

        // compute the Ij and Krad for this segment starting at the point closest to the ray origin
        const auto& raySegment =
            raySegments[raySegmentIndex];  //! This is allowed to be cast to auto and indexed raySegmentIndex because there is only one physical ray segment that we are reading from.
        for (const auto& cellSegment : raySegment) {
            const PetscReal* sol = nullptr;          //!< The solution value at any given location
            const PetscReal* temperature = nullptr;  //!< The temperature at any given location
            DMPlexPointLocalRead(solDm, cellSegment.cell, solArray, &sol);
            if (sol) {
                DMPlexPointLocalFieldRead(auxDm, cellSegment.cell, temperatureField.id, auxArray, &temperature);
                if (temperature) {               /** Input absorptivity (kappa) values from model here. */
                    PetscReal kappa[absorptivityFunction.propertySize];  //!< Absorptivity coefficient, property of each cell. This is an array that we will iterate through for every evaluation
                    absorptivityFunction.function(sol, *temperature, kappa, absorptivityFunctionContext);
                    //! Get the pointer to the returned array of absorption values. Iterate through every wavelength for the evaluation.
                    if (cellSegment.pathLength < 0) {
                        // This is a boundary cell
                        for (int wavelengthIndex = 0; wavelengthIndex < absorptivityFunction.propertySize; ++wavelengthIndex) {
                            raySegmentsCalculations[absorptivityFunction.propertySize * raySegmentIndex + wavelengthIndex].Ij +=
                                FlameIntensity(1.0, *temperature) * raySegmentsCalculations[absorptivityFunction.propertySize * raySegmentIndex + wavelengthIndex].Krad;
                        }
                    } else {
                        for (int wavelengthIndex = 0; wavelengthIndex < absorptivityFunction.propertySize; ++wavelengthIndex) {
                            // This is not a boundary cell
                            raySegmentsCalculations[absorptivityFunction.propertySize * raySegmentIndex + wavelengthIndex].Ij +=
                                FlameIntensity(1 - exp(-kappa[wavelengthIndex] * cellSegment.pathLength), *temperature) * raySegmentsCalculations[absorptivityFunction.propertySize * raySegmentIndex + wavelengthIndex].Krad;

                            // Compute the total absorption for this domain
                            raySegmentsCalculations[absorptivityFunction.propertySize * raySegmentIndex + wavelengthIndex].Krad *= exp(-kappa[wavelengthIndex] * cellSegment.pathLength);
                        }
                    }
                }
            }
        }
    }

    // Now that all the ray information is computed, transfer it back to rank that originated each ray using a pull
    PetscSFBcastBegin(remoteAccess, carrierMpiType, (const void*)raySegmentsCalculations.data(), (void*)raySegmentSummary.data(), MPI_REPLACE) >> utilities::PetscUtilities::checkError;
    PetscSFBcastEnd(remoteAccess, carrierMpiType, (const void*)raySegmentsCalculations.data(), (void*)raySegmentSummary.data(), MPI_REPLACE) >> utilities::PetscUtilities::checkError;

    /** March over each
     * INDEXING ANNOTATIONS:
     * evaluatedGains: There is a gain evaluation for every cell * wavelength
     *  Therefore, the indexing of the gain evaluation is [absorptivityFunction.propertySize * cell + i] because the cells are looped through outside of the wavelengths
     * raySegmentsPerOriginRay: This only stores the number of segments in each ray. There is no reason to index this with wavelength.
     *  Therefore, the indexing is [rayOffset], where the ray refers to the wavelength independent ray count.
     * raySegmentSummary: This will store a value for every ray segment and wavelength. Each ray will integrate its ray segments together for every wavelength.
     * */
    std::size_t segmentOffset = 0;
    std::size_t rayOffset = 0;
    for (PetscInt cellIndex = 0; cellIndex < numberOriginCells; ++cellIndex) {
        for (unsigned short int wavelengthIndex = 0; wavelengthIndex < absorptivityFunction.propertySize; ++wavelengthIndex)
            evaluatedGains[absorptivityFunction.propertySize * cellIndex + wavelengthIndex] = 0.0;  //! Zero the evaluated gains for this ray specifically. Do this for all wavelengths.
        for (PetscInt rayIndex = 0; rayIndex < raysPerCell; ++rayIndex) {
            // Add the black body radiation transmitted through the domain to the source term
            PetscReal iSource[absorptivityFunction.propertySize];
            for (unsigned short int i = 0; i < absorptivityFunction.propertySize; ++i) iSource[i] = 0.0;  //! Initialize the wavelength dependent arrays to be of size zero.

            // Add the absorption for this domain to the total absorption of the ray
            PetscReal kRadd[absorptivityFunction.propertySize];
            for (unsigned short int i = 0; i < absorptivityFunction.propertySize; ++i) kRadd[i] = 1.0;  //! Initialize the wavelength dependent arrays to be of size zero.

            /** for each segment in this ray
             * Integrate the wavelength dependent intensity calculation for each segment for each wavelength
             * We need to store all of the wavelength results on the final evaluation of the cell
             * Therefore, we should first iterate through the wavelengths first and sum the effects of every wavelength on every cell.
             */
            for (unsigned short int s = 0; s < raySegmentsPerOriginRay[rayOffset]; ++s) {
                for (unsigned short int wavelengthIndex = 0; wavelengthIndex < absorptivityFunction.propertySize; wavelengthIndex++) {
                    iSource[wavelengthIndex] += raySegmentSummary[absorptivityFunction.propertySize * segmentOffset + wavelengthIndex].Ij * kRadd[wavelengthIndex];
                    kRadd[wavelengthIndex] *= raySegmentSummary[absorptivityFunction.propertySize * segmentOffset + wavelengthIndex].Krad;
                }
                segmentOffset++;
            }

            for (unsigned short int wavelengthIndex = 0; wavelengthIndex < absorptivityFunction.propertySize; wavelengthIndex++)
                evaluatedGains[absorptivityFunction.propertySize * cellIndex + wavelengthIndex] += iSource[wavelengthIndex] * gainsFactor[rayOffset];
            rayOffset++;
        }
    }

    /** Cleanup */
    VecRestoreArrayRead(solVec, &solArray);
    VecRestoreArrayRead(auxVec, &auxArray);
    EndEvent();
}

void ablate::radiation::Radiation::DeleteOutOfBounds(ablate::domain::SubDomain& subDomain) {
    PetscReal* coord;
    PetscInt* index;                    //!< Pointer to the coordinate field information
    struct Virtualcoord* virtualcoord;  //!< Pointer to the primary (virtual) coordinate field information
    struct Identifier* identifier;      //!< Pointer to the ray identifier information

    /** Get the fields associated with the particle swarm so that they can be modified */
    DMSwarmGetField(radSearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> utilities::PetscUtilities::checkError;
    DMSwarmGetField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> utilities::PetscUtilities::checkError;
    DMSwarmGetField(radSearch, IdentifierField, nullptr, nullptr, (void**)&identifier) >> utilities::PetscUtilities::checkError;
    DMSwarmGetField(radSearch, VirtualCoordField, nullptr, nullptr, (void**)&virtualcoord) >> utilities::PetscUtilities::checkError;

    PetscInt npoints = 0;
    DMSwarmGetLocalSize(radSearch, &npoints) >> utilities::PetscUtilities::checkError;  //!< Recalculate the number of particles that are in the domain
    PetscMPIInt rank = 0;
    MPI_Comm_rank(subDomain.GetComm(), &rank);

    /**  */
    for (PetscInt ipart = 0; ipart < npoints; ipart++) {
        //!< If the particles that were just created are sitting in the boundary cell of the face that they belong to, delete them
        if (!(region->InRegion(region, subDomain.GetDM(), index[ipart]))) {  //!< If the particle location index and boundary cell index are the same, then they should be deleted
            DMSwarmRestoreField(radSearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> utilities::PetscUtilities::checkError;
            DMSwarmRestoreField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> utilities::PetscUtilities::checkError;
            DMSwarmRestoreField(radSearch, IdentifierField, nullptr, nullptr, (void**)&identifier) >> utilities::PetscUtilities::checkError;
            DMSwarmRestoreField(radSearch, VirtualCoordField, nullptr, nullptr, (void**)&virtualcoord) >> utilities::PetscUtilities::checkError;

            DMSwarmRemovePointAtIndex(radSearch, ipart);  //!< Delete the particle!
            DMSwarmGetLocalSize(radSearch, &npoints);

            DMSwarmGetField(radSearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> utilities::PetscUtilities::checkError;
            DMSwarmGetField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> utilities::PetscUtilities::checkError;
            DMSwarmGetField(radSearch, IdentifierField, nullptr, nullptr, (void**)&identifier) >> utilities::PetscUtilities::checkError;
            DMSwarmGetField(radSearch, VirtualCoordField, nullptr, nullptr, (void**)&virtualcoord) >> utilities::PetscUtilities::checkError;
            ipart--;  //!< Check the point replacing the one that was deleted
        }
    }

    /** Restore the fields associated with the particles */
    DMSwarmRestoreField(radSearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> utilities::PetscUtilities::checkError;
    DMSwarmRestoreField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> utilities::PetscUtilities::checkError;
    DMSwarmRestoreField(radSearch, IdentifierField, nullptr, nullptr, (void**)&identifier) >> utilities::PetscUtilities::checkError;
    DMSwarmRestoreField(radSearch, VirtualCoordField, nullptr, nullptr, (void**)&virtualcoord) >> utilities::PetscUtilities::checkError;
}

std::ostream& ablate::radiation::operator<<(std::ostream& os, const ablate::radiation::Radiation::Identifier& id) {
    os << "Identifier origin(" << id.originRank << ": " << id.originRayId << ") remote(" << id.remoteRank << ": " << id.remoteRayId << ") nSeg:" << id.nSegment;

    return os;
}

#include "registrar.hpp"
REGISTER_DEFAULT(ablate::radiation::Radiation, ablate::radiation::Radiation, "A solver for radiative heat transfer in participating media", ARG(std::string, "id", "the name of the flow field"),
                 ARG(ablate::domain::Region, "region", "the region to apply this solver."), ARG(int, "rays", "number of rays used by the solver"),
                 ARG(ablate::eos::radiationProperties::RadiationModel, "properties", "the radiation properties model"),
                 OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));
