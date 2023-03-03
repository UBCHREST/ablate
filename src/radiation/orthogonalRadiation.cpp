#include "orthogonalRadiation.hpp"

ablate::radiation::OrthogonalRadiation::OrthogonalRadiation(const std::string& solverId, const std::shared_ptr<domain::Region>& region,
                                                            std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, std::shared_ptr<ablate::monitors::logs::Log> log)
    : SurfaceRadiation(solverId, region, 0, radiationModelIn, log) {}  //! The ray number should never be used because there is only one ray emanating from every boundary face

ablate::radiation::OrthogonalRadiation::~OrthogonalRadiation() {}

void ablate::radiation::OrthogonalRadiation::Setup(const ablate::domain::Range& cellRange, ablate::domain::SubDomain& subDomain) {
    dim = subDomain.GetDimensions();   //!< Number of dimensions already defined in the setup
    nTheta = (dim == 1) ? 1 : nTheta;  //!< Reduce the number of rays if one dimensional symmetry can be taken advantage of

    /** Begins radiation properties model
     * Runs the ray initialization, finding cell indices
     * Initialize the log if provided
     */
    absorptivityFunction = radiationModel->GetAbsorptionPropertiesTemperatureFunction(eos::radiationProperties::RadiationProperty::Absorptivity, subDomain.GetFields());

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

    StartEvent("OrthogonalRadiation::Setup");
    if (log) log->Printf("Starting Initialize\n");

    DMPlexGetMinRadius(subDomain.GetDM(), &minCellRadius) >> utilities::PetscUtilities::checkError;

    /** do a simple sanity check for labels */
    PetscMPIInt rank = 0;
    MPI_Comm_rank(subDomain.GetComm(), &rank);  //!< Get the origin rank of the current process. The particle belongs to this rank. The rank only needs to be read once.

    /** Setup the particles and their associated fields including: origin domain/ ray identifier / # domains crossed, and coordinates. Instantiate ray particles for each local cell only. */
    numberOriginCells = (cellRange.end - cellRange.start);
    raysPerCell = 2;
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
    gainsFactor.resize(numberOriginRays);

    //!< Initialize a counter to represent the particle index. This will be iterated every time that the inner loop is passed through.
    PetscInt ipart = 0;

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {            //!< This will iterate only though local cells
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;  //!< Isolates the valid cells
        PetscReal centroid[3];
        PetscReal normal[3] = {0.0, 0.0, 0.0};
        DMPlexComputeCellGeometryFVM(subDomain.GetDM(), iCell, nullptr, centroid, normal) >> utilities::PetscUtilities::checkError;

        /** Since we don't know which side of the face is inside the region, we can just spawn two particles on opposite sides of the face and give them both the same gains factor weighting.
         * One of them will be on the correct side and one if them will not. Whichever one is not will be deleted.
         * The iteration represents a loop where i is the coefficient of the normal vector.
         * It is equal to -1 on the first iteration and equal to 1 on the second iteration. This will put a particle on both sides of the normal vector.
         * */
        for (int i = -1; i < 2; i = i + 2) {
            /** Update the direction vector of the search particle */
            virtualcoord[ipart].xdir = (dim > 0) ? i * normal[0] : 0;  //!< x component direction from the cell face normal
            virtualcoord[ipart].ydir = (dim > 1) ? i * normal[1] : 0;  //!< y component direction from the cell face normal
            virtualcoord[ipart].zdir = (dim > 2) ? i * normal[2] : 0;  //!< z component direction from the cell face normal

            /** Get the particle coordinate field and write the cellGeom->centroid[xyz] into it */
            virtualcoord[ipart].x = centroid[0] + (virtualcoord[ipart].xdir * 0.1 * minCellRadius);  //!< Offset from the centroid slightly so they sit in a cell if they are on its face.
            virtualcoord[ipart].y = centroid[1] + (virtualcoord[ipart].ydir * 0.1 * minCellRadius);
            virtualcoord[ipart].z = centroid[2] + (virtualcoord[ipart].zdir * 0.1 * minCellRadius);

            // Init hhere to default value
            virtualcoord[ipart].hhere = 0.0;

            /** Update the physical coordinate field so that the real particle location can be updated. */
            /** Update the physical coordinate field so that the real particle location can be updated. */
            UpdateCoordinates(ipart, virtualcoord, coord, 0.0);  //! adv value of 0.0 places the particle exactly where the virtual coordinates are.

            /** Label the particle with the ray identifier. With what is known at this point */
            identifier[ipart].originRank = rank;    //!< Input the ray identifier. This location scheme represents stepping four entries for every particle index increase
            identifier[ipart].originRayId = ipart;  //! This serve to identify the ray id on the origin
            identifier[ipart].remoteRank = PETSC_DECIDE;
            identifier[ipart].remoteRayId = PETSC_DECIDE;
            identifier[ipart].nSegment = -1;

            // Compute the intensityFactor
            gainsFactor[ipart] = ablate::utilities::Constants::pi;  //! Intensity of the irradiation at the surface is pi * intensity integrated.

            /** Set the index of the field value so that it can be written to for every particle */
            ipart++;  //!< Must be iterated at the end since the value is initialized at zero.}
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

#include "registrar.hpp"
REGISTER_DERIVED(ablate::radiation::SurfaceRadiation, ablate::radiation::OrthogonalRadiation);
REGISTER(ablate::radiation::OrthogonalRadiation, ablate::radiation::OrthogonalRadiation, "A solver for radiative heat transfer in participating media",
         ARG(std::string, "id", "the name of the flow field"), ARG(ablate::domain::Region, "region", "the boundary region to apply this solver."),
         ARG(ablate::eos::radiationProperties::RadiationModel, "properties", "the radiation properties model"), OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));
