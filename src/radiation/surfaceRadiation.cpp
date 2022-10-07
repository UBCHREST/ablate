#include "surfaceRadiation.hpp"

ablate::radiation::SurfaceRadiation::SurfaceRadiation(const std::string& solverId, const std::shared_ptr<domain::Region>& region, std::shared_ptr<domain::Region> fieldBoundary,
                                                      const PetscInt raynumber, std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn,
                                                      std::shared_ptr<ablate::monitors::logs::Log> log)
    : Radiation(solverId, region, fieldBoundary, raynumber, radiationModelIn, log) {
    nTheta = raynumber;    //!< The number of angles to solve with, given by user input
    nPhi = 2 * raynumber;  //!< The number of angles to solve with, given by user input
}

ablate::radiation::SurfaceRadiation::~SurfaceRadiation() {
    if (radsolve) DMDestroy(&radsolve) >> checkError;  //!< Destroy the radiation particle swarm
    VecDestroy(&faceGeomVec) >> checkError;
    VecDestroy(&cellGeomVec) >> checkError;
}

void ablate::radiation::SurfaceRadiation::Initialize(const solver::Range& cellRange, ablate::domain::SubDomain& subDomain) { /** Declare some information associated with the field declarations */
    PetscReal* coord;                                                                                                        //!< Pointer to the coordinate field information
    struct Virtualcoord* virtualcoord;                                                                                       //!< Pointer to the primary (virtual) coordinate field information
    struct Identifier* identifier;                                                                                           //!< Pointer to the ray identifier information

    /** Get the fields associated with the particle swarm so that they can be modified */
    DMSwarmGetField(radsearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> checkError;
    DMSwarmGetField(radsearch, "identifier", nullptr, nullptr, (void**)&identifier) >> checkError;
    DMSwarmGetField(radsearch, "virtual coord", nullptr, nullptr, (void**)&virtualcoord) >> checkError;

    PetscInt npoints = 0;
    DMSwarmGetLocalSize(radsearch, &npoints) >> checkError;  //!< Recalculate the number of particles that are in the domain
    PetscMPIInt rank = 0;
    MPI_Comm_rank(subDomain.GetComm(), &rank);

    /** Iterate over the particles that are present in the domain
     * Add the cell index to the ray
     * Step every particle in the domain one step and then perform a migration
     * */
    Vec intersect;
    VecCreate(PETSC_COMM_SELF, &intersect) >> checkError;  //!< Instantiates the vector
    VecSetBlockSize(intersect, dim) >> checkError;
    VecSetSizes(intersect, PETSC_DECIDE, npoints * dim) >> checkError;  //!< Set size
    VecSetFromOptions(intersect) >> checkError;
    PetscInt i[3] = {0, 1, 2};                   //!< Establish the vector here so that it can be iterated.
    for (PetscInt ip = 0; ip < npoints; ip++) {  //!< Iterate over the particles present in the domain.
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

    PetscInt numberNeighborCells;
    const PetscInt* neighborCells;

    /** Delete all of the particles that were transported to their origin domains -> Delete if the particle has travelled to get here and isn't native */
    for (PetscInt ipart = 0; ipart < npoints; ipart++) {
        DMPlexGetSupportSize(subDomain.GetDM(), identifier[ipart].iCell, &numberNeighborCells) >> ablate::checkError;  //!< Get the cells on each side of this face to check for boundary cells
        DMPlexGetSupport(subDomain.GetDM(), identifier[ipart].iCell, &neighborCells) >> ablate::checkError;
        PetscInt index = -1;  //!< Index value to compare the Locate Points result against.
        for (PetscInt n = 0; n < numberNeighborCells; n++) {
            PetscInt bcell = neighborCells[n];  //!< Contains the cell indexes of the neighbor cells
            if (fieldBoundary->InRegion(fieldBoundary, subDomain.GetDM(), bcell)) {
                index = bcell;
            }
        }

        if (index == -1) throw std::invalid_argument("SurfaceRadiation must be given the same boundary cell region as its boundary solver!"); //!< Throw an error if the boundary region is incorrect

        //!< If the particles that were just created are sitting in the boundary cell of the face that they belong to, delete them
        if (index == cell[ipart].index) {  //!< If the particle location index and boundary cell index are the same, then they should be deleted
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
    }

    /** Restore the fields associated with the particles */
    DMSwarmRestoreField(radsearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> checkError;
    DMSwarmRestoreField(radsearch, "identifier", nullptr, nullptr, (void**)&identifier) >> checkError;
    DMSwarmRestoreField(radsearch, "virtual coord", nullptr, nullptr, (void**)&virtualcoord) >> checkError;

    ablate::radiation::Radiation::Initialize(cellRange, subDomain);
}

PetscReal ablate::radiation::SurfaceRadiation::SurfaceComponent(DM* faceDM, const PetscScalar* faceGeomArray, PetscInt iCell, PetscInt nphi, PetscInt ntheta) {
    PetscFVFaceGeom* faceGeom;
    DMPlexPointLocalRead(*(faceDM), iCell, faceGeomArray, &faceGeom) >> checkError;
    /** Now that we are iterating over every ray identifier in this local domain, we can get all of the particles that are associated with this ray.
     * We will need to sort the rays in order of domain segment. We need to start at the end of the ray and go towards the beginning of the ray. */
    PetscReal faceNormNormalized = sqrt((faceGeom->normal[0] * faceGeom->normal[0]) + (faceGeom->normal[1] * faceGeom->normal[1]) + (faceGeom->normal[2] * faceGeom->normal[2]));
    PetscReal faceNormx = faceGeom->normal[0] / faceNormNormalized;  //!< Get the normalized face normal (not area scaled)
    PetscReal faceNormy = faceGeom->normal[1] / faceNormNormalized;
    PetscReal faceNormz = faceGeom->normal[2] / faceNormNormalized;
    /** Update the direction vector of the search particle */
    PetscReal phi = ((double)nphi / (double)nPhi) * 2.0 * ablate::utilities::Constants::pi;
    PetscReal thetalocal = (((double)ntheta + 0.5) / (double)nTheta) * ablate::utilities::Constants::pi;
    PetscReal ldotn = abs(((sin(thetalocal) * cos(phi)) * faceNormx) + ((sin(thetalocal) * sin(phi)) * faceNormy) + (cos(thetalocal) * faceNormz));
    return ldotn;
}

PetscInt ablate::radiation::SurfaceRadiation::GetLossCell(PetscInt iCell, PetscReal& losses, DM* solDm) {
    losses = 0.5;  //!< Cut the losses in half if this is a surface implementation
    PetscInt numberNeighborCells;
    const PetscInt* neighborCells;
    PetscInt index = -1;

    DMPlexGetSupportSize(*(solDm), iCell, &numberNeighborCells) >> ablate::checkError;  //!< Get the cells on each side of this face to check for boundary cells
    DMPlexGetSupport(*(solDm), iCell, &neighborCells) >> ablate::checkError;
    for (PetscInt n = 0; n < numberNeighborCells; n++) {
        PetscInt cell = neighborCells[n];  //!< Contains the cell indexes of the neighbor cells
        if (fieldBoundary->InRegion(fieldBoundary, *(solDm), cell)) {
            index = cell; //!< Take the index of the cell that is a boundary cell.
        }
    }
    if (index == -1) throw std::runtime_error("I don't know what's going on");
    return index;
}

#include "registrar.hpp"
REGISTER(ablate::radiation::Radiation, ablate::radiation::SurfaceRadiation, "A solver for radiative heat transfer in participating media", ARG(std::string, "id", "the name of the flow field"),
         ARG(ablate::domain::Region, "region", "the region to apply this solver."), ARG(ablate::domain::Region, "fieldBoundary", "boundary of the radiation region"),
         ARG(int, "rays", "number of rays used by the solver"), ARG(ablate::eos::radiationProperties::RadiationModel, "properties", "the radiation properties model"),
         OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));