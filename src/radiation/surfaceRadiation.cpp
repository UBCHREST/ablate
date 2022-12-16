#include "surfaceRadiation.hpp"

ablate::radiation::SurfaceRadiation::SurfaceRadiation(const std::string& solverId, const std::shared_ptr<domain::Region>& region, const PetscInt raynumber,
                                                      std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, std::shared_ptr<ablate::monitors::logs::Log> log)
    : Radiation(solverId, region, raynumber, radiationModelIn, log) {
}

ablate::radiation::SurfaceRadiation::~SurfaceRadiation() {
}

void ablate::radiation::SurfaceRadiation::Initialize(const solver::Range& cellRange, ablate::domain::SubDomain& subDomain) { /** Declare some information associated with the field declarations */
    StartEvent("SurfaceRadiation::Initialize");
    PetscReal* coord;
    PetscInt* index;                    //!< Pointer to the coordinate field information
    struct Virtualcoord* virtualcoord;  //!< Pointer to the primary (virtual) coordinate field information
    struct Identifier* identifier;      //!< Pointer to the ray identifier information

    /** Get the fields associated with the particle swarm so that they can be modified */
    DMSwarmGetField(radSearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> checkError;
    DMSwarmGetField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> checkError;
    DMSwarmGetField(radSearch, "identifier", nullptr, nullptr, (void**)&identifier) >> checkError;
    DMSwarmGetField(radSearch, "virtual coord", nullptr, nullptr, (void**)&virtualcoord) >> checkError;

    PetscInt npoints = 0;
    DMSwarmGetLocalSize(radSearch, &npoints) >> checkError;  //!< Recalculate the number of particles that are in the domain
    PetscMPIInt rank = 0;
    MPI_Comm_rank(subDomain.GetComm(), &rank);

    /**  */
    for (PetscInt ipart = 0; ipart < npoints; ipart++) {
        //!< If the particles that were just created are sitting in the boundary cell of the face that they belong to, delete them
        if (!(region->InRegion(region, subDomain.GetDM(), index[ipart]))) {  //!< If the particle location index and boundary cell index are the same, then they should be deleted
            DMSwarmRestoreField(radSearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> checkError;
            DMSwarmRestoreField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> checkError;
            DMSwarmRestoreField(radSearch, "identifier", nullptr, nullptr, (void**)&identifier) >> checkError;
            DMSwarmRestoreField(radSearch, "virtual coord", nullptr, nullptr, (void**)&virtualcoord) >> checkError;

            DMSwarmRemovePointAtIndex(radSearch, ipart);  //!< Delete the particle!
            DMSwarmGetLocalSize(radSearch, &npoints);

            DMSwarmGetField(radSearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> checkError;
            DMSwarmGetField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> checkError;
            DMSwarmGetField(radSearch, "identifier", nullptr, nullptr, (void**)&identifier) >> checkError;
            DMSwarmGetField(radSearch, "virtual coord", nullptr, nullptr, (void**)&virtualcoord) >> checkError;
            ipart--;  //!< Check the point replacing the one that was deleted
        }
    }

    /** Restore the fields associated with the particles */
    DMSwarmRestoreField(radSearch, DMSwarmPICField_coor, nullptr, nullptr, (void**)&coord) >> checkError;
    DMSwarmRestoreField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> checkError;
    DMSwarmRestoreField(radSearch, "identifier", nullptr, nullptr, (void**)&identifier) >> checkError;
    DMSwarmRestoreField(radSearch, "virtual coord", nullptr, nullptr, (void**)&virtualcoord) >> checkError;

    EndEvent();
    ablate::radiation::Radiation::Initialize(cellRange, subDomain);
}

PetscReal ablate::radiation::SurfaceRadiation::SurfaceComponent(DM faceDM, const PetscScalar* faceGeomArray, PetscInt iCell, PetscInt nphi, PetscInt ntheta) {
    PetscFVFaceGeom* faceGeom;
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
    PetscReal ldotn = abs(((sin(thetalocal) * cos(phi)) * faceNormx) + ((sin(thetalocal) * sin(phi)) * faceNormy) + (cos(thetalocal) * faceNormz));
    return ldotn;
}

PetscInt ablate::radiation::SurfaceRadiation::GetLossCell(PetscInt iCell, PetscReal& losses, DM faceDm, DM cellDm) {
    losses = 0.5;  //!< Cut the losses in half if this is a surface implementation
    PetscInt numberNeighborCells;
    const PetscInt* neighborCells;
    PetscInt index = -1;

    DMPlexGetSupportSize(faceDm, iCell, &numberNeighborCells) >> ablate::checkError;  //!< Get the cells on each side of this face to check for boundary cells
    DMPlexGetSupport(faceDm, iCell, &neighborCells) >> ablate::checkError;
    for (PetscInt n = 0; n < numberNeighborCells; n++) {
        PetscInt cell = neighborCells[n];  //!< Contains the cell indexes of the neighbor cells
        if (!(region->InRegion(region, cellDm, cell))) {
            index = cell;  //!< Take the index of the cell that is a boundary cell.
        }
    }
    if (index == -1) throw std::runtime_error("Loss cell error!");
    return index;
}

void ablate::radiation::SurfaceRadiation::GetFuelEmissivity(double& kappa) { kappa = 1; }

#include "registrar.hpp"
REGISTER(ablate::radiation::Radiation, ablate::radiation::SurfaceRadiation, "A solver for radiative heat transfer in participating media", ARG(std::string, "id", "the name of the flow field"),
         ARG(ablate::domain::Region, "region", "the region to apply this solver."), ARG(int, "rays", "number of rays used by the solver"),
         ARG(ablate::eos::radiationProperties::RadiationModel, "properties", "the radiation properties model"), OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));