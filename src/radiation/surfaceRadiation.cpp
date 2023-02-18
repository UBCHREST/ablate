#include "surfaceRadiation.hpp"

ablate::radiation::SurfaceRadiation::SurfaceRadiation(const std::string& solverId, const std::shared_ptr<domain::Region>& region, const PetscInt raynumber,
                                                      std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, std::shared_ptr<ablate::monitors::logs::Log> log)
    : Radiation(solverId, region, raynumber, radiationModelIn, log) {}

ablate::radiation::SurfaceRadiation::~SurfaceRadiation() {}

void ablate::radiation::SurfaceRadiation::Initialize(const solver::Range& cellRange, ablate::domain::SubDomain& subDomain) { /** Declare some information associated with the field declarations */
    StartEvent("SurfaceRadiation::Initialize");
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

    EndEvent();
    ablate::radiation::Radiation::Initialize(cellRange, subDomain);

    // set up the reverse lookup for faces
    indexLookup = ablate::domain::ReverseRange(cellRange);
}

PetscReal ablate::radiation::SurfaceRadiation::SurfaceComponent(const PetscReal normal[], PetscInt iCell, PetscInt nphi, PetscInt ntheta) {
    /** Now that we are iterating over every ray identifier in this local domain, we can get all of the particles that are associated with this ray.
     * We will need to sort the rays in order of domain segment. We need to start at the end of the ray and go towards the beginning of the ray. */
    PetscReal faceNormNormalized = sqrt((normal[0] * normal[0]) + (normal[1] * normal[1]) + (normal[2] * normal[2]));
    PetscReal faceNormx = normal[0] / faceNormNormalized;  //!< Get the normalized face normal (not area scaled)
    PetscReal faceNormy = normal[1] / faceNormNormalized;
    PetscReal faceNormz = normal[2] / faceNormNormalized;
    /** Update the direction vector of the search particle */
    PetscReal phi = ((double)nphi / (double)nPhi) * 2.0 * ablate::utilities::Constants::pi;
    PetscReal thetalocal = (((double)ntheta + 0.5) / (double)nTheta) * ablate::utilities::Constants::pi;
    PetscReal ldotn = abs(((sin(thetalocal) * cos(phi)) * faceNormx) + ((sin(thetalocal) * sin(phi)) * faceNormy) + (cos(thetalocal) * faceNormz));
    return ldotn;
}

#include "registrar.hpp"
REGISTER_DERIVED(ablate::radiation::Radiation, ablate::radiation::SurfaceRadiation);
REGISTER(ablate::radiation::SurfaceRadiation, ablate::radiation::SurfaceRadiation, "A solver for radiative heat transfer in participating media", ARG(std::string, "id", "the name of the flow field"),
         ARG(ablate::domain::Region, "region", "the region to apply this solver."), ARG(int, "rays", "number of rays used by the solver"),
         ARG(ablate::eos::radiationProperties::RadiationModel, "properties", "the radiation properties model"), OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));
