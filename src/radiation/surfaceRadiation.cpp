#include "surfaceRadiation.hpp"

ablate::radiation::SurfaceRadiation::SurfaceRadiation(const std::string& solverId, const std::shared_ptr<domain::Region>& region, const PetscInt raynumber,
                                                      std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, std::shared_ptr<ablate::monitors::logs::Log> log)
    : Radiation(solverId, region, raynumber, radiationModelIn, log) {}

ablate::radiation::SurfaceRadiation::~SurfaceRadiation() {}

void ablate::radiation::SurfaceRadiation::Initialize(const solver::Range& cellRange, ablate::domain::SubDomain& subDomain) { /** Declare some information associated with the field declarations */
    StartEvent("SurfaceRadiation::Initialize");

    DeleteOutOfBounds(subDomain);

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
         ARG(ablate::eos::radiationProperties::RadiationModel, "properties", "the radiation properties model"),
         OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));