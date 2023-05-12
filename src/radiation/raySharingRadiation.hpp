#ifndef ABLATELIBRARY_RAYSHARINGRADIATION_H
#define ABLATELIBRARY_RAYSHARINGRADIATION_H

#include "domain/reverseRange.hpp"
#include "radiation.hpp"

namespace ablate::radiation {

class RaySharingRadiation : public ablate::radiation::Radiation {
   public:
    RaySharingRadiation(const std::string& solverId, const std::shared_ptr<domain::Region>& region, const PetscInt raynumber,
                        std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, std::shared_ptr<ablate::monitors::logs::Log> = {});
    ~RaySharingRadiation();

    void Setup(const ablate::domain::Range& cellRange, ablate::domain::SubDomain& subDomain) override;

    void IdentifyNewRaysOnRank(ablate::domain::SubDomain& subDomain, DM radReturn, PetscInt npoints);

    void ParticleStep(ablate::domain::SubDomain& subDomain, DM faceDM, const PetscScalar* faceGeomArray, DM radReturn, PetscInt nlocalpoints,
                      PetscInt nglobalpoints) override;  //!< Routine to move the particle one step

    static inline std::string GetClassType() { return "RaySharingRadiation"; }

   protected:
    //! used to look up from the cell id to range index
    ablate::domain::ReverseRange indexLookup;

    //! the indexes mapping to the ray id
    std::vector<PetscReal> remoteMap;
};

}  // namespace ablate::radiation

#endif  // ABLATELIBRARY_RAYSHARINGRADIATION_H
