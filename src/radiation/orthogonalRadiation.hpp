#ifndef ABLATELIBRARY_ORTHOGONALRADIATION_HPP
#define ABLATELIBRARY_ORTHOGONALRADIATION_HPP

#include "domain/reverseRange.hpp"
#include "radiation.hpp"
#include "surfaceRadiation.hpp"

namespace ablate::radiation {

class OrthogonalRadiation : public ablate::radiation::SurfaceRadiation {
   public:
    OrthogonalRadiation(const std::string& solverId, const std::shared_ptr<domain::Region>& region, std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn,
                        std::shared_ptr<ablate::monitors::logs::Log> = {});
    ~OrthogonalRadiation();

    void Setup(const ablate::domain::Range& cellRange, ablate::domain::SubDomain& subDomain) override;

    /**
     * Represents the name of the class for logging and other utilities
     * @return
     */
    static inline std::string GetClassType() { return "OrthogonalRadiation"; }

    inline void GetSurfaceIntensity(PetscReal* intensityReturn, PetscInt faceId, PetscReal temperature, PetscReal emissivity = 1.0) override {
        for (int i = 0; i < (int)absorptivityFunction.propertySize; ++i) {  // Compute the losses
            PetscReal netIntensity = evaluatedGains[absorptivityFunction.propertySize * indexLookup.GetAbsoluteIndex(faceId) + i];
            intensityReturn[i] = abs(netIntensity) > ablate::utilities::Constants::large ? ablate::utilities::Constants::large * PetscSignReal(netIntensity) : netIntensity;
        }
    }
};
}  // namespace ablate::radiation
#endif  // ABLATELIBRARY_ORTHOGONALRADIATION_HPP
