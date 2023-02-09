#ifndef ABLATELIBRARY_ORTHOGONALRADIATION_HPP
#define ABLATELIBRARY_ORTHOGONALRADIATION_HPP

#include "radiation.hpp"

namespace ablate::radiation {

class OrthogonalRadiation : public ablate::radiation::Radiation {
   public:
    OrthogonalRadiation(const std::string& solverId, const std::shared_ptr<domain::Region>& region, std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn,
                        std::shared_ptr<ablate::monitors::logs::Log> = {});
    ~OrthogonalRadiation();

    void Setup(const solver::Range& cellRange, ablate::domain::SubDomain& subDomain) override;

    // TODO: Maybe overide the GetIntensity function here so that the spherical or surface characteristics are not accidentally captured in the implementation?
};
}  // namespace ablate::radiation

#endif  // ABLATELIBRARY_ORTHOGONALRADIATION_HPP
