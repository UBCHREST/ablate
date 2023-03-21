#ifndef ABLATELIBRARY_ORTHOGONALRADIATION_HPP
#define ABLATELIBRARY_ORTHOGONALRADIATION_HPP

#include "radiation.hpp"
#include "solver/reverseRange.hpp"
#include "surfaceRadiation.hpp"

namespace ablate::radiation {

class OrthogonalRadiation : public ablate::radiation::SurfaceRadiation {
   public:
    OrthogonalRadiation(const std::string& solverId, const std::shared_ptr<domain::Region>& region, std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn,
                        std::shared_ptr<ablate::monitors::logs::Log> = {});
    ~OrthogonalRadiation();

    void Setup(const ablate::domain::Range& cellRange, ablate::domain::SubDomain& subDomain) override;
};
}  // namespace ablate::radiation

#endif  // ABLATELIBRARY_ORTHOGONALRADIATION_HPP
