#ifndef ABLATELIBRARY_ORTHOGONALRADIATION_HPP
#define ABLATELIBRARY_ORTHOGONALRADIATION_HPP

#include "domain/range.hpp"
#include "radiation.hpp"
#include "surfaceRadiation.hpp"

namespace ablate::radiation {

class OrthogonalRadiation : public ablate::radiation::SurfaceRadiation {
   private:
    //! used to look up from the face id to range index
    solver::ReverseRange indexLookup;

   public:
    OrthogonalRadiation(const std::string& solverId, const std::shared_ptr<domain::Region>& region, std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, int num = 1,
                        std::shared_ptr<ablate::monitors::logs::Log> = {});
    ~OrthogonalRadiation();

    void Setup(const ablate::domain::Range& cellRange, ablate::domain::SubDomain& subDomain) override;
};
}  // namespace ablate::radiation
#endif  // ABLATELIBRARY_ORTHOGONALRADIATION_HPP
