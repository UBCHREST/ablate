#ifndef ABLATELIBRARY_SURFACERADIATION_HPP
#define ABLATELIBRARY_SURFACERADIATION_HPP

#include "radiation.hpp"
#include "utilities/constants.hpp"

namespace ablate::radiation {

class SurfaceRadiation : public ablate::radiation::Radiation {
   public:
    SurfaceRadiation(const std::string& solverId, const std::shared_ptr<domain::Region>& region, std::shared_ptr<domain::Region> fieldBoundary, const PetscInt raynumber,
                     std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, std::shared_ptr<ablate::monitors::logs::Log> = {});
    ~SurfaceRadiation();

    void Initialize(const solver::Range& cellRange, ablate::domain::SubDomain& subDomain) override;
    PetscReal SurfaceComponent(DM* faceDM, const PetscScalar* faceGeomArray, PetscInt iCell, PetscInt nphi, PetscInt ntheta) override;
    PetscInt GetLossCell(PetscInt iCell, PetscReal& losses, DM* solDm, DM* pPDm) override;
};
}  // namespace ablate::radiation
#endif  // ABLATELIBRARY_SURFACERADIATION_HPP
