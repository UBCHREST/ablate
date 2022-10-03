#ifndef ABLATELIBRARY_SURFACERADIATION_HPP
#define ABLATELIBRARY_SURFACERADIATION_HPP

#include "radiation.hpp"
#include "utilities/constants.hpp"

namespace ablate::radiation {

class SurfaceRadiation : public ablate::radiation::Radiation {
   public:
    void Initialize(const solver::Range& cellRange, ablate::domain::SubDomain& subDomain) override;
    PetscReal SurfaceComponent(DM faceDM, const PetscScalar* faceGeomArray, PetscFVFaceGeom* faceGeom, PetscInt iCell, PetscInt nphi, PetscInt ntheta) override;
};
}  // namespace ablate::radiation
#endif  // ABLATELIBRARY_SURFACERADIATION_HPP
