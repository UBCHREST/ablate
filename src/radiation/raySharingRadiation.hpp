#ifndef ABLATELIBRARY_RAYSHARINGRADIATION_HPP
#define ABLATELIBRARY_RAYSHARINGRADIATION_HPP

#include "radiation.hpp"

namespace ablate::radiation {

class RaySharingRadiation : public ablate::radiation::Radiation {
   public:
    void ParticleStep(ablate::domain::SubDomain& subDomain, PetscSF cellSF, DM faceDM, const PetscScalar* faceGeomArray, PetscInt stepcount) override;
};
}  // namespace ablate::radiation
#endif  // ABLATELIBRARY_RAYSHARINGRADIATION_HPP
