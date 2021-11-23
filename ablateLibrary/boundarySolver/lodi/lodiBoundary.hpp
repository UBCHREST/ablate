#ifndef ABLATELIBRARY_LODIBOUNDARY_HPP
#define ABLATELIBRARY_LODIBOUNDARY_HPP

#include "boundarySolver/boundaryProcess.hpp"
#include "eos/eos.hpp"

namespace ablate::boundarySolver::lodi {
class LODIBoundary : public BoundaryProcess {
   protected:
    const std::shared_ptr<eos::EOS> eos;

    void GetVelAndCPrims(PetscReal velNorm, PetscReal speedOfSound, PetscReal Cp, PetscReal Cv, PetscReal &velNormPrim, PetscReal &speedOfSoundPrim);

   public:
    explicit LODIBoundary(std::shared_ptr<eos::EOS> eos);

    void Initialize(ablate::boundarySolver::BoundarySolver &bSolver) override = 0;
};

}  // namespace ablate::boundarySolver::lodi
#endif  // ABLATELIBRARY_LODIBOUNDARY_HPP
