#ifndef ABLATELIBRARY_LODIBOUNDARY_HPP
#define ABLATELIBRARY_LODIBOUNDARY_HPP

#include "boundarySolver/boundaryProcess.hpp"
#include "eos/eos.hpp"
#include "finiteVolume/processes/flowProcess.hpp"

namespace ablate::boundarySolver::lodi {
class LODIBoundary : public BoundaryProcess {
   protected:
    typedef enum {
        RHO = finiteVolume::processes::FlowProcess::RHO,
        RHOE = finiteVolume::processes::FlowProcess::RHOE,
        RHOVELN = finiteVolume::processes::FlowProcess::RHOU,
        RHOVELT1 = finiteVolume::processes::FlowProcess::RHOV,
        RHOVELT2 = finiteVolume::processes::FlowProcess::RHOW
    } BoundaryEulerComponents;

    const std::shared_ptr<eos::EOS> eos;

    static void GetVelAndCPrims(PetscReal velNorm, PetscReal speedOfSound, PetscReal Cp, PetscReal Cv, PetscReal& velNormPrim, PetscReal& speedOfSoundPrim);

    static void GetEigenValues(PetscInt ndims, PetscInt nSpec, PetscInt nEV, PetscReal veln, PetscReal c, PetscReal velnprm, PetscReal cprm, PetscReal lamda[]);

    static void GetmdFdn(PetscInt ndims, PetscInt neqs, PetscInt nspeceq, PetscInt nEVeq, const PetscReal* velNormCord, PetscReal rho, PetscReal T, PetscReal Cp, PetscReal Cv, PetscReal C,
                         PetscReal Enth, PetscReal velnprm, PetscReal Cprm, const PetscReal* Yi, const PetscReal* EV, const PetscReal* sL, const PetscReal transformationMatrix[3][3],
                         PetscReal* mdFdn);

   public:
    explicit LODIBoundary(std::shared_ptr<eos::EOS> eos);

    void Initialize(ablate::boundarySolver::BoundarySolver& bSolver) override = 0;
};

}  // namespace ablate::boundarySolver::lodi
#endif  // ABLATELIBRARY_LODIBOUNDARY_HPP
