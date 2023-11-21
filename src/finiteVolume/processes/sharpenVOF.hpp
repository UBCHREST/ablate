#ifndef ABLATELIBRARY_SHARPENVOF_HPP
#define ABLATELIBRARY_SHARPENVOF_HPP

#include <petsc.h>
#include <memory>
#include <vector>
#include "flowProcess.hpp"



#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "process.hpp"
#include "solver/solver.hpp"
#include "twoPhaseEulerAdvection.hpp"

namespace ablate::finiteVolume::processes {

class SharpenVOF : public FlowProcess {
   private:
    const std::vector<PetscReal> tol;

    /**
     * private function to compute gravity source
     * @return
     */
     static PetscErrorCode SharpenInterface(ablate::finiteVolume::FiniteVolumeSolver *fv, TS ts, PetscReal time, bool initialStage, Vec locX, void *ctx);

   public:
    explicit SharpenVOF(const std::vector<PetscReal> tol);

    ~SharpenVOF() override;

    void Setup(ablate::finiteVolume::FiniteVolumeSolver& fv) override;
};

}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_SHARPENVOF_HPP
