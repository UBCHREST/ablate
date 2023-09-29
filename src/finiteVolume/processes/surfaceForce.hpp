#ifndef ABLATELIBRARY_FINITEVOLUME_CHEMISTRY_HPP
#define ABLATELIBRARY_FINITEVOLUME_CHEMISTRY_HPP

#include <petsc.h>
#include <memory>
#include <vector>
#include "domain/range.hpp"
#include "domain/RBF/rbf.hpp"
#include "domain/reverseRange.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "process.hpp"
#include "solver/solver.hpp"
#include "twoPhaseEulerAdvection.hpp"

namespace ablate::finiteVolume::processes {

  class SurfaceForce : public Process {


    private:

    PetscReal sigma;



    public:


    explicit SurfaceForce(PetscReal sigma);


    ~SurfaceForce() override;

    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) override;

    /**
     * static function private function to compute surface force and add source to eulerset
     * @param solver
     * @param dm - DM of the cell-centered data
     * @param time - Current time of data in locX
     * @param locX - Local vector containing current solution
     * @param fVec - Vector to store the Cell-centered body-force
     * @param ctx - Pointer to ablate::finiteVolume::processes::SurfaceForce
     * @return
     */
    static PetscErrorCode ComputeSource(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx);
  };
}  // namespace ablate::finiteVolume::processes
#endif
