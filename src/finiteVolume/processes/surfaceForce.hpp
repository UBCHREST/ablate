#ifndef ABLATELIBRARY_FINITEVOLUME_CHEMISTRY_HPP
#define ABLATELIBRARY_FINITEVOLUME_CHEMISTRY_HPP

#include <petsc.h>
#include <memory>
#include <vector>
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "process.hpp"
#include "solver/range.hpp"
#include "solver/solver.hpp"
#include "twoPhaseEulerAdvection.hpp"

namespace ablate::finiteVolume::processes {

class SurfaceForce : public Process {
    PetscReal sigma;

   private:
    /**
     * struct to hold the vortex stencil
     */
    struct VertexStencil {
        /** The points in the stencil*/
        std::vector<PetscInt> stencil;
        /** The size of stencil*/
        PetscInt stencilSize;
        /** The point*/
        PetscInt vertexId;
        /** The weights */
        std::vector<PetscScalar> gradientWeights;
        /** Coordinate of the vertex */
        std::vector<PetscScalar> stencilCoord;
    };
    DM dmData;

   public:
    // Hold a list of VortexStencils
    std::vector<VertexStencil> vertexStencils;

    explicit SurfaceForce(PetscReal sigma);

    /**
     * public function to link this process with the flow
     * a@param flow
     */
    ~SurfaceForce() override;

    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;
    /**
     * static function private function to compute surface force and add source to eulerset
     * @param solver
     * @param dm
     * @param time
     * @param locX
     * @param fVec
     * @param ctx
     * @return
     */
    static PetscErrorCode ComputeSource(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx);
};
}  // namespace ablate::finiteVolume::processes
#endif